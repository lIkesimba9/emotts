import os
import json
import copy
import math
from collections import OrderedDict

import random
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import Normal
from typing import List, Tuple
from torch.nn import functional as f

from .utils import get_mask_from_lengths, pad

from src.models.fastspeech2.config import VariancePredictorParams, VarianceAdaptorParams, DurationParams, GaussianUpsampleParams, RangeParams


class Idomp(nn.Module):
    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return x

class LinearWithActivation(torch.nn.Module):

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        w_init_gain: str = "linear",
        activation: nn.Module = Idomp(),
    ):
        super().__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight, gain=torch.nn.init.calculate_gain(w_init_gain)
        )
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear_layer(x))


class PositionalEncoding(nn.Module):
    def __init__(self, dimension: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.dimension = dimension


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        position = torch.arange(x.shape[1]).unsqueeze(1).to(x.device)
        div_term = torch.exp(
            torch.arange(0, self.dimension, 2) * (-math.log(10000.0) / self.dimension)
        ).to(x.device)
        pe: torch.Tensor = torch.zeros(1, x.shape[1], self.dimension).to(x.device)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        pe = torch.repeat_interleave(pe, x.shape[0], 0)

        x = x + pe[: x.shape[0]]
        return self.dropout(x)

class VarianceAdaptor(nn.Module):
    """Variance Adaptor"""

    def __init__(self, config: VarianceAdaptorParams, pitch_min: float, pitch_max: float, 
            energy_min: float, energy_max: float, encoder_hidden: int):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = VariancePredictor(config.predictor_params, encoder_hidden)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(config.predictor_params, encoder_hidden)
        self.energy_predictor = VariancePredictor(config.predictor_params, encoder_hidden)


        pitch_quantization = config.embedding_params.pitch_quantization
        energy_quantization = config.embedding_params.energy_quantization
        n_bins = config.embedding_params.n_bins
        assert pitch_quantization in ["linear", "log"]
        assert energy_quantization in ["linear", "log"]


        if pitch_quantization == "log":
            self.pitch_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(pitch_min), np.log(pitch_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.pitch_bins = nn.Parameter(
                torch.linspace(pitch_min, pitch_max, n_bins - 1),
                requires_grad=False,
            )
        if energy_quantization == "log":
            self.energy_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(energy_min), np.log(energy_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.energy_bins = nn.Parameter(
                torch.linspace(energy_min, energy_max, n_bins - 1),
                requires_grad=False,
            )

        self.pitch_embedding = nn.Embedding(
            n_bins, encoder_hidden
        )
        self.energy_embedding = nn.Embedding(
            n_bins, encoder_hidden
        )

    def get_pitch_embedding(self, x, target, mask, control):
        prediction = self.pitch_predictor(x, mask)
        if target is not None:
            embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
        else:
            prediction = prediction * control
            embedding = self.pitch_embedding(
                torch.bucketize(prediction, self.pitch_bins)
            )
        return prediction, embedding

    def get_energy_embedding(self, x, target, mask, control):
        prediction = self.energy_predictor(x, mask)
        if target is not None:
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            prediction = prediction * control
            embedding = self.energy_embedding(
                torch.bucketize(prediction, self.energy_bins)
            )
        return prediction, embedding

    def forward(self, x, src_mask, mel_mask, max_len, pitch_target, energy_target, duration_target,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):

        log_duration_prediction = self.duration_predictor(x, src_mask)

        pitch_prediction, pitch_embedding = self.get_pitch_embedding(
            x, pitch_target, src_mask, p_control
        )
        x = x + pitch_embedding

        energy_prediction, energy_embedding = self.get_energy_embedding(
            x, energy_target, src_mask, e_control
        )
        x = x + energy_embedding

        x, _ = self.length_regulator(x, duration_target, max_len)

        return (
            x,
            pitch_prediction,
            energy_prediction,
            log_duration_prediction,
            mel_mask,
        )
    
    def inference(self, x, src_mask, p_control=1.0, e_control=1.0, d_control=1.0):
        log_duration_prediction = self.duration_predictor(x, src_mask)

        _, pitch_embedding = self.get_pitch_embedding(
            x, None, src_mask, p_control
        )
        x = x + pitch_embedding

        _, energy_embedding = self.get_energy_embedding(
            x, None, src_mask, e_control
        )
        x = x + energy_embedding

        duration_rounded = torch.clamp(
            (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
            min=0,
        )
        x, mel_len = self.length_regulator(x, duration_rounded, None)
        mel_mask = get_mask_from_lengths(mel_len, torch.max(mel_len).item(), mel_len.device)

        return (
            x,
            mel_len,
            mel_mask,
        )






class DurationPredictor(nn.Module):
    def __init__(self, embedding_dim: int, config: DurationParams):
        super().__init__()

        self.lstm = nn.LSTM(
            embedding_dim,
            config.lstm_hidden,
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = config.dropout
        self.projection = LinearWithActivation(config.lstm_hidden * 2, 1, bias=False)

    def forward(self, x: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True, enforce_sorted=False
        )
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(packed_x)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        outputs = f.dropout(outputs, self.dropout, self.training)
        x = self.projection(outputs)
        return x


class RangePredictor(nn.Module):
    def __init__(self, embedding_dim: int, config: RangeParams):
        super().__init__()

        self.lstm = nn.LSTM(
            embedding_dim + 1,
            config.lstm_hidden,
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = config.dropout
        self.projection = LinearWithActivation(
            config.lstm_hidden * 2, 1, activation=nn.Softplus()
        )

    def forward(
        self, x: torch.Tensor, durations: torch.Tensor, input_lengths: torch.Tensor
    ) -> torch.Tensor:

        x = torch.cat((x, durations), dim=-1)
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True, enforce_sorted=False
        )
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(packed_x)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        outputs = f.dropout(outputs, self.dropout, self.training)
        outputs = self.projection(outputs)
        return outputs


class Attention(nn.Module):
    def __init__(self, embedding_dim: int, config: GaussianUpsampleParams):
        super().__init__()
        self.teacher_forcing_ratio = config.teacher_forcing_ratio
        self.eps = torch.Tensor([config.eps])
        self.dropout = config.attention_dropout
        self.duration_predictor = DurationPredictor(
            embedding_dim, config.duration_config
        )
        self.range_predictor = RangePredictor(embedding_dim, config.range_config)
        self.positional_encoder = PositionalEncoding(
            config.positional_dim, dropout=config.positional_dropout
        )

    def calc_scores(
        self, durations: torch.Tensor, ranges: torch.Tensor
    ) -> torch.Tensor:
        # Calc gaussian weight for Gaussian upsampling attention
        duration_cumsum = durations.cumsum(dim=1).float()
        max_duration = duration_cumsum[:, -1, :].max().long()
        mu = duration_cumsum - 0.5 * durations
        ranges = torch.maximum(ranges, self.eps.to(ranges.device))
        distr = Normal(mu, ranges)

        t = torch.arange(0, max_duration.item()).view(1, 1, -1).to(ranges.device)

        weights: torch.Tensor = f.softmax(distr.log_prob(t), dim=1)

        return weights

    def forward(
        self,
        embeddings: torch.Tensor,
        input_lengths: torch.Tensor,
        y_durations: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        input_lengths = input_lengths.cpu().numpy()

        durations = self.duration_predictor(embeddings, input_lengths)
        ranges = self.range_predictor(embeddings, durations, input_lengths)

        if random.uniform(0, 1) > self.teacher_forcing_ratio:  # type: ignore
            scores = self.calc_scores(durations, ranges)
        else:
            scores = self.calc_scores(y_durations.unsqueeze(2), ranges)

        embeddings_per_duration = torch.matmul(scores.transpose(1, 2), embeddings)
        embeddings_per_duration = self.positional_encoder(embeddings_per_duration)
        return durations.squeeze(2), embeddings_per_duration

    def inference(
        self, embeddings: torch.Tensor, input_lengths: torch.Tensor
    ) -> torch.Tensor:

        durations = self.duration_predictor(embeddings, input_lengths)
        ranges = self.range_predictor(embeddings, durations, input_lengths)
        scores = self.calc_scores(durations, ranges)

        embeddings_per_duration = torch.matmul(scores.transpose(1, 2), embeddings)
        embeddings_per_duration = self.positional_encoder(embeddings_per_duration)
        return embeddings_per_duration, durations.squeeze(2)



class VarianceAdaptorGaus(nn.Module):
    """Variance Adaptor"""

    def __init__(self, config: VarianceAdaptorParams, pitch_min: float, pitch_max: float, 
            energy_min: float, energy_max: float, encoder_hidden: int):
        super(VarianceAdaptorGaus, self).__init__()
        #self.duration_predictor = VariancePredictor(config.predictor_params, encoder_hidden)
        #self.length_regulator = LengthRegulator()
        self.attention = Attention(encoder_hidden, config.attention_config)
        self.pitch_predictor = VariancePredictor(config.predictor_params, encoder_hidden)
        self.energy_predictor = VariancePredictor(config.predictor_params, encoder_hidden)


        pitch_quantization = config.embedding_params.pitch_quantization
        energy_quantization = config.embedding_params.energy_quantization
        n_bins = config.embedding_params.n_bins
        assert pitch_quantization in ["linear", "log"]
        assert energy_quantization in ["linear", "log"]


        if pitch_quantization == "log":
            self.pitch_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(pitch_min), np.log(pitch_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.pitch_bins = nn.Parameter(
                torch.linspace(pitch_min, pitch_max, n_bins - 1),
                requires_grad=False,
            )
        if energy_quantization == "log":
            self.energy_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(energy_min), np.log(energy_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.energy_bins = nn.Parameter(
                torch.linspace(energy_min, energy_max, n_bins - 1),
                requires_grad=False,
            )

        self.pitch_embedding = nn.Embedding(
            n_bins, encoder_hidden
        )
        self.energy_embedding = nn.Embedding(
            n_bins, encoder_hidden
        )

    def get_pitch_embedding(self, x, target, mask, control):
        prediction = self.pitch_predictor(x, mask)
        if target is not None:
            embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
        else:
            prediction = prediction * control
            embedding = self.pitch_embedding(
                torch.bucketize(prediction, self.pitch_bins)
            )
        return prediction, embedding

    def get_energy_embedding(self, x, target, mask, control):
        prediction = self.energy_predictor(x, mask)
        if target is not None:
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            prediction = prediction * control
            embedding = self.energy_embedding(
                torch.bucketize(prediction, self.energy_bins)
            )
        return prediction, embedding

    def forward(self, x, src_mask, mel_mask, max_len, pitch_target, energy_target, duration_target, num_phonemes,
        p_control=1.0,
        e_control=1.0,  
    ):

      
        pitch_prediction, pitch_embedding = self.get_pitch_embedding(
            x, pitch_target, src_mask, p_control
        )

        energy_prediction, energy_embedding = self.get_energy_embedding(
            x, energy_target, src_mask, e_control
        )
        
        x = x + pitch_embedding
        x = x + energy_embedding
        durations, attented_embeddings = self.attention(
            x, num_phonemes, duration_target
        )

        
        log_duration_prediction = torch.log(durations + 1)
        return (
            attented_embeddings,
            pitch_prediction,
            energy_prediction,
            log_duration_prediction,
            mel_mask,
        )
    
    def inference(self, x, src_mask, num_phonemes, p_control=1.0, e_control=1.0, d_control=1.0):
        #log_duration_prediction = self.duration_predictor(x, src_mask)

        _, pitch_embedding = self.get_pitch_embedding(
            x, None, src_mask, p_control
        )


        _, energy_embedding = self.get_energy_embedding(
            x, None, src_mask, e_control
        )
        x = x + pitch_embedding
        x = x + energy_embedding
        attented_embeddings, durations = self.attention.inference(x, num_phonemes.to("cpu"))
   
        mel_lens = durations.sum(dim=-1).long()
        mel_mask = get_mask_from_lengths(mel_lens, torch.max(mel_lens).item(), mel_lens.device)

        return (
            attented_embeddings,
            mel_lens,
            mel_mask,
        )



class LengthRegulator(nn.Module):
    """Length Regulator"""

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(duration.device)

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len


class VariancePredictor(nn.Module):
    """Duration, Pitch and Energy Predictor"""

    def __init__(self, conifg: VariancePredictorParams, encoder_hidden: int):
        super(VariancePredictor, self).__init__()

        self.input_size = encoder_hidden
        self.filter_size = conifg.filter_size
        self.kernel = conifg.kernel_size
        self.conv_output_size = conifg.filter_size
        self.dropout = conifg.dropout

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=1,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init="linear",
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x



class VarianceAdaptorDurationOnly(nn.Module):
    """Variance Adaptor"""

    def __init__(self, config: VarianceAdaptorParams, pitch_min: float, pitch_max: float, 
            energy_min: float, energy_max: float, encoder_hidden: int):
        super(VarianceAdaptorDurationOnly, self).__init__()
        self.duration_predictor = VariancePredictor(config.predictor_params, encoder_hidden)
        self.length_regulator = LengthRegulator()


    def forward(self, x, src_mask, mel_mask, max_len, pitch_target, energy_target, duration_target,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):

        log_duration_prediction = self.duration_predictor(x, src_mask)

        x, _ = self.length_regulator(x, duration_target, max_len)

        return (
            x,
            log_duration_prediction,
            mel_mask,
        )
    
    def inference(self, x, src_mask, p_control=1.0, e_control=1.0, d_control=1.0):
        log_duration_prediction = self.duration_predictor(x, src_mask)

        duration_rounded = torch.clamp(
            (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
            min=0,
        )
        x, mel_len = self.length_regulator(x, duration_rounded, None)
        mel_mask = get_mask_from_lengths(mel_len, torch.max(mel_len).item(), mel_len.device)

        return (
            x,
            mel_len,
            mel_mask,
        )