import math
import random
from typing import List, Tuple

import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as f
import numpy as np

from src.data_process import RegularBatch
from src.data_process.voiceprint_dataset import VoicePrintBatch
from src.data_process.voiceprint_variance_adaptor_dataset import VoicePrintVarianceBatch

from .config import (
    DecoderParams,
    DurationParams,
    EncoderParams,
    GSTParams,
    GaussianUpsampleParams,
    ModelParams,
    PostNetParams,
    RangeParams,
)
from .gst import GST
from .layers import Conv1DNorm, LinearWithActivation, PositionalEncoding, IdompSecond
from .utils import get_mask_from_lengths, norm_emb_layer

from src.models.fastspeech2.modules import VariancePredictor, VarianceAdaptorParams, LengthRegulator


class Prenet(nn.Module):
    def __init__(self, in_dim: int, sizes: List[int], dropout: float):
        super().__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [
                LinearWithActivation(
                    in_size,
                    out_size,
                    bias=False,
                    activation=nn.ReLU(),
                )
                for (in_size, out_size) in zip(in_sizes, sizes)
            ]
        )
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for linear in self.layers:
            x = f.dropout(f.relu(linear(x)), p=self.dropout, training=self.training)
        return x


class Postnet(nn.Module):
    def __init__(self, n_mel_channels: int, config: PostNetParams):
        super().__init__()
        self.dropout = config.dropout
        convolutions: List[nn.Module] = [
            Conv1DNorm(
                n_mel_channels,
                config.embedding_dim,
                kernel_size=config.kernel_size,
                stride=1,
                padding=int((config.kernel_size - 1) / 2),
                dilation=1,
                dropout_rate=config.dropout,
                w_init_gain="tanh",
            ),
            nn.Tanh(),
        ]

        for _ in range(config.n_convolutions - 2):
            convolutions.append(
                Conv1DNorm(
                    config.embedding_dim,
                    config.embedding_dim,
                    kernel_size=config.kernel_size,
                    stride=1,
                    padding=int((config.kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="tanh",
                )
            )
            convolutions.append(nn.Tanh())

        convolutions.append(
            Conv1DNorm(
                config.embedding_dim,
                n_mel_channels,
                kernel_size=config.kernel_size,
                stride=1,
                padding=int((config.kernel_size - 1) / 2),
                dilation=1,
                w_init_gain="linear",
            )
        )
        self.convolutions = nn.Sequential(*convolutions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.convolutions) - 1):
            x = torch.tanh(self.convolutions[i](x))
        x = self.convolutions[-1](x)

        return x


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
        max_duration = torch.ceil(duration_cumsum[:, -1, :].max()).long()
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

    def inference_with_durations(
        self, embeddings: torch.Tensor, input_lengths: torch.Tensor
    ) -> torch.Tensor:

        durations = self.duration_predictor(embeddings, input_lengths)
        ranges = self.range_predictor(embeddings, durations, input_lengths)
        scores = self.calc_scores(durations, ranges)

        embeddings_per_duration = torch.matmul(scores.transpose(1, 2), embeddings)
        embeddings_per_duration = self.positional_encoder(embeddings_per_duration)
        return embeddings_per_duration, durations

    def inference(self, embeddings: torch.Tensor, input_lengths: torch.Tensor
    ) -> torch.Tensor:
        embeddings_per_duration, durations = self.inference_with_durations(embeddings, input_lengths)
        return embeddings_per_duration


class Encoder(nn.Module):
    def __init__(self, phonem_embedding_dim: int, config: EncoderParams):
        super().__init__()

        convolutions: List[nn.Module] = [
            Conv1DNorm(
                phonem_embedding_dim,
                config.conv_channel,
                kernel_size=config.kernel_size,
                stride=1,
                padding=int((config.kernel_size - 1) / 2),
                dilation=1,
                dropout_rate=config.dropout,
                w_init_gain="relu",
            )
        ]

        for _ in range(config.n_convolutions - 2):
            conv_layer = Conv1DNorm(
                config.conv_channel,
                config.conv_channel,
                kernel_size=config.kernel_size,
                stride=1,
                padding=int((config.kernel_size - 1) / 2),
                dilation=1,
                w_init_gain="relu",
            )
            convolutions.append(conv_layer)

        convolutions.append(
            Conv1DNorm(
                config.conv_channel,
                phonem_embedding_dim,
                kernel_size=config.kernel_size,
                stride=1,
                padding=int((config.kernel_size - 1) / 2),
                dilation=1,
                w_init_gain="relu",
            ),
        )
        self.convolutions = nn.Sequential(*convolutions)
        self.lstm = nn.LSTM(
            phonem_embedding_dim,
            config.lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(
        self, phonem_emb: torch.Tensor, input_lengths: torch.Tensor
    ) -> torch.Tensor:

        phonem_emb = self.convolutions(phonem_emb)
        phonem_emb = phonem_emb.transpose(1, 2)
        phonem_emb_packed = nn.utils.rnn.pack_padded_sequence(
            phonem_emb, input_lengths, batch_first=True, enforce_sorted=False
        )

        self.lstm.flatten_parameters()
        phonem_emb_packed, _ = self.lstm(phonem_emb_packed)

        phonem_emb, _ = nn.utils.rnn.pad_packed_sequence(
            phonem_emb_packed, batch_first=True
        )

        return phonem_emb


class Decoder(nn.Module):
    def __init__(
        self,
        n_mel_channels: int,
        n_frames_per_step: int,
        attention_out_dim: int,
        config: DecoderParams,
    ):
        super().__init__()
        self.n_mel_channels = n_mel_channels
        self.decoder_rnn_dim = config.decoder_rnn_dim
        self.teacher_forcing_ratio = config.teacher_forcing_ratio
        self.p_decoder_dropout = config.dropout
        self.n_frames_per_step = n_frames_per_step

        self.prenet = Prenet(
            self.n_mel_channels,
            config.prenet_layers,
            config.prenet_dropout,
        )

        self.decoder_rnn = nn.LSTM(
            config.prenet_layers[-1] * self.n_frames_per_step + attention_out_dim,
            config.decoder_rnn_dim,
            num_layers=config.decoder_num_layers,
            bidirectional=False,
            batch_first=True,
        )

        self.linear_projection = LinearWithActivation(
            config.decoder_rnn_dim + attention_out_dim,
            n_mel_channels * self.n_frames_per_step,
        )

    def forward(self, memory: torch.Tensor, y_mels: torch.Tensor) -> torch.Tensor:
        ## memory == encoder outputs passed through attention for each decoder step
        ## decoder_input_dim == attention_output_dim + prenet_output_dim
        ## memory: [ batch_size, n_decoder_steps, attention_output_dim ] 
        batch_size = memory.shape[0]
        mels_view_size = self.n_mel_channels * self.n_frames_per_step
        init_previous_frame = torch.zeros(batch_size, 1, mels_view_size).to(
            memory.device
        )
        n_decoder_steps = memory.shape[1]
        padded_size = (
            n_decoder_steps * mels_view_size
        )
        ## NOTE: Right now the durations accumulated error leads 
        ##          to the size of mels lagging behind the durations,
        ##          so we need to extend the mels with zeros... (e.g. 38 vs 40 --> 6-frames difference)
        padded_memory = memory
        
        if (self.n_frames_per_step > 1):
            ## one decoder step less: to be fed as previous decoder step (in teacher forcing mode)
            ## NOTE: once we move to previous frame being the size of one frame, this should be changed
            to_get = (n_decoder_steps-1)*self.n_frames_per_step ##int((padded_size - mels_view_size)/self.n_mel_channels)
            ## NOTE: no padding here, because memory shape should exactly match the number of decoder steps
            ##                                                                  and not the number of frames
            padded_y_mels_previous = torch.zeros(y_mels.shape[0], to_get, y_mels.shape[2]).to(memory.device)
            if (padded_y_mels_previous.shape[1] > y_mels.shape[1]):
                padded_y_mels_previous[:, :y_mels.shape[1], :] = y_mels
            else:
                ## y_mels is not padded here, which is why padded_y_mels_previous may be shorter than y_mels,
                ## since it may be the result of rounding down to the previous multiple of n_frames_per_step
                padded_y_mels_previous = y_mels[:, :padded_y_mels_previous.shape[1]] 
            padded_y_mels_previous = padded_y_mels_previous.reshape(batch_size, -1, mels_view_size)
        else:
            padded_y_mels_previous = y_mels
        padded_y_mels_previous = torch.cat((init_previous_frame, padded_y_mels_previous), dim=1)

        ## NOTE: same. no reshaping here, because memory shape should exactly match the number of decoder steps
        ##                                                                          and not the number of frames
        init_previous_frame = init_previous_frame[:, 0, :]

        mel_outputs = []
        decoder_state = None
        previous_frame = init_previous_frame

        for j in range(n_decoder_steps):
            previous_frame = self.prenet(
                previous_frame.view(batch_size, -1, self.n_mel_channels)
            )
            decoder_input: torch.Tensor = torch.cat(
                (previous_frame.view(batch_size, -1), padded_memory[:, j, :]), dim=-1
            )
            out, decoder_state = self.decoder_rnn(
                decoder_input.unsqueeze(1), decoder_state
            )
            out = torch.cat((out, padded_memory[:, j, :].unsqueeze(1)), dim=-1)
            mel_out = self.linear_projection(out)
            mel_outputs.append(mel_out)
            if random.uniform(0, 1) > self.teacher_forcing_ratio:
                previous_frame = mel_out.squeeze(1)
            else:
                previous_frame = padded_y_mels_previous[:, min(j+1, padded_y_mels_previous.shape[1]-1), :]

        mel_tensor_outputs: torch.Tensor = torch.cat(mel_outputs, dim=1)
        mel_tensor_outputs = mel_tensor_outputs.reshape(
            batch_size, -1, self.n_mel_channels
        )
        return mel_tensor_outputs

    def inference(self, memory: torch.Tensor) -> torch.Tensor:

        batch_size = memory.shape[0]
        mels_view_size = self.n_mel_channels * self.n_frames_per_step
        init_previous_frame = torch.zeros(
            memory.shape[0], mels_view_size
        ).to(memory.device)
        n_decoder_steps = memory.shape[1]
        padded_size = (
            n_decoder_steps * self.n_frames_per_step
        )
        padded_memory = memory

        mel_outputs = []
        decoder_state = None
        previous_frame = init_previous_frame

        for i in range(n_decoder_steps):
            previous_frame = self.prenet(
                previous_frame.view(batch_size, -1, self.n_mel_channels)
            )
            decoder_input: torch.Tensor = torch.cat(
                (previous_frame.view(batch_size, -1), padded_memory[:, i, :]), dim=-1
            )
            out, decoder_state = self.decoder_rnn(
                decoder_input.unsqueeze(1), decoder_state
            )
            out = torch.cat((out, padded_memory[:, i, :].unsqueeze(1)), dim=-1)
            mel_out = self.linear_projection(out)
            mel_outputs.append(mel_out)
            previous_frame = mel_out.squeeze(1)

        mel_tensor_outputs: torch.Tensor = torch.cat(mel_outputs, dim=1)
        mel_tensor_outputs = mel_tensor_outputs.reshape(
            batch_size, -1, self.n_mel_channels
        )
        return mel_tensor_outputs


class NonAttentiveTacotron(nn.Module):
    def __init__(
        self,
        n_phonems: int,
        n_speakers: int,
        n_mel_channels: int,
        config: ModelParams,
        gst_config: GSTParams,
        finetune: bool,
    ):
        super().__init__()

        self.n_frames_per_step = config.n_frames_per_step

        full_embedding_dim = (
            config.phonem_embedding_dim
            + config.speaker_embedding_dim
            + gst_config.emb_dim
        )
        self.finetune = finetune
        self.gst_emb_dim = gst_config.emb_dim
        self.phonem_embedding = nn.Embedding(
            n_phonems, config.phonem_embedding_dim, padding_idx=0
        )
        self.speaker_embedding = nn.Embedding(
            n_speakers,
            config.speaker_embedding_dim,
        )
        norm_emb_layer(
            self.phonem_embedding,
            n_phonems,
            config.phonem_embedding_dim,
        )
        norm_emb_layer(
            self.speaker_embedding,
            n_speakers,
            config.speaker_embedding_dim,
        )
        self.encoder = Encoder(
            config.phonem_embedding_dim,
            config.encoder_config,
        )
        self.attention = Attention(full_embedding_dim, config.attention_config)
        self.gst = GST(n_mel_channels=n_mel_channels, config=gst_config)
        self.decoder = Decoder(
            n_mel_channels,
            config.n_frames_per_step,
            full_embedding_dim + config.attention_config.positional_dim,
            config.decoder_config,
        )
        self.postnet = Postnet(
            n_mel_channels,
            config.postnet_config,
        )

    def forward(
        self, batch: RegularBatch
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        phonem_emb = self.phonem_embedding(batch.phonemes).transpose(1, 2)
        speaker_emb: torch.Tensor = self.speaker_embedding(batch.speaker_ids).unsqueeze(
            1
        )

        phonem_emb = self.encoder(phonem_emb, batch.num_phonemes)
        if self.finetune:
            gst_emb = self.gst(batch.mels)
        else:
            gst_emb = torch.zeros(phonem_emb.shape[0], 1, self.gst_emb_dim).to(
                batch.mels.device
            )

        style_emb = torch.cat((gst_emb, speaker_emb), dim=-1)
        style_emb = torch.repeat_interleave(style_emb, phonem_emb.shape[1], dim=1)
        embeddings = torch.cat((phonem_emb, style_emb), dim=-1)

        durations, attented_embeddings = self.attention(
            embeddings, batch.num_phonemes, batch.durations
        )

        mel_outputs = self.decoder(attented_embeddings, batch.mels)
        mel_outputs_postnet = self.postnet(mel_outputs.transpose(1, 2))
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(1, 2)
        mask = get_mask_from_lengths(
            batch.durations.cumsum(dim=1)[:, -1].long()*self.n_frames_per_step, 
            max_len = mel_outputs_postnet.shape[1],
            device=batch.phonemes.device
        )
        mask = mask.unsqueeze(2)
        mel_outputs_postnet = mel_outputs_postnet * (1 - mask.float())
        mel_outputs = mel_outputs * (1 - mask.float())

        return (
            durations,
            mel_outputs_postnet,
            mel_outputs,
            gst_emb.squeeze(1),
        )

    def inference(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:

        text_inputs, text_lengths, speaker_ids, reference_mel = batch
        phonem_emb = self.phonem_embedding(text_inputs).transpose(1, 2)
        speaker_emb = self.speaker_embedding(speaker_ids).unsqueeze(1)

        phonem_emb = self.encoder(phonem_emb, text_lengths)
        if self.finetune:
            gst_emb = self.gst(reference_mel)
        else:
            gst_emb = torch.zeros(phonem_emb.shape[0], 1, self.gst_emb_dim).to(
                reference_mel.device
            )
        style_emb = torch.cat((gst_emb, speaker_emb), dim=-1)
        style_emb = torch.repeat_interleave(style_emb, phonem_emb.shape[1], dim=1)
        embeddings = torch.cat((phonem_emb, style_emb), dim=-1)

        attented_embeddings = self.attention.inference(embeddings, text_lengths)

        mel_outputs = self.decoder.inference(attented_embeddings)
        mel_outputs_postnet = self.postnet(mel_outputs.transpose(1, 2))
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(1, 2)

        return mel_outputs_postnet


class NonAttentiveTacotronVoicePrint(NonAttentiveTacotron):

    def forward(
        self, batch: VoicePrintBatch
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        phonem_emb = self.phonem_embedding(batch.phonemes).transpose(1, 2)
        speaker_emb: torch.Tensor = batch.speaker_embs.unsqueeze(1)

        phonem_emb = self.encoder(phonem_emb, batch.num_phonemes)
        if self.finetune:
            gst_emb = self.gst(batch.mels)
        else:
            gst_emb = torch.zeros(phonem_emb.shape[0], 1, self.gst_emb_dim).to(
                batch.mels.device
            )

        style_emb = torch.cat((gst_emb, speaker_emb), dim=-1)
        style_emb = torch.repeat_interleave(style_emb, phonem_emb.shape[1], dim=1)
        embeddings = torch.cat((phonem_emb, style_emb), dim=-1)

        durations, attented_embeddings = self.attention(
            embeddings, batch.num_phonemes, batch.durations
        )

        mel_outputs = self.decoder(attented_embeddings, batch.mels)
        mel_outputs_postnet = self.postnet(mel_outputs.transpose(1, 2))
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(1, 2)
        mask = get_mask_from_lengths(
            batch.durations.cumsum(dim=1)[:, -1].long()*self.n_frames_per_step,
            max_len = mel_outputs_postnet.shape[1],
            device=batch.phonemes.device
        )
        mask = mask.unsqueeze(2)
        mel_outputs_postnet = mel_outputs_postnet * (1 - mask.float())
        mel_outputs = mel_outputs * (1 - mask.float())

        return (
            durations,
            mel_outputs_postnet,
            mel_outputs,
            gst_emb.squeeze(1),
        )

    def inference(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:

        text_inputs, text_lengths, speaker_embs, reference_mel = batch
        phonem_emb = self.phonem_embedding(text_inputs).transpose(1, 2)
        speaker_emb = speaker_embs.unsqueeze(1)

        phonem_emb = self.encoder(phonem_emb, text_lengths)
        if self.finetune:
            gst_emb = self.gst(reference_mel)
        else:
            gst_emb = torch.zeros(phonem_emb.shape[0], 1, self.gst_emb_dim).to(
                reference_mel.device
            )
        style_emb = torch.cat((gst_emb, speaker_emb), dim=-1)
        style_emb = torch.repeat_interleave(style_emb, phonem_emb.shape[1], dim=1)
        embeddings = torch.cat((phonem_emb, style_emb), dim=-1)

        attented_embeddings = self.attention.inference(embeddings, text_lengths)

        mel_outputs = self.decoder.inference(attented_embeddings)
        mel_outputs_postnet = self.postnet(mel_outputs.transpose(1, 2))
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(1, 2)

        return mel_outputs_postnet



class VarianceAdaptor(nn.Module):
    """Variance Adaptor"""

    def __init__(self, config: VarianceAdaptorParams, pitch_min: float, pitch_max: float, 
            energy_min: float, energy_max: float, encoder_hidden: int, out_size: int):
        super(VarianceAdaptor, self).__init__()
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
            n_bins, out_size
        )
        self.energy_embedding = nn.Embedding(
            n_bins, out_size
        )


    def forward(self, embeddings, src_mask, pitch_target, energy_target):

        prediction_pitch = self.pitch_predictor(embeddings, src_mask)
        embedding_pitch = self.pitch_embedding(torch.bucketize(pitch_target, self.pitch_bins))


        prediction_energy = self.energy_predictor(embeddings, src_mask)
        embedding_energy = self.energy_embedding(torch.bucketize(energy_target, self.energy_bins))

        return (
            prediction_pitch,
            embedding_pitch,
            prediction_energy,
            embedding_energy,
        )

    def inference(self, embeddings, src_mask, p_control=1.0, e_control=1.0):
        
        prediction_pitch = self.pitch_predictor(embeddings, src_mask)
        prediction_pitch = prediction_pitch * p_control
        embedding_pitch = self.pitch_embedding(
            torch.bucketize(prediction_pitch, self.pitch_bins)
        )

        prediction_energy = self.energy_predictor(embeddings, src_mask)
        prediction_energy = prediction_energy * e_control
        embedding_energy = self.energy_embedding(
            torch.bucketize(prediction_energy, self.energy_bins)
        )

        return embedding_pitch, embedding_energy
        



class NonAttentiveTacotronVoicePrintVarianceAdaptor(nn.Module):
    def __init__(
        self,
        n_phonems: int,
        n_speakers: int,
        n_mel_channels: int,
        config: ModelParams,
        variance_adaptor_config: VarianceAdaptorParams,
        gst_config: GSTParams,
        finetune: bool,
        pitch_min: float, 
        pitch_max: float, 
        energy_min: float, 
        energy_max: float,
    ):
        super().__init__()

        self.n_frames_per_step = config.n_frames_per_step

        full_embedding_dim = (
            config.phonem_embedding_dim
            + config.speaker_embedding_dim
            + gst_config.emb_dim
        ) * 3  - gst_config.emb_dim * 2 # out variance adaptor - 
        self.finetune = finetune
        self.gst_emb_dim = gst_config.emb_dim
        self.phonem_embedding = nn.Embedding(
            n_phonems, config.phonem_embedding_dim, padding_idx=0
        )
        self.speaker_embedding = nn.Embedding(
            n_speakers,
            config.speaker_embedding_dim,
        )
        norm_emb_layer(
            self.phonem_embedding,
            n_phonems,
            config.phonem_embedding_dim,
        )
        norm_emb_layer(
            self.speaker_embedding,
            n_speakers,
            config.speaker_embedding_dim,
        )
        self.encoder = Encoder(
            config.phonem_embedding_dim,
            config.encoder_config,
        )
        self.attention = Attention(full_embedding_dim, config.attention_config)
        self.gst = GST(n_mel_channels=n_mel_channels, config=gst_config)

        self.variance_adaptor = VarianceAdaptor(variance_adaptor_config, pitch_min, pitch_max, 
            energy_min, energy_max, 
            config.phonem_embedding_dim + config.speaker_embedding_dim, 
            config.phonem_embedding_dim + config.speaker_embedding_dim)

        self.decoder = Decoder(
            n_mel_channels,
            config.n_frames_per_step,
            full_embedding_dim + config.attention_config.positional_dim,
            config.decoder_config,
        )
        self.postnet = Postnet(
            n_mel_channels,
            config.postnet_config,
        )

    def forward(
        self, batch: VoicePrintVarianceBatch
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        phonem_emb = self.phonem_embedding(batch.phonemes).transpose(1, 2)
        speaker_emb: torch.Tensor = batch.speaker_embs.unsqueeze(1)

        phonem_emb = self.encoder(phonem_emb, batch.num_phonemes)
        if self.finetune:
            gst_emb = self.gst(batch.mels)
        else:
            gst_emb = torch.zeros(phonem_emb.shape[0], 1, self.gst_emb_dim).to(
                batch.mels.device
            )

        style_emb = torch.cat((gst_emb, speaker_emb), dim=-1)
        style_emb = torch.repeat_interleave(style_emb, phonem_emb.shape[1], dim=1)
        style_emb_without_gst = torch.repeat_interleave(speaker_emb, phonem_emb.shape[1], dim=1)
        embeddings = torch.cat((phonem_emb, style_emb), dim=-1)
        embeddings_for_va = torch.cat((phonem_emb, style_emb_without_gst), dim=-1)

        src_masks = get_mask_from_lengths(batch.num_phonemes, batch.num_phonemes.device)
        prediction_pitch, embedding_pitch, prediction_energy, embedding_energy = self.variance_adaptor(
            embeddings_for_va, 
            src_masks.to(batch.phonemes.device), 
            batch.pitches,
            batch.energies
        )
        embeddings = torch.cat((embeddings, embedding_energy, embedding_pitch), dim=-1)
        durations, attented_embeddings = self.attention(
            embeddings, batch.num_phonemes, batch.durations
        )
        
        
        
        mel_outputs = self.decoder(attented_embeddings, batch.mels)
        mel_outputs_postnet = self.postnet(mel_outputs.transpose(1, 2))
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(1, 2)
        mask = get_mask_from_lengths(
            batch.durations.cumsum(dim=1)[:, -1].long()*self.n_frames_per_step,
            max_len = mel_outputs_postnet.shape[1],
            device=batch.phonemes.device
        )
        mask = mask.unsqueeze(2)
        mel_outputs_postnet = mel_outputs_postnet * (1 - mask.float())
        mel_outputs = mel_outputs * (1 - mask.float())

        return (
            durations,
            mel_outputs_postnet,
            mel_outputs,
            gst_emb.squeeze(1),
            prediction_pitch,
            prediction_energy,
        )

    def inference(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:

        text_inputs, text_lengths, speaker_embs, reference_mel = batch
        phonem_emb = self.phonem_embedding(text_inputs).transpose(1, 2)
        speaker_emb = speaker_embs.unsqueeze(1)

        phonem_emb = self.encoder(phonem_emb, text_lengths)
        if self.finetune:
            gst_emb = self.gst(reference_mel)
        else:
            gst_emb = torch.zeros(phonem_emb.shape[0], 1, self.gst_emb_dim).to(
                reference_mel.device
            )
        style_emb = torch.cat((gst_emb, speaker_emb), dim=-1)
        style_emb = torch.repeat_interleave(style_emb, phonem_emb.shape[1], dim=1)
        style_emb_without_gst = torch.repeat_interleave(speaker_emb, phonem_emb.shape[1], dim=1)
        embeddings = torch.cat((phonem_emb, style_emb), dim=-1)
        embeddings_for_va = torch.cat((phonem_emb, style_emb_without_gst), dim=-1)
        src_masks = get_mask_from_lengths(text_lengths, text_lengths.device)

        embedding_pitch, embedding_energy = self.variance_adaptor.inference(
            embeddings_for_va, 
            src_masks.to(text_inputs.device)
        )
        embeddings = torch.cat((embeddings, embedding_pitch, embedding_energy), dim=-1)
        attented_embeddings = self.attention.inference(embeddings, text_lengths)
        mel_outputs = self.decoder.inference(attented_embeddings)
        mel_outputs_postnet = self.postnet(mel_outputs.transpose(1, 2))
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(1, 2)

        return mel_outputs_postnet




class NonAttentiveTacotronVoicePrintVarianceAdaptorU(nn.Module):
    def __init__(
        self,
        n_phonems: int,
        n_speakers: int,
        n_mel_channels: int,
        config: ModelParams,
        variance_adaptor_config: VarianceAdaptorParams,
        gst_config: GSTParams,
        finetune: bool,
        pitch_min: float, 
        pitch_max: float, 
        energy_min: float, 
        energy_max: float,
    ):
        super().__init__()

        self.n_frames_per_step = config.n_frames_per_step
        self.duration_preparation_type = config.duration_preparation.method
        self.duration_prep_config = config.duration_preparation.conv_config

        full_embedding_dim = (
            config.phonem_embedding_dim
            + config.speaker_embedding_dim
            + gst_config.emb_dim
        )
        self.finetune = finetune
        self.gst_emb_dim = gst_config.emb_dim
        self.phonem_embedding = nn.Embedding(
            n_phonems, config.phonem_embedding_dim, padding_idx=0
        )
        self.speaker_embedding = nn.Embedding(
            n_speakers,
            config.speaker_embedding_dim,
        )
        norm_emb_layer(
            self.phonem_embedding,
            n_phonems,
            config.phonem_embedding_dim,
        )
        norm_emb_layer(
            self.speaker_embedding,
            n_speakers,
            config.speaker_embedding_dim,
        )
        self.encoder = Encoder(
            config.phonem_embedding_dim,
            config.encoder_config,
        )
        self.attention = Attention(full_embedding_dim, config.attention_config)
        self.gst = GST(n_mel_channels=n_mel_channels, config=gst_config)
        self.length_regulator = LengthRegulator()

        if self.duration_preparation_type == "identity":
            self.duration_preparation = IdompSecond()
        elif self.duration_preparation_type == "conv":
            self.duration_preparation = Conv1DNormDurationPrep(in_channels = config.phonem_embedding_dim + 1,
                                                out_channels = self.duration_prep_config.inner_channels)

        self.variance_adaptor = VarianceAdaptor(variance_adaptor_config, pitch_min, pitch_max, 
            energy_min, energy_max, 
            config.phonem_embedding_dim + config.speaker_embedding_dim, 
            config.phonem_embedding_dim + config.speaker_embedding_dim)

        self.decoder = Decoder(
            n_mel_channels,
            config.n_frames_per_step,
            full_embedding_dim + config.attention_config.positional_dim + (config.phonem_embedding_dim + config.speaker_embedding_dim) * 2,
            config.decoder_config,
        )
        self.postnet = Postnet(
            n_mel_channels,
            config.postnet_config,
        )

    def forward(
        self, batch: VoicePrintVarianceBatch
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        phonem_emb = self.phonem_embedding(batch.phonemes).transpose(1, 2)
        speaker_emb: torch.Tensor = batch.speaker_embs.unsqueeze(1)

        phonem_emb = self.encoder(phonem_emb, batch.num_phonemes)
        if self.finetune:
            gst_emb = self.gst(batch.mels)
        else:
            gst_emb = torch.zeros(phonem_emb.shape[0], 1, self.gst_emb_dim).to(
                batch.mels.device
            )

        style_emb = torch.cat((gst_emb, speaker_emb), dim=-1)
        style_emb = torch.repeat_interleave(style_emb, phonem_emb.shape[1], dim=1)
        style_emb_without_gst = torch.repeat_interleave(speaker_emb, phonem_emb.shape[1], dim=1)
        embeddings = torch.cat((phonem_emb, style_emb), dim=-1)
        embeddings_for_va = torch.cat((phonem_emb, style_emb_without_gst), dim=-1)

        src_masks = get_mask_from_lengths(batch.num_phonemes, batch.num_phonemes.device)
        prediction_pitch, embedding_pitch, prediction_energy, embedding_energy = self.variance_adaptor(
            embeddings_for_va, 
            src_masks.to(batch.phonemes.device), 
            batch.pitches,
            batch.energies
        )
               
        durations, attented_embeddings = self.attention(
            embeddings, batch.num_phonemes, batch.durations
        )

        durations_for_rounding = self.duration_preparation(phonem_emb, durations)

        max_decoder_seq_length = attented_embeddings.shape[1]

        ## NOTE: length_regulator has conversion to int inside it (rounds to the nearest int)...  
        upsampled_embedding_energy, _ = self.length_regulator(embedding_energy, durations_for_rounding, max_decoder_seq_length)
        upsampled_embedding_pitch, _ = self.length_regulator(embedding_pitch, durations_for_rounding, max_decoder_seq_length)

        embeddings_energy_pitch = torch.cat((upsampled_embedding_energy, upsampled_embedding_pitch), dim=-1)

        attented_embeddings = torch.cat((attented_embeddings, embeddings_energy_pitch), dim=-1)
        
        mel_outputs = self.decoder(attented_embeddings, batch.mels)
        
        mel_outputs_postnet = self.postnet(mel_outputs.transpose(1, 2))
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(1, 2)
        mask = get_mask_from_lengths(
            batch.durations.cumsum(dim=1)[:, -1].long()*self.n_frames_per_step,
            max_len = mel_outputs_postnet.shape[1],
            device=batch.phonemes.device
        )
        mask = mask.unsqueeze(2)
        mel_outputs_postnet = mel_outputs_postnet * (1 - mask.float())
        mel_outputs = mel_outputs * (1 - mask.float())

        return (
            durations,
            mel_outputs_postnet,
            mel_outputs,
            gst_emb.squeeze(1),
            prediction_pitch,
            prediction_energy,
        )

    def inference(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:

        text_inputs, text_lengths, speaker_embs, reference_mel = batch
        phonem_emb = self.phonem_embedding(text_inputs).transpose(1, 2)
        speaker_emb = speaker_embs.unsqueeze(1)

        phonem_emb = self.encoder(phonem_emb, text_lengths)
        if self.finetune:
            gst_emb = self.gst(reference_mel)
        else:
            gst_emb = torch.zeros(phonem_emb.shape[0], 1, self.gst_emb_dim).to(
                reference_mel.device
            )
        style_emb = torch.cat((gst_emb, speaker_emb), dim=-1)
        style_emb = torch.repeat_interleave(style_emb, phonem_emb.shape[1], dim=1)
        style_emb_without_gst = torch.repeat_interleave(speaker_emb, phonem_emb.shape[1], dim=1)
        embeddings = torch.cat((phonem_emb, style_emb), dim=-1)
        embeddings_for_va = torch.cat((phonem_emb, style_emb_without_gst), dim=-1)
        src_masks = get_mask_from_lengths(text_lengths, text_lengths.device)

        embedding_pitch, embedding_energy = self.variance_adaptor.inference(
            embeddings_for_va, 
            src_masks.to(text_inputs.device)
        )
        

        attented_embeddings, predicted_durations = self.attention.inference_with_durations(embeddings, 
                                                                                           text_lengths)

        durations_for_rounding = self.duration_preparation(phonem_emb, predicted_durations)
        max_decoder_seq_length = attented_embeddings.shape[1]

        ## NOTE: length_regulator has conversion to int inside it (rounds to the nearest int)...
        upsampled_embedding_energy, _ = self.length_regulator(embedding_energy, durations_for_rounding, max_decoder_seq_length)
        upsampled_embedding_pitch, _ = self.length_regulator(embedding_pitch, durations_for_rounding, max_decoder_seq_length)

        embeddings_energy_pitch = torch.cat((upsampled_embedding_energy, upsampled_embedding_pitch), dim=-1)

        attented_embeddings = torch.cat((attented_embeddings, embeddings_energy_pitch), dim=-1)

        mel_outputs = self.decoder.inference(attented_embeddings)
        mel_outputs_postnet = self.postnet(mel_outputs.transpose(1, 2))
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(1, 2)

        return mel_outputs_postnet
