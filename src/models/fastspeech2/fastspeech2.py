import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.fastspeech2.transformer.Models import Encoder, Decoder
from src.models.fastspeech2.transformer.Layers import PostNet
from .modules import VarianceAdaptor, LengthRegulator, VariancePredictor
from .utils import get_mask_from_lengths
from typing import List, Tuple

from src.data_process.fastspeech2_dataset import FastSpeech2Batch
from src.models.feature_models.gst import GST
from src.models.feature_models.config import GSTParams
from .config import FastSpeech2Params, VarianceAdaptorParams

from src.data_process.fastspeech2_dataset_voiceprint import FastSpeech2VoicePrintBatch


class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, config: FastSpeech2Params, n_mel_channels: int,
            n_phonems: int, n_speakers: int, pitch_min: float, pitch_max: float, 
            energy_min: float, energy_max: float, gst_config: GSTParams, finetune: bool, variance_adaptor: VarianceAdaptorParams):
        super(FastSpeech2, self).__init__()
        self.model_config = config
        self.gst_emb_dim = gst_config.emb_dim
        self.finetune = finetune
        self.use_gst = config.use_gst


        self.encoder = Encoder(config.encoder_params, config.max_seq_len, n_phonems)

        self.gst = GST(n_mel_channels=n_mel_channels, config=gst_config)
        
        self.variance_adaptor = VarianceAdaptor(variance_adaptor, pitch_min, pitch_max, 
        energy_min, energy_max, config.encoder_params.encoder_hidden)
        
        self.decoder = Decoder(config.decoder_params, config.max_seq_len)
        
        self.mel_linear = nn.Linear(
            config.decoder_params.decoder_hidden,
            n_mel_channels,
        )
        self.postnet = PostNet()


        self.speaker_emb = nn.Embedding(
            n_speakers,
            config.encoder_params.encoder_hidden,
        )

    def forward(self, batch: FastSpeech2Batch,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        max_phonemes_lenght = torch.max(batch.num_phonemes).item()
        max_mels_lenght = torch.max(batch.mels_lens).item()
        src_masks = get_mask_from_lengths(batch.num_phonemes, max_phonemes_lenght, batch.phonemes.device)
        mel_masks = get_mask_from_lengths(batch.mels_lens, max_mels_lenght, batch.phonemes.device)

        output = self.encoder(batch.phonemes, src_masks)

        if self.finetune:
            gst_emb = self.gst(batch.mels)
        else:
            gst_emb = torch.zeros(output.shape[0], 1, self.gst_emb_dim).to(
            batch.mels.device
        )
        

        speaker_emb = self.speaker_emb(batch.speaker_ids).unsqueeze(1).expand(
            -1, max_phonemes_lenght, -1
        )

        if self.use_gst:
            output += gst_emb
        output = output + speaker_emb
        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mels_lenght,
            batch.pitches,
            batch.energies,
            batch.durations,
            p_control,
            e_control,
            d_control,
        )

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            src_masks,
            mel_masks,
            gst_emb.squeeze(1)
        )

    def inference(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
    
        phonemes, num_phonemes, speaker_ids, reference_mel = batch
        max_phonemes_len = torch.max(num_phonemes).item()
        src_masks = get_mask_from_lengths(num_phonemes, max_phonemes_len, phonemes.device)
 
        output = self.encoder(phonemes, src_masks)


        if self.finetune:
            gst_emb = self.gst(reference_mel)
        else:
            gst_emb = torch.zeros(output.shape[0], 1, self.gst_emb_dim).to(
            reference_mel.device
        )
        

        speaker_emb = self.speaker_emb(speaker_ids).unsqueeze(1).expand(
            -1, max_phonemes_len, -1
        )

        if self.use_gst:
            output += gst_emb
        
        output = output + speaker_emb

        (output, mel_lens, mel_masks) = self.variance_adaptor.inference(output, src_masks, p_control, e_control, d_control)

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            src_masks,
            mel_masks,
            num_phonemes,
            mel_lens,
        )




class FastSpeech2VoicePrint(FastSpeech2):

    def __init__(self, config: FastSpeech2Params, n_mel_channels: int,
            n_phonems: int, n_speakers: int, pitch_min: float, pitch_max: float, 
            energy_min: float, energy_max: float, gst_config: GSTParams, finetune: bool, variance_adaptor: VarianceAdaptorParams):
        super(FastSpeech2, self).__init__()
        self.model_config = config
        self.gst_emb_dim = gst_config.emb_dim
        self.finetune = finetune
        self.use_gst = config.use_gst


        self.encoder = Encoder(config.encoder_params, config.max_seq_len, n_phonems)

        self.gst = GST(n_mel_channels=n_mel_channels, config=gst_config)
        
        self.variance_adaptor = VarianceAdaptor(variance_adaptor, pitch_min, pitch_max, 
        energy_min, energy_max, config.encoder_params.encoder_hidden)
        
        self.decoder = Decoder(config.decoder_params, config.max_seq_len)
        
        self.mel_linear = nn.Linear(
            config.decoder_params.decoder_hidden,
            n_mel_channels,
        )
        self.postnet = PostNet()


        self.voiceprint_emb = nn.Linear(
            config.encoder_params.encoder_hidden,
            config.encoder_params.encoder_hidden,
        )

    def forward(self, batch: FastSpeech2VoicePrintBatch,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        max_phonemes_lenght = torch.max(batch.num_phonemes).item()
        max_mels_lenght = torch.max(batch.mels_lens).item()
        src_masks = get_mask_from_lengths(batch.num_phonemes, max_phonemes_lenght, batch.phonemes.device)
        mel_masks = get_mask_from_lengths(batch.mels_lens, max_mels_lenght, batch.phonemes.device)

        output = self.encoder(batch.phonemes, src_masks)

        if self.finetune:
            gst_emb = self.gst(batch.mels)
        else:
            gst_emb = torch.zeros(output.shape[0], 1, self.gst_emb_dim).to(
            batch.mels.device
        )
        


        if self.use_gst:
            output += gst_emb

        speaker_emb = self.voiceprint_emb(batch.speaker_embs)
        output = output + speaker_emb.unsqueeze(1).expand(
            -1, max_phonemes_lenght, -1
        )
        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mels_lenght,
            batch.pitches,
            batch.energies,
            batch.durations,
            p_control,
            e_control,
            d_control,
        )

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            src_masks,
            mel_masks,
            gst_emb.squeeze(1)
        )

    def inference(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
    
        phonemes, num_phonemes, speaker_emb, reference_mel = batch
        max_phonemes_len = torch.max(num_phonemes).item()
        src_masks = get_mask_from_lengths(num_phonemes, max_phonemes_len, phonemes.device)
 
        output = self.encoder(phonemes, src_masks)


        if self.finetune:
            gst_emb = self.gst(reference_mel)
        else:
            gst_emb = torch.zeros(output.shape[0], 1, self.gst_emb_dim).to(
            reference_mel.device
        )
        


        if self.use_gst:
            output += gst_emb
        
        speaker_emb = self.voiceprint_emb(speaker_emb)
        output = output + speaker_emb.unsqueeze(1).expand(
            -1, max_phonemes_len, -1
        )

        (output, mel_lens, mel_masks) = self.variance_adaptor.inference(output, src_masks, p_control, e_control, d_control)

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            src_masks,
            mel_masks,
            num_phonemes,
            mel_lens,
        )





class FastSpeech2Dutaion(nn.Module):
    """ FastSpeech2 """

    def __init__(self, config: FastSpeech2Params, n_mel_channels: int,
            n_phonems: int, n_speakers: int, gst_config: GSTParams, finetune: bool, variance_adaptor: VarianceAdaptorParams):
        super(FastSpeech2Dutaion, self).__init__()
        self.model_config = config
        self.gst_emb_dim = gst_config.emb_dim
        self.finetune = finetune
        self.use_gst = config.use_gst
        
        
        self.duration_predictor = VariancePredictor(variance_adaptor.predictor_params, config.encoder_params.encoder_hidden)
        self.length_regulator = LengthRegulator()


        self.encoder = Encoder(config.encoder_params, config.max_seq_len, n_phonems)

        self.gst = GST(n_mel_channels=n_mel_channels, config=gst_config)

        self.voiceprint_emb = nn.Linear(
            config.encoder_params.encoder_hidden,
            config.encoder_params.encoder_hidden,
        )
        
        self.decoder = Decoder(config.decoder_params, config.max_seq_len)
        
        self.mel_linear = nn.Linear(
            config.decoder_params.decoder_hidden,
            n_mel_channels,
        )
        self.postnet = PostNet()



    def forward(self, batch: FastSpeech2VoicePrintBatch):
        max_phonemes_lenght = torch.max(batch.num_phonemes).item()
        max_mels_lenght = torch.max(batch.mels_lens).item()
        src_masks = get_mask_from_lengths(batch.num_phonemes, max_phonemes_lenght, batch.phonemes.device)
        mel_masks = get_mask_from_lengths(batch.mels_lens, max_mels_lenght, batch.phonemes.device)

        output = self.encoder(batch.phonemes, src_masks)

        if self.finetune:
            gst_emb = self.gst(batch.mels)
        else:
            gst_emb = torch.zeros(output.shape[0], 1, self.gst_emb_dim).to(
            batch.mels.device
        )
        
        
        speaker_emb = self.voiceprint_emb(batch.speaker_embs)
        output = output + speaker_emb.unsqueeze(1).expand(
            -1, max_phonemes_lenght, -1
        )

        if self.use_gst:
            output += gst_emb

        log_duration_prediction = self.duration_predictor(output, src_masks)
        output, _ = self.length_regulator(output, batch.durations, max_mels_lenght)

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            log_duration_prediction,
            src_masks,
            mel_masks,
            gst_emb.squeeze(1)
        )

    def inference(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], d_control=1.0):
    
        phonemes, num_phonemes, speaker_emb, reference_mel = batch
        max_phonemes_len = torch.max(num_phonemes).item()
        src_masks = get_mask_from_lengths(num_phonemes, max_phonemes_len, phonemes.device)
 
        output = self.encoder(phonemes, src_masks)


        if self.finetune:
            gst_emb = self.gst(reference_mel)
        else:
            gst_emb = torch.zeros(output.shape[0], 1, self.gst_emb_dim).to(
            reference_mel.device
        )
        
        speaker_emb = self.voiceprint_emb(speaker_emb)
        #speaker_emb = self.voiceprint_emb(batch.speaker_embs)
        output = output + speaker_emb.unsqueeze(1).expand(
            -1, max_phonemes_len, -1
        )


        if self.use_gst:
            output += gst_emb


        log_duration_prediction = self.duration_predictor(output, src_masks)
        duration_rounded = torch.clamp(
            (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
            min=0,
        )

        output, mel_len = self.length_regulator(output, duration_rounded, None)
        mel_masks = get_mask_from_lengths(mel_len, torch.max(mel_len).item(), mel_len.device)

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            src_masks,
            mel_masks,
            num_phonemes,
            mel_len,
        )