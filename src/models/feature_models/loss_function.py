from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from src.train_config import LossParams


class NonAttentiveTacotronLoss(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        hop_size: int,
        config: LossParams
    ):
        super(NonAttentiveTacotronLoss, self).__init__()
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.mel_weight = config.mels_weight
        self.duration_weight = config.duration_weight

    def forward(
        self, prenet_mels: torch.Tensor, postnet_mels: torch.Tensor, model_durations: torch.Tensor,
            target_durations: torch.Tensor, target_mels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_mels.requires_grad = False
        target_durations.requires_grad = False
        target_mels_padded = torch.zeros(prenet_mels.shape, device = prenet_mels.device)
        target_mels_padded[:, :target_mels.shape[1], :] = target_mels
        prenet_l1 = F.l1_loss(prenet_mels, target_mels_padded)
        prenet_l2 = F.mse_loss(prenet_mels, target_mels_padded)
        postnet_l1 = F.l1_loss(postnet_mels, target_mels_padded)
        postnet_l2 = F.mse_loss(postnet_mels, target_mels_padded)
        loss_prenet = self.mel_weight * (prenet_l1 + prenet_l2)
        loss_postnet = self.mel_weight * (postnet_l1 + postnet_l2)
        model_durations = model_durations * self.hop_size / self.sample_rate
        target_durations = target_durations * self.hop_size / self.sample_rate
        loss_durations = self.duration_weight * F.mse_loss(model_durations, target_durations)
        return loss_prenet, loss_postnet, loss_durations


class NonAttentiveTacotronVarianceAdaptorLoss(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        hop_size: int,
        config: LossParams
    ):
        super(NonAttentiveTacotronVarianceAdaptorLoss, self).__init__()
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.mel_weight = config.mels_weight
        self.duration_weight = config.duration_weight
        self.config = config

    def forward(
        self, prenet_mels: torch.Tensor, postnet_mels: torch.Tensor, model_durations: torch.Tensor,
            target_durations: torch.Tensor, target_mels: torch.Tensor,
            model_pitches: torch.Tensor, model_enegries: torch.Tensor,
            target_pitches: torch.Tensor, target_energies: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_mels.requires_grad = False
        target_durations.requires_grad = False
        target_mels_padded = torch.zeros(prenet_mels.shape, device = prenet_mels.device)
        target_mels_padded[:, :target_mels.shape[1], :] = target_mels

        prenet_l1 = F.l1_loss(prenet_mels, target_mels_padded)
        prenet_l2 = F.mse_loss(prenet_mels, target_mels_padded)
        postnet_l1 = F.l1_loss(postnet_mels, target_mels_padded)
        postnet_l2 = F.mse_loss(postnet_mels, target_mels_padded)
        loss_prenet = self.mel_weight * (prenet_l1 + prenet_l2)
        loss_postnet = self.mel_weight * (postnet_l1 + postnet_l2)
        model_durations = model_durations * self.hop_size / self.sample_rate
        target_durations = target_durations * self.hop_size / self.sample_rate
        loss_pitch = self.config.pitch_coef * F.mse_loss(model_pitches, target_pitches)
        loss_energy = self.config.energy_coef * F.mse_loss(model_enegries, target_energies)
        loss_durations = self.duration_weight * F.mse_loss(model_durations, target_durations)
        return loss_prenet, loss_postnet, loss_durations, loss_pitch, loss_energy
