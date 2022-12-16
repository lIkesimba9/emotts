import torch
import torch.nn as nn
from src.data_process.fastspeech2_dataset import FastSpeech2Batch
from src.data_process.fastspeech2_dataset_voiceprint import FastSpeech2VoicePrintBatch

class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self):
        super(FastSpeech2Loss, self).__init__()

        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, batch: FastSpeech2Batch, mel_predictions,
                    postnet_mel_predictions,
                    pitch_predictions,
                    energy_predictions,
                    log_duration_predictions,
                    src_masks,
                    mel_masks
                ):

        src_masks = ~src_masks
        mel_masks = ~mel_masks
        log_duration_targets = torch.log(batch.durations.float() + 1)
        mel_targets = batch.mels[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        log_duration_targets.requires_grad = False
        batch.pitches.requires_grad = False
        batch.energies.requires_grad = False
        mel_targets.requires_grad = False


        pitch_predictions = pitch_predictions.masked_select(src_masks)
        pitch_targets = batch.pitches.masked_select(src_masks)


        energy_predictions = energy_predictions.masked_select(src_masks)
        energy_targets = batch.energies.masked_select(src_masks)


        log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)

        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        postnet_mel_predictions = postnet_mel_predictions.masked_select(
            mel_masks.unsqueeze(-1)
        )
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        mel_loss = self.mae_loss(mel_predictions, mel_targets)
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)

        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        energy_loss = self.mse_loss(energy_predictions, energy_targets)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

        total_loss = (
            mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss
        )

        return (
            total_loss,
            mel_loss,
            postnet_mel_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
        )

class FastSpeech2VoicePrintLoss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self):
        super(FastSpeech2VoicePrintLoss, self).__init__()

        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, batch: FastSpeech2VoicePrintBatch, mel_predictions,
                    postnet_mel_predictions,
                    pitch_predictions,
                    energy_predictions,
                    log_duration_predictions,
                    src_masks,
                    mel_masks
                ):

        src_masks = ~src_masks
        mel_masks = ~mel_masks
        log_duration_targets = torch.log(batch.durations.float() + 1)
        mel_targets = batch.mels[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        log_duration_targets.requires_grad = False
        batch.pitches.requires_grad = False
        batch.energies.requires_grad = False
        mel_targets.requires_grad = False


        pitch_predictions = pitch_predictions.masked_select(src_masks)
        pitch_targets = batch.pitches.masked_select(src_masks)


        energy_predictions = energy_predictions.masked_select(src_masks)
        energy_targets = batch.energies.masked_select(src_masks)


        log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)

        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        postnet_mel_predictions = postnet_mel_predictions.masked_select(
            mel_masks.unsqueeze(-1)
        )
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        mel_loss = self.mae_loss(mel_predictions, mel_targets)
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)

        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        energy_loss = self.mse_loss(energy_predictions, energy_targets)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

        total_loss = (
            mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss
        )

        return (
            total_loss,
            mel_loss,
            postnet_mel_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
        )


class FastSpeech2DurationLoss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self):
        super(FastSpeech2DurationLoss, self).__init__()

        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, batch: FastSpeech2VoicePrintBatch, mel_predictions,
                    postnet_mel_predictions,
                    log_duration_predictions,
                    src_masks,
                    mel_masks
                ):

        src_masks = ~src_masks
        mel_masks = ~mel_masks
        log_duration_targets = torch.log(batch.durations.float() + 1)
        mel_targets = batch.mels[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        log_duration_targets.requires_grad = False
        batch.pitches.requires_grad = False
        batch.energies.requires_grad = False
        mel_targets.requires_grad = False


        log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)

        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        postnet_mel_predictions = postnet_mel_predictions.masked_select(
            mel_masks.unsqueeze(-1)
        )
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        mel_loss = self.mae_loss(mel_predictions, mel_targets)
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)

        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

        total_loss = (
            mel_loss + postnet_mel_loss + duration_loss
        )

        return (
            total_loss,
            mel_loss,
            postnet_mel_loss,
            duration_loss,
        )




class FastSpeech2LossNoVPDurationOnly(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self):
        super(FastSpeech2LossNoVPDurationOnly, self).__init__()

        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, batch: FastSpeech2Batch, mel_predictions,
                    postnet_mel_predictions,
                    log_duration_predictions,
                    src_masks,
                    mel_masks
                ):

        src_masks = ~src_masks
        mel_masks = ~mel_masks
        log_duration_targets = torch.log(batch.durations.float() + 1)
        mel_targets = batch.mels[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        log_duration_targets.requires_grad = False
        batch.pitches.requires_grad = False
        batch.energies.requires_grad = False
        mel_targets.requires_grad = False

        log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)

        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        postnet_mel_predictions = postnet_mel_predictions.masked_select(
            mel_masks.unsqueeze(-1)
        )
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        mel_loss = self.mae_loss(mel_predictions, mel_targets)
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)

        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

        total_loss = (
            mel_loss + postnet_mel_loss + duration_loss 
        )

        return (
            total_loss,
            mel_loss,
            postnet_mel_loss,
            duration_loss,
        )