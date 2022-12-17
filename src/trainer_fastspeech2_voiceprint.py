import json
import os
from pathlib import Path
from typing import Dict, Optional, OrderedDict, Tuple

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from src.constants import (
    CHECKPOINT_DIR,
    FASTSPEECH2_CHECKPOINT_NAME,
    FASTSPEECH2_MODEL_FILENAME,
    PHONEMES_ENG,
    PHONEMES_CHI,
    LOG_DIR,
    MELS_MEAN_FILENAME,
    MELS_STD_FILENAME,
    ENERGY_MEAN_FILENAME,
    ENERGY_STD_FILENAME,
    ENERGY_MIN_FILENAME,
    ENERGY_MAX_FILENAME,
    PITCH_MEAN_FILENAME,
    PITCH_STD_FILENAME,
    PITCH_MIN_FILENAME,
    PITCH_MAX_FILENAME,
    PHONEMES_FILENAME,
    REFERENCE_PATH,
    SPEAKERS_FILENAME,
    SPEAKER_PRINT_DIR,
)
from src.data_process.fastspeech2_dataset_voiceprint import FastSpeech2VoicePrintBatch, FastSpeech2VoicePrintCollate, FastSpeech2VoicePrintFactory
from src.models.fastspeech2.fastspeech2 import FastSpeech2VoicePrint
from src.models.fastspeech2.loss import FastSpeech2VoicePrintLoss
from src.models.hifi_gan.models import Generator, load_model as load_hifi
from src.train_config import TrainParams



class Trainer:

    MODEL_OPTIMIZER_FILENAME = "model_optimizer.pth"
    DISC_OPTIMIZER_FILENAME = "disc_optimizer.pth"
    DISC_MODEL_FILENAME = "discriminator.pth"
    ITERATION_FILENAME = "iter.json"
    ITERATION_NAME = "iteration"
    EPOCH_NAME = "epoch"
    SAMPLE_SIZE = 10

    def __init__(self, config: TrainParams):
        self.config = config
        base_model_path = Path(self.config.base_model)

        self.use_gst = config.fastspeech2.use_gst

        self.checkpoint_path = (
            CHECKPOINT_DIR / self.config.checkpoint_name / FASTSPEECH2_CHECKPOINT_NAME
        )
        mapping_folder = (
            base_model_path if self.config.finetune else self.checkpoint_path
        )
        self.log_dir = LOG_DIR / self.config.checkpoint_name / FASTSPEECH2_CHECKPOINT_NAME
        self.references = list( (REFERENCE_PATH / Path(self.config.lang)).rglob("*.pkl"))

            
        self.create_dirs()
        self.phonemes_to_id: Dict[str, int] = {}
        self.speakers_to_id: Dict[str, int] = {}
        self.device = torch.device(self.config.device)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.iteration_step = 1
        self.upload_mapping(mapping_folder)
        self.train_loader, self.valid_loader = self.prepare_loaders()

        self.mels_mean = self.train_loader.dataset.mels_mean
        self.mels_std = self.train_loader.dataset.mels_std
        
        self.energy_mean = self.train_loader.dataset.energy_mean
        self.energy_std = self.train_loader.dataset.energy_std
        self.energy_min = self.train_loader.dataset.energy_min
        self.energy_max = self.train_loader.dataset.energy_max

        self.pitch_mean = self.train_loader.dataset.pitch_mean
        self.pitch_std = self.train_loader.dataset.pitch_std
        self.pitch_min = self.train_loader.dataset.pitch_min
        self.pitch_max = self.train_loader.dataset.pitch_max
        

        self.fastspeech2_model = FastSpeech2VoicePrint(
            config=self.config.fastspeech2,
            n_mel_channels=self.config.n_mels,
            n_phonems=len(self.phonemes_to_id),
            n_speakers=len(self.speakers_to_id),
            pitch_min=self.pitch_min,
            pitch_max=self.pitch_max,
            energy_min=self.energy_min,
            energy_max=self.energy_max,
            gst_config=self.config.gst_config,
            finetune=self.config.finetune,
            variance_adaptor=self.config.variance_adapter_params,
        ).to(self.device)

        if self.config.finetune:
            self.fastspeech2_model = torch.load(
                base_model_path / FASTSPEECH2_MODEL_FILENAME, map_location=self.device
            )

            self.fastspeech2_model.finetune = self.config.finetune
            self.fastspeech2_model.encoder.requires_grad_(False)

            self.mels_mean = torch.load(mapping_folder / MELS_MEAN_FILENAME)
            self.mels_std = torch.load(mapping_folder / MELS_STD_FILENAME)
            
            self.energy_mean = torch.load(mapping_folder / ENERGY_MEAN_FILENAME)
            self.energy_std = torch.load(mapping_folder / ENERGY_STD_FILENAME)
            self.energy_min = torch.load(mapping_folder / ENERGY_MIN_FILENAME)
            self.energy_max = torch.load(mapping_folder / ENERGY_MAX_FILENAME)
            
            self.pitch_mean = torch.load(mapping_folder / PITCH_MEAN_FILENAME)
            self.pitch_std = torch.load(mapping_folder / PITCH_STD_FILENAME)
            self.pitch_min = torch.load(mapping_folder / PITCH_MIN_FILENAME)
            self.pitch_max = torch.load(mapping_folder / PITCH_MAX_FILENAME)
        
        if self.use_gst:

            self.discriminator = nn.Sequential(
                nn.Linear(self.config.gst_config.emb_dim, len(self.speakers_to_id)),
                nn.Softmax(),
            )
            self.discriminator = self.discriminator.to(self.device)

            self.discriminator_optimizer = Adam(
                self.discriminator.parameters(),
                lr=self.config.optimizer.learning_rate,
                weight_decay=self.config.optimizer.reg_weight,
                betas=(self.config.optimizer.adam_beta1, self.config.optimizer.adam_beta2),
                eps=self.config.optimizer.adam_epsilon,
            )
            
            self.discriminator_scheduler = StepLR(
                optimizer=self.discriminator_optimizer,
                step_size=self.config.scheduler.decay_steps,
                gamma=self.config.scheduler.decay_rate,
            )

            self.adversatial_criterion = nn.NLLLoss()

        self.vocoder: Generator = load_hifi(
            model_path=self.config.pretrained_hifi,
            hifi_config=self.config.train_hifi.model_param,
            num_mels=self.config.n_mels,
            device=self.device,
        )

        
        self.model_optimizer = Adam(
            self.fastspeech2_model.parameters(),
            lr=self.config.optimizer.learning_rate,
            weight_decay=self.config.optimizer.reg_weight,
            betas=(self.config.optimizer.adam_beta1, self.config.optimizer.adam_beta2),
            eps=self.config.optimizer.adam_epsilon,
        )

        self.model_scheduler = StepLR(
            optimizer=self.model_optimizer,
            step_size=self.config.scheduler.decay_steps,
            gamma=self.config.scheduler.decay_rate,
        )


        self.criterion = FastSpeech2VoicePrintLoss()


        self.upload_checkpoints()

    def batch_to_device(self, batch: FastSpeech2VoicePrintBatch) -> FastSpeech2VoicePrintBatch:
        batch_on_device = FastSpeech2VoicePrintBatch(
            speaker_ids=batch.speaker_ids.to(self.device).detach(),
            phonemes=batch.phonemes.to(self.device).detach(),
            num_phonemes=batch.num_phonemes.to(self.device).detach(),
            mels=batch.mels.to(self.device).detach(),
            mels_lens=batch.mels_lens.to(self.device).detach(),
            energies=batch.energies.to(self.device).detach(),
            pitches=batch.pitches.to(self.device).detach(),
            durations=batch.durations.to(self.device).detach(),
            speaker_embs=batch.speaker_embs.to(self.device).detach(),
        )
        return batch_on_device

    def create_dirs(self) -> None:
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def mapping_is_exist(mapping_folder: Path) -> bool:
        if not os.path.isfile(mapping_folder / SPEAKERS_FILENAME):
            return False
        if not os.path.isfile(mapping_folder / PHONEMES_FILENAME):
            return False
        return True

    def get_last_model(self) -> Optional[Path]:
        models = list(self.checkpoint_path.rglob(f"*_{FASTSPEECH2_MODEL_FILENAME}"))
        if len(models) == 0:
            return None
        return max(models, key=lambda x: int(x.name.split("_")[0]))

    def checkpoint_is_exist(self) -> bool:  # noqa: CFQ004
        model_path = self.get_last_model()
        if model_path is None:
            return False
        if self.use_gst and not (self.checkpoint_path / self.DISC_MODEL_FILENAME).is_file():
            return False
        if not (self.checkpoint_path / self.MODEL_OPTIMIZER_FILENAME).is_file():
            return False
        if self.use_gst and not (self.checkpoint_path / self.DISC_OPTIMIZER_FILENAME).is_file():
            return False
            
        if not (self.checkpoint_path / self.ITERATION_FILENAME).is_file():
            return False
        else:
            with open(self.checkpoint_path / self.ITERATION_FILENAME) as f:
                iter_dict = json.load(f)
                if self.ITERATION_NAME not in iter_dict:
                    return False
        return True

    def upload_mapping(self, mapping_folder: Path) -> None:
        if self.mapping_is_exist(mapping_folder):
            with open(mapping_folder / SPEAKERS_FILENAME) as f:
                self.speakers_to_id.update(json.load(f))
            with open(mapping_folder / PHONEMES_FILENAME) as f:
                self.phonemes_to_id.update(json.load(f))

    def upload_checkpoints(self) -> None:
        if self.checkpoint_is_exist():
            model_path = self.get_last_model()
            self.fastspeech2_model: FastSpeech2VoicePrint = torch.load(
                model_path, map_location=self.device
            )

            model_optimizer_state_dict: OrderedDict[str, torch.Tensor] = torch.load(
                self.checkpoint_path / self.MODEL_OPTIMIZER_FILENAME, map_location=self.device
            )

            if self.use_gst:
                self.discriminator: nn.Module = torch.load(
                    self.checkpoint_path / self.DISC_MODEL_FILENAME,
                    map_location=self.device
                )
                disc_optimizer_state_dict: OrderedDict[str, torch.Tensor] = torch.load(
                    self.checkpoint_path / self.DISC_OPTIMIZER_FILENAME, map_location=self.device
                )


            with open(self.checkpoint_path / self.ITERATION_FILENAME) as f:
                iteration_dict: Dict[str, int] = json.load(f)
            self.model_optimizer.load_state_dict(model_optimizer_state_dict)
            self.model_scheduler = StepLR(
                optimizer=self.model_optimizer,
                step_size=self.config.scheduler.decay_steps,
                gamma=self.config.scheduler.decay_rate,
            )
            if self.use_gst:
                self.discriminator_optimizer.load_state_dict(disc_optimizer_state_dict)
                self.discriminator_scheduler = StepLR(
                    optimizer=self.discriminator_optimizer,
                    step_size=self.config.scheduler.decay_steps,
                    gamma=self.config.scheduler.decay_rate,
                )
            self.iteration_step = iteration_dict[self.ITERATION_NAME]

    def save_checkpoint(self) -> None:
        with open(self.checkpoint_path / SPEAKERS_FILENAME, "w") as f:
            json.dump(self.speakers_to_id, f)
        with open(self.checkpoint_path / PHONEMES_FILENAME, "w") as f:
            json.dump(self.phonemes_to_id, f)
        with open(self.checkpoint_path / self.ITERATION_FILENAME, "w") as f:
            json.dump({self.ITERATION_NAME: self.iteration_step}, f)
        torch.save(
            self.fastspeech2_model,
            self.checkpoint_path / f"{self.iteration_step}_{FASTSPEECH2_MODEL_FILENAME}",
        )
        torch.save(
            self.model_optimizer.state_dict(),
            self.checkpoint_path / self.MODEL_OPTIMIZER_FILENAME,
        )
        if self.use_gst:
            torch.save(
                self.discriminator,
                self.checkpoint_path / self.DISC_MODEL_FILENAME
            )
            torch.save(
                self.discriminator_optimizer.state_dict(),
                self.checkpoint_path / self.DISC_OPTIMIZER_FILENAME,
            )

        torch.save(self.mels_mean, self.checkpoint_path / MELS_MEAN_FILENAME)
        torch.save(self.mels_std, self.checkpoint_path / MELS_STD_FILENAME)

        torch.save(self.energy_mean, self.checkpoint_path / ENERGY_MEAN_FILENAME)
        torch.save(self.energy_std, self.checkpoint_path / ENERGY_STD_FILENAME)
        torch.save(self.energy_min, self.checkpoint_path / ENERGY_MIN_FILENAME)
        torch.save(self.energy_max, self.checkpoint_path / ENERGY_MAX_FILENAME)

        torch.save(self.pitch_mean, self.checkpoint_path / PITCH_MEAN_FILENAME)
        torch.save(self.pitch_std, self.checkpoint_path / PITCH_STD_FILENAME)
        torch.save(self.pitch_min, self.checkpoint_path / PITCH_MIN_FILENAME)
        torch.save(self.pitch_max, self.checkpoint_path / PITCH_MAX_FILENAME)


    def prepare_loaders(self) -> Tuple[DataLoader[FastSpeech2VoicePrintBatch], DataLoader[FastSpeech2VoicePrintBatch]]:

        factory = FastSpeech2VoicePrintFactory(
            sample_rate=self.config.sample_rate,
            hop_size=self.config.hop_size,
            n_mels=self.config.n_mels,
            config=self.config.data,
            phonemes_to_id=self.phonemes_to_id,
            speakers_to_id=self.speakers_to_id,
            ignore_speakers=self.config.data.ignore_speakers,
            finetune=self.config.finetune,
        )
        self.phonemes_to_id = factory.phoneme_to_id
        self.speakers_to_id = factory.speaker_to_id
        trainset, valset = factory.split_train_valid(self.config.test_size)
        collate_fn = FastSpeech2VoicePrintCollate()

        train_loader = DataLoader(
            trainset,
            shuffle=False,
            batch_size=self.config.batch_size,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            valset,
            shuffle=False,
            batch_size=self.config.batch_size,
            collate_fn=collate_fn,
        )

        return train_loader, val_loader  # type: ignore

    def write_losses(
        self,
        tag: str,
        losses_dict: Dict[str, float],
    ) -> None:
        for name, value in losses_dict.items():
            self.writer.add_scalar(
                f"Loss/{tag}/{name}", value, global_step=self.iteration_step
            )

    def vocoder_inference(self, tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = tensor.unsqueeze(0).to(self.device)
            y_g_hat = self.vocoder(x)
            audio = y_g_hat.squeeze()
        return audio

    def calc_adv_loss(
        self, style_emb: torch.Tensor, batch: FastSpeech2VoicePrintBatch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        log_model = torch.log(1 - self.discriminator(style_emb))
        loss_model: torch.Tensor = (
            self.config.loss.adversarial_weight
            * self.adversatial_criterion(log_model, batch.speaker_ids)
        )

        log_dicriminator = torch.log(self.discriminator(style_emb.detach()))

        loss_discriminator = (
            self.config.loss.adversarial_weight
            * self.adversatial_criterion(log_dicriminator, batch.speaker_ids)
        )

        return loss_model, loss_discriminator


    def train(self) -> None:
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed(self.config.seed)

        self.fastspeech2_model.train()

        while self.iteration_step < self.config.total_iterations:
            for batch in self.train_loader:
                batch = self.batch_to_device(batch)
                self.model_optimizer.zero_grad()
                (
                    mel_predictions,
                    postnet_mel_predictions,
                    pitch_predictions,
                    energy_predictions,
                    log_duration_predictions,
                    src_masks,
                    mel_masks,
                    gst_emb
                ) = self.fastspeech2_model(batch)
                
                total_loss, mel_loss, postnet_mel_loss, pitch_loss, energy_loss, duration_loss  = self.criterion(batch, 
                    mel_predictions,
                    postnet_mel_predictions,
                    pitch_predictions,
                    energy_predictions,
                    log_duration_predictions,
                    src_masks,
                    mel_masks
                )
                if self.use_gst:
                    loss_generator, loss_discriminator = self.calc_adv_loss(
                        gst_emb, batch
                    )
                    total_loss = total_loss + loss_generator

                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.fastspeech2_model.parameters(), self.config.grad_clip_thresh
                )

                self.model_optimizer.step()
                if self.use_gst:
                    self.discriminator_optimizer.zero_grad()

                    loss_discriminator.backward()

                    self.discriminator_optimizer.step()

                if (
                    self.config.scheduler.start_decay
                    <= self.iteration_step
                    <= self.config.scheduler.last_epoch
                ):
                    self.model_scheduler.step()

                if self.iteration_step % self.config.log_steps == 0:
                    if self.use_gst:
                        self.write_losses(
                            "train",
                            {
                                "total": total_loss,
                                "mel_loss": mel_loss,
                                "postnet_mel_loss": postnet_mel_loss,
                                "pitch_loss": pitch_loss,
                                "energy_loss": energy_loss,
                                "duration_loss": duration_loss,
                                "generator": loss_generator,
                                "discriminator": loss_discriminator
                            }
                        )
                    else:
                        self.write_losses(
                            "train",
                            {
                                "total": total_loss,
                                "mel_loss": mel_loss,
                                "postnet_mel_loss": postnet_mel_loss,
                                "pitch_loss": pitch_loss,
                                "energy_loss": energy_loss,
                                "duration_loss": duration_loss
                            }
                        )                        

                if self.iteration_step % self.config.iters_per_checkpoint == 0:
                    self.fastspeech2_model.eval()
                    self.validate()
                    self.generate_samples()
                    self.save_checkpoint()
                    self.fastspeech2_model.train()

                self.iteration_step += 1
                if self.iteration_step >= self.config.total_iterations:
                    break

        self.writer.close()

    def validate(self) -> None:
        with torch.no_grad():
            val_loss = 0.0
            val_mel_loss = 0.0
            val_postnet_mel_loss = 0.0
            val_pitch_loss = 0.0
            val_energy_loss = 0.0
            val_duration_loss = 0.0
            for batch in self.valid_loader:
                batch = self.batch_to_device(batch)
                (
                    mel_predictions,
                    postnet_mel_predictions,
                    pitch_predictions,
                    energy_predictions,
                    log_duration_predictions,
                    src_masks,
                    mel_masks,
                    _
                ) = self.fastspeech2_model(batch)
                
                total_loss, mel_loss, postnet_mel_loss, pitch_loss, energy_loss, duration_loss  = self.criterion(batch, 
                    mel_predictions,
                    postnet_mel_predictions,
                    pitch_predictions,
                    energy_predictions,
                    log_duration_predictions,
                    src_masks,
                    mel_masks
                )
                
                val_loss += total_loss.item()
                val_mel_loss += mel_loss.item()
                val_postnet_mel_loss += postnet_mel_loss.item()
                val_pitch_loss += pitch_loss.item()
                val_energy_loss += energy_loss.item()
                val_duration_loss += duration_loss.item()

            val_loss = val_loss / len(self.valid_loader)
            val_mel_loss = val_mel_loss / len(self.valid_loader)
            val_postnet_mel_loss = val_postnet_mel_loss / len(self.valid_loader)
            val_pitch_loss = val_pitch_loss / len(self.valid_loader)
            val_energy_loss = val_energy_loss / len(self.valid_loader)
            val_duration_loss = val_duration_loss / len(self.valid_loader)

            self.write_losses(
                "valid",
                {
                    "total": val_loss,
                    "mel_loss": val_mel_loss,
                    "postnet_mel_loss": val_postnet_mel_loss,
                    "pitch_loss": val_pitch_loss,
                    "energy_loss": val_energy_loss,
                    "duration_loss": val_duration_loss,
                }
            )

    def generate_samples(self) -> None:

        phonemes = [
            [self.phonemes_to_id.get(p, 0) for p in sequence.split(" ")]
            for sequence in (PHONEMES_ENG if self.config.lang == "english" else PHONEMES_CHI)
        ]

        with torch.no_grad():

            for reference_path in self.references:
                for i, sequence in enumerate(phonemes):
                    phonemes_tensor = torch.LongTensor([sequence]).to(self.device)
                    num_phonemes_tensor = torch.IntTensor([len(sequence)]).to(self.device)
                    speaker = reference_path.parent.name
                    emo = reference_path.stem
                    speaker_print_file = SPEAKER_PRINT_DIR / self.config.lang / speaker / f"{emo}.npy"
                    speaker_print_array = np.load(str(speaker_print_file))
                    speaker_print_tensor = torch.FloatTensor(
                        speaker_print_array
                    ).unsqueeze(0)
                    reference = (
                        torch.load(reference_path, map_location="cpu") - self.mels_mean
                    ) / self.mels_std
                    reference = reference.unsqueeze(0)
                    batch = (
                        phonemes_tensor,
                        num_phonemes_tensor,
                        speaker_print_tensor.to(self.device),
                        reference.to(self.device).permute(0, 2, 1).float(),
                    )                    
                    output = self.fastspeech2_model.inference(batch)
                    output = output[1].permute(0, 2, 1).squeeze(0)
                    output = output * self.mels_std.to(self.device) + self.mels_mean.to(
                        self.device
                    )
                    audio = self.vocoder_inference(output.float())

                    name = f"{speaker}_{reference_path.stem}_{i}"
                    self.writer.add_audio(
                        f"Audio/Val/{name}",
                        audio.cpu(),
                        sample_rate=self.config.sample_rate,
                        global_step=self.iteration_step,
                    )
