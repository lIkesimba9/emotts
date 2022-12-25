import json
from pathlib import Path
from typing import Dict

import numpy as np
import tgt
import torch
from tqdm import tqdm

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
    REMOVE_SPEAKERS
)
from src.data_process.fastspeech2_dataset import FastSpeech2Batch, FastSpeech2Collate, FastSpeech2Factory

from src.models.fastspeech2.fastspeech2_no_vp_no_va import FastSpeech2

from src.train_config import load_config


class Inferencer:

    PAD_TOKEN = "<PAD>"
    PHONES_TIER = "phones"
    LEXICON_OOV_TOKEN = "spn"
    MEL_EXT = "pth"

    def __init__(
        self, config_path: str
    ):
        config = load_config(config_path)
        self.config = config
        checkpoint_path = CHECKPOINT_DIR / config.checkpoint_name / FASTSPEECH2_CHECKPOINT_NAME
        with open(checkpoint_path / PHONEMES_FILENAME) as f:
            self.phonemes_to_idx: Dict[str, int] = json.load(f)
        with open(checkpoint_path / SPEAKERS_FILENAME) as f:
            self.speakers_to_idx: Dict[str, int] = json.load(f)
        self.sample_rate = config.sample_rate
        self.hop_size = config.hop_size
        self.device = torch.device(config.device)
        self.mels_mean = torch.load(checkpoint_path / MELS_MEAN_FILENAME)
        self.mels_std = torch.load(checkpoint_path / MELS_STD_FILENAME)
        self.fastspeech2_model_mels_path = Path(config.data.feature_dir)
        self.fastspeech2_model_mels_path.mkdir(parents=True, exist_ok=True)

        self.energy_mean = torch.load(checkpoint_path / ENERGY_MEAN_FILENAME)
        self.energy_std = torch.load(checkpoint_path / ENERGY_STD_FILENAME)
        self.energy_min = torch.load(checkpoint_path / ENERGY_MIN_FILENAME)
        self.energy_max = torch.load(checkpoint_path / ENERGY_MAX_FILENAME)
        
        self.pitch_mean = torch.load(checkpoint_path / PITCH_MEAN_FILENAME)
        self.pitch_std = torch.load(checkpoint_path / PITCH_STD_FILENAME)
        self.pitch_min = torch.load(checkpoint_path / PITCH_MIN_FILENAME)
        self.pitch_max = torch.load(checkpoint_path / PITCH_MAX_FILENAME)
        
        self.fastspeech2_model = FastSpeech2(
            config=self.config.fastspeech2,
            n_mel_channels=self.config.n_mels,
            n_phonems=len(self.phonemes_to_idx),
            n_speakers=len(self.speakers_to_idx),
            pitch_min=self.pitch_min,
            pitch_max=self.pitch_max,
            energy_min=self.energy_min,
            energy_max=self.energy_max,
            gst_config=self.config.gst_config,
            finetune=self.config.finetune,
            variance_adaptor=self.config.variance_adapter_params
        ).to(self.device)

        self.fastspeech2_model = torch.load(
            checkpoint_path / FASTSPEECH2_MODEL_FILENAME, map_location=self.device
        )
#        if isinstance(self.fastspeech2_model.attention.eps, float):
#            self.fastspeech2_model.attention.eps = torch.Tensor([self.fastspeech2_model.attention.eps])


    def proceed_data(self) -> None:
        print("Loading data...")
        factory = FastSpeech2Factory(
            sample_rate=self.config.sample_rate,
            hop_size=self.config.hop_size,
            n_mels=self.config.n_mels,
            config=self.config.data,
            phonemes_to_id=self.phonemes_to_idx,
            speakers_to_id=self.speakers_to_idx,
            ignore_speakers=self.config.data.ignore_speakers,
            finetune=self.config.finetune,
        )

        trainset, valset = factory.split_train_valid(0)
        self.fastspeech2_model = self.fastspeech2_model.eval()
        print(f"Generate mels in {self.fastspeech2_model_mels_path}...")
        for sample in tqdm(trainset):

            save_dir = self.fastspeech2_model_mels_path / sample.speaker_id_str
            save_dir.mkdir(exist_ok=True)
            filepath = save_dir / f"{sample.wav_id}.{self.MEL_EXT}"
            with torch.no_grad():

                phonemes_tensor = torch.LongTensor([sample.phonemes]).to(self.device)
                num_phonemes_tensor = torch.IntTensor([sample.num_phonemes]).to(self.device)
                """
                batch = (
                    phonemes_tensor,
                    num_phonemes_tensor,
                    torch.LongTensor([sample.speaker_id]).to(self.device),
                    sample.mel.unsqueeze(0).permute(0, 2, 1).to(self.device)
                )"""
                batch = FastSpeech2Batch(
                    speaker_ids=torch.LongTensor(np.array([sample.speaker_id])).to(self.device),
                    phonemes=phonemes_tensor,
                    num_phonemes=num_phonemes_tensor,
                    mels_lens=torch.IntTensor(np.array([sample.mel.shape[1]])).to(self.device),
                    mels=sample.mel.unsqueeze(0).permute(0, 2, 1).to(self.device).float(),
                    energies=torch.FloatTensor(np.array([sample.energy])).to(self.device),
                    pitches=torch.FloatTensor(np.array([sample.pitch])).to(self.device),
                    durations=torch.FloatTensor(np.array([sample.duration])).to(self.device),
                )

                (
                    mel_predictions,
                    postnet_mel_predictions,
                    log_duration_predictions,
                    src_masks,
                    mel_masks,
                    gst_emb
                ) = self.fastspeech2_model(batch)
          
                #output = self.fastspeech2_model.inference(batch)
                output = postnet_mel_predictions.permute(0, 2, 1).squeeze(0)
                output = output * self.mels_std.to(self.device) + self.mels_mean.to(self.device)

            torch.save(output.float(), filepath)

