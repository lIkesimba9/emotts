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
from src.data_process.fastspeech2_dataset_voiceprint import FastSpeech2VoicePrintBatch, FastSpeech2VoicePrintFactory
from src.models.fastspeech2.fastspeech2 import (
    FastSpeech2Dutaion,
)
from src.train_config import load_config


class Inferencer:

    PAD_TOKEN = "<PAD>"
    PHONES_TIER = "phones"
    LEXICON_OOV_TOKEN = "spn"
    #MEL_EXT = "pth"
    MEL_EXT = "npy"

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
        self.fastspeech2_model_mels_path = Path(config.data.feature_dir)
        self.fastspeech2_model_mels_path.mkdir(parents=True, exist_ok=True)
        self.fastspeech2_model: FastSpeech2Dutaion  = torch.load(
            checkpoint_path / FASTSPEECH2_MODEL_FILENAME, map_location=config.device
        )

        self.mels_mean = torch.load(checkpoint_path / MELS_MEAN_FILENAME)
        self.mels_std = torch.load(checkpoint_path / MELS_STD_FILENAME)



    def proceed_data(self) -> None:
        factory = FastSpeech2VoicePrintFactory(
            sample_rate=self.config.sample_rate,
            hop_size=self.config.hop_size,
            n_mels=self.config.n_mels,
            config=self.config.data,
            phonemes_to_id=self.phonemes_to_idx,
            speakers_to_id=self.speakers_to_idx,
            ignore_speakers=self.config.data.ignore_speakers,
            finetune=self.config.finetune,
        )
        self.phonemes_to_id = factory.phoneme_to_id
        self.speakers_to_id = factory.speaker_to_id
        trainset, valset = factory.split_train_valid(0)
        self.fastspeech2_model = self.fastspeech2_model.eval()

        for sample in trainset:
            save_dir = self.fastspeech2_model_mels_path / sample.speaker_id_str
            save_dir.mkdir(exist_ok=True)
            filepath = save_dir / f"{sample.wav_id}.{self.MEL_EXT}"
            with torch.no_grad():
                batch = FastSpeech2VoicePrintBatch(
                    speaker_ids=torch.LongTensor(np.array([sample.speaker_id])).to(self.device),
                    phonemes=torch.LongTensor(np.array([sample.phonemes])).to(self.device),
                    num_phonemes=torch.LongTensor(np.array([sample.num_phonemes])).to(self.device),
                    mels_lens=torch.LongTensor(np.array([sample.mel.shape[1]])).to(self.device),
                    mels= sample.mel.unsqueeze(0).permute(0, 2, 1).float().to(self.device),
                    energies=torch.Tensor(np.array([sample.energy])).to(self.device),
                    pitches=torch.Tensor(np.array([sample.pitch])).to(self.device),
                    durations=torch.Tensor(np.array([sample.duration])).to(self.device),
                    speaker_embs=sample.speaker_emb.unsqueeze(0).to(self.device),
                )
                _, output, *_ = self.fastspeech2_model(batch)
                output = output.permute(0, 2, 1).squeeze(0)
                output = output * self.mels_std.to(self.device) + self.mels_mean.to(self.device)

            #torch.save(output.float(), filepath)
            np.save(filepath, output.float().cpu().detach().numpy())
