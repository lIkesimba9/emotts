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
    ENERGY_MIN_FILENAME,
    ENERGY_MAX_FILENAME,
    PITCH_MIN_FILENAME,
    PITCH_MAX_FILENAME,
    PHONEMES_FILENAME,
    REFERENCE_PATH,
    SPEAKERS_FILENAME,
    SPEAKER_PRINT_DIR,
    TEST_FILENAME,
    TRAIN_FILENAME,
    STATISTIC_FILENAME,
    DATASET_INFO_PATH
)
    #REMOVE_SPEAKERS

from src.data_process.basic_dataset import BasicBatch, BasicDataset
from src.models.fastspeech2.fastspeech2 import FastSpeech2Gaus
from src.train_config import load_config


class Inferencer:

    PAD_TOKEN = "<PAD>"
    PHONES_TIER = "phones"
    LEXICON_OOV_TOKEN = "spn"
    MEL_EXT = "npy"

    def __init__(
        self, config_path: str
    ):
        config = load_config(config_path)
        self.config = config
        checkpoint_path = CHECKPOINT_DIR / config.checkpoint_name / FASTSPEECH2_CHECKPOINT_NAME
        with open(checkpoint_path / PHONEMES_FILENAME) as f:
            self.phonemes_to_id: Dict[str, int] = json.load(f)
        with open(checkpoint_path / SPEAKERS_FILENAME) as f:
            self.speakers_to_id: Dict[str, int] = json.load(f)
        self.sample_rate = config.sample_rate
        self.hop_size = config.hop_size
        self.device = torch.device(config.device)
        self.fastspeech2_model_mels_path = Path(config.data.feature_dir)
        self.fastspeech2_model_mels_path.mkdir(parents=True, exist_ok=True)
        self.fastspeech2_model: FastSpeech2Gaus  = torch.load(
            checkpoint_path / FASTSPEECH2_MODEL_FILENAME, map_location=config.device
        )

        self.mels_mean = torch.load(checkpoint_path / MELS_MEAN_FILENAME)
        self.mels_std = torch.load(checkpoint_path / MELS_STD_FILENAME)

        self.data_info_dir = Path(DATASET_INFO_PATH) / Path(config.data.dataset_name)
        self.data_info_dir.mkdir(parents=True, exist_ok=True)
        mapping_folder = self.data_info_dir

        #with open(mapping_folder / SPEAKERS_FILENAME) as f:
        #    self.speakers_to_id.update(json.load(f))
        #with open(mapping_folder / PHONEMES_FILENAME) as f:
        #    self.phonemes_to_id.update(json.load(f))

        with open(mapping_folder / STATISTIC_FILENAME) as f:
            self.statistic_dict = json.load(f)
        
        self.mels_mean = torch.load(mapping_folder / MELS_MEAN_FILENAME)
        self.mels_std = torch.load(mapping_folder / MELS_STD_FILENAME)
        
        self.energy_min = torch.load(mapping_folder / ENERGY_MIN_FILENAME)
        self.energy_max = torch.load(mapping_folder / ENERGY_MAX_FILENAME)
        
        self.pitch_min = torch.load(mapping_folder / PITCH_MIN_FILENAME)
        self.pitch_max = torch.load(mapping_folder / PITCH_MAX_FILENAME)


    def proceed_data(self) -> None:

        self.fastspeech2_model = self.fastspeech2_model.eval()

        train_dataset = BasicDataset(
            sample_rate=self.config.sample_rate,
            hop_size=self.config.hop_size,
            mels_mean=self.mels_mean,
            mels_std=self.mels_std,
            statistic_dict=self.statistic_dict,
            energy_min=self.energy_min, 
            energy_max=self.energy_max,
            pitch_min=self.pitch_min, 
            pitch_max=self.pitch_max,
            phoneme_to_ids=self.phonemes_to_id,
            path_to_train_json=self.data_info_dir / TRAIN_FILENAME,
            pitch_norm=self.config.data.pitch_norm,
            energy_norm=self.config.data.energy_norm,
        )

        for sample in train_dataset:
            speaker_name = sample.wav_id.split('_')[0]
            save_dir = self.fastspeech2_model_mels_path / speaker_name
            save_dir.mkdir(exist_ok=True)
            filepath = save_dir / f"{sample.wav_id}.{self.MEL_EXT}"
            with torch.no_grad():
                batch = BasicBatch(
                    speaker_ids=torch.LongTensor(np.array([sample.speaker_id])).to(self.device),
                    phonemes=torch.LongTensor(np.array([sample.phonemes])).to(self.device),
                    num_phonemes=torch.LongTensor(np.array([sample.num_phonemes])).to(self.device),
                    mels_lens=torch.LongTensor(np.array([sample.mels.shape[1]])).to(self.device),
                    mels=sample.mels.unsqueeze(0).permute(0, 2, 1).to(self.device).float(),
                    energies=torch.Tensor(np.array([sample.energy])).to(self.device),
                    pitches=torch.Tensor(np.array([sample.pitch])).to(self.device),
                    durations=torch.Tensor(np.array([sample.durations])).to(self.device),
                    speaker_embs=sample.speaker_emb.unsqueeze(0).to(self.device),
                )
                _, output, _, _, _, _, _, _ = self.fastspeech2_model(batch)
                output = output.permute(0, 2, 1).squeeze(0)
                output = output * self.mels_std.to(self.device) + self.mels_mean.to(self.device)

            #torch.save(output.float(), filepath)
            np.save(filepath, output.float().cpu().detach().numpy())
        print('base inferencer done')
