from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union
from torch.utils.data import Dataset

from src.data_process.audio_utils import seconds_to_frame

import numpy as np
import torch
import tgt

from .utils import load_json, PHONES_TIER


@dataclass
class BasicSample:

    phonemes: List[int]
    num_phonemes: int
    speaker_id: int
    durations: np.array
    energy: np.array
    pitch: np.array
    mels: torch.Tensor
    speaker_emb: torch.Tensor
    wav_id: str

@dataclass
class BasicBatch:

    speaker_embs: torch.Tensor
    speaker_ids: torch.Tensor
    phonemes: torch.Tensor
    num_phonemes: torch.Tensor
    mels: torch.Tensor
    mels_lens: torch.Tensor
    energies: torch.Tensor
    pitches: torch.Tensor
    durations: torch.Tensor




class BasicDataset(Dataset[BasicSample]):
    def __init__(
        self,
        sample_rate: int,
        hop_size: int,
        frames_per_step: int,
        duration_type: str,
        mels_mean: torch.Tensor,
        mels_std: torch.Tensor,
        statistic_dict: Dict,
        energy_min: float,
        energy_max: float,
        pitch_min: float,
        pitch_max: float,
        phoneme_to_ids: Dict[str, int],
        path_to_train_json: Path,
        pitch_norm: bool,
        energy_norm: bool,
    ):
        self._phoneme_to_id = phoneme_to_ids
        self._dataset = load_json(path_to_train_json)
        self._dataset.sort(key=lambda x: x["phonemes_length"])
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.frames_per_step = frames_per_step
        if (self.frames_per_step is None):
            self.frames_per_step = 1
        self.duration_type = duration_type
        self.mels_mean = mels_mean
        self.mels_std = mels_std
        self.statistic_dict = statistic_dict
        self.energy_min = energy_min
        self.energy_max = energy_max
        self.pitch_min = pitch_min 
        self.pitch_max = pitch_max
        self.pitch_norm = pitch_norm
        self.energy_norm = energy_norm


    def __len__(self) -> int:
        return len(self._dataset)


    def __getitem__(self, idx: int) -> BasicSample:

        info = self._dataset[idx]
        text_grid = tgt.read_textgrid(info["text_path"])
        phones_tier = text_grid.get_tier_by_name(PHONES_TIER)
        phoneme_ids = [
            self._phoneme_to_id[x.text] for x in phones_tier.get_copy_with_gaps_filled()
        ]
        phonemes = [x.text for x in phones_tier.get_copy_with_gaps_filled()]

        ## loading number of frames
        ## TODO: Best to change to number of seconds and float32 in later versions
        if (self.duration_type == "int"):
            durations_in_frames = np.array(
                        [
                                int(np.round(seconds_to_frame(x.end_time,
                                    self.sample_rate, self.hop_size))) - int(np.round(seconds_to_frame(x.start_time,
self.sample_rate, self.hop_size)))
                                for x in phones_tier.get_copy_with_gaps_filled()
                        ],
                        dtype=np.float32
                )
        elif (self.duration_type == "float"):
            durations_in_frames = np.array(
                        [
                                seconds_to_frame(x.end_time,
                                    self.sample_rate, self.hop_size) - seconds_to_frame(x.start_time, self.sample_rat
e, self.hop_size)
                                for x in phones_tier.get_copy_with_gaps_filled()
                        ],
                        dtype=np.float32
                )
        elif (self.duration_type == "from_disk"):
            durations_in_frames: np.array = np.load(info["duration_path"])    
        else:
            raise ValueError("Unknown value for duration_type: " + self.duration_type)

        durations = durations_in_frames

        mels: torch.Tensor = torch.Tensor(np.load(info["mel_path"]))
        mels = (mels - self.mels_mean) / self.mels_std

        pad_size = mels.shape[-1] - np.int64(durations.sum())
        durations[-1] += pad_size
        assert durations[-1] >= 0      

        energy = np.load(info["energy_path"])
        nonzero_idxs = np.where(energy != 0)[0]
        energy[nonzero_idxs] = np.log(energy[nonzero_idxs])

        pitch = np.load(info["pitch_path"])
        nonzero_idxs = np.where(pitch != 0)[0]
        pitch[nonzero_idxs] = np.log(pitch[nonzero_idxs])

        for i, phoneme in enumerate(phonemes):
            if self.energy_norm:
                energy[i] = (energy[i] - float(self.statistic_dict[phoneme]["pitch_mean"])) / float(self.statistic_dict[phoneme]["pitch_std"])
            if self.pitch_norm:
                pitch[i] = (pitch[i] - float(self.statistic_dict[phoneme]["pitch_mean"])) / float(self.statistic_dict[phoneme]["pitch_std"])
                

        speaker_embs: np.ndarray = np.load(info["speaker_path"])
        speaker_embs_tensor = torch.from_numpy(speaker_embs)



        return BasicSample(
            phonemes=phoneme_ids,
            num_phonemes=len(phoneme_ids),
            speaker_id=info["speaker_id"],
            mels=mels,
            durations=durations,
            energy=energy,
            pitch=pitch,
            wav_id=Path(info["mel_path"]).stem,
            speaker_emb=speaker_embs_tensor,
        )




class BasicCollate:
    """
    Zero-pads model inputs and targets based on number of frames per setep
    """


    def __call__(self, batch: List[BasicSample]) -> BasicBatch:
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [{...}, {...}, ...]
        """
        # Right zero-pad all one-hot text sequences to max input length
        batch_size = len(batch)
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x.phonemes) for x in batch]), dim=0, descending=True,
        )
        max_input_len = int(input_lengths[0])

        input_speaker_ids = torch.LongTensor(
            [batch[i].speaker_id for i in ids_sorted_decreasing]
        )

        speaker_emb_size = batch[0].speaker_emb.shape[0]

        text_padded = torch.zeros((batch_size, max_input_len), dtype=torch.long)
        durations_padded = torch.zeros((batch_size, max_input_len), dtype=torch.float)
        energy_padded = torch.zeros((batch_size, max_input_len), dtype=torch.float)
        pitch_padded = torch.zeros((batch_size, max_input_len), dtype=torch.float)
        speaker_emb_tensor = torch.zeros((batch_size, speaker_emb_size))

        for i, idx in enumerate(ids_sorted_decreasing):
            text = batch[idx].phonemes
            text_padded[i, : len(text)] = torch.LongTensor(text)
            durations = batch[idx].durations
            durations_padded[i, : len(durations)] = torch.FloatTensor(durations)
            energy = batch[idx].energy
            energy_padded[i, : len(energy)] = torch.FloatTensor(energy)
            pitch = batch[idx].pitch
            pitch_padded[i, : len(pitch)] = torch.FloatTensor(pitch)
            speaker_emb_tensor[i] = batch[idx].speaker_emb

        num_mels = batch[0].mels.size(0)
        max_target_len = max([x.mels.size(1) for x in batch])
        mels_lens = torch.LongTensor([x.mels.size(1) for x in batch])


        # include mel padded and gate padded
        mel_padded = torch.zeros(
            (batch_size, num_mels, max_target_len), dtype=torch.float
        )
        for i, idx in enumerate(ids_sorted_decreasing):
            mels: torch.Tensor = batch[idx].mels
            mel_padded[i, :, : mels.shape[1]] = mels
        mel_padded = mel_padded.permute(0, 2, 1)

        return BasicBatch(
            speaker_ids=input_speaker_ids,
            phonemes=text_padded,
            num_phonemes=input_lengths,
            mels_lens=mels_lens,
            mels=mel_padded,
            energies=energy_padded,
            pitches=pitch_padded,
            durations=durations_padded,
            speaker_embs=speaker_emb_tensor,
        )
