import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import tgt
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from src.data_process.config import DatasetParams
from src.data_process.audio_utils import seconds_to_frame
from src.constants import REMOVE_SPEAKERS

NUMBER = Union[int, float]
PHONES_TIER = "phones"
PAD_TOKEN = "<PAD>"


@dataclass
class VoicePrintVarianceSample:

    phonemes: List[int]
    num_phonemes: int
    speaker_emb: torch.Tensor
    speaker_id: int
    durations: np.ndarray
    mels: torch.Tensor
    energy: np.ndarray
    pitch: np.ndarray


@dataclass
class VoicePrintVarianceInfo:

    mel_path: Path
    speaker_path: Path
    speaker_id: int
    phonemes_length: int
    duration_path: Path
    text_path: Path
    pitch_path: Path
    energy_path: Path


@dataclass
class VoicePrintVarianceBatch:

    phonemes: torch.Tensor
    num_phonemes: torch.Tensor
    speaker_embs: torch.Tensor
    speaker_ids: torch.Tensor
    durations: torch.Tensor
    mels: torch.Tensor
    pitches: torch.Tensor
    energies: torch.Tensor


class VoicePrintVarianceDataset(Dataset[VoicePrintVarianceSample]):
    def __init__(
        self,
        sample_rate: int,
        hop_size: int,
        frames_per_step: int,
        duration_type: str,
        mels_mean: torch.Tensor,
        mels_std: torch.Tensor,
        energy_mean: float,
        energy_std: float,
        pitch_mean: float,
        pitch_std: float,
        energy_min: float,
        energy_max: float,
        pitch_min: float,
        pitch_max: float,
        phoneme_to_ids: Dict[str, int],
        data: List[VoicePrintVarianceInfo],
    ):
        self._phoneme_to_id = phoneme_to_ids
        self._dataset = data
        self._dataset.sort(key=lambda x: x.phonemes_length)
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.frames_per_step = frames_per_step
        self.duration_type = duration_type
        self.mels_mean = mels_mean
        self.mels_std = mels_std
        self.energy_mean = energy_mean
        self.energy_std = energy_std
        self.pitch_mean = pitch_mean
        self.pitch_std = pitch_std
        self.energy_min = energy_min
        self.energy_max = energy_max
        self.pitch_min = pitch_min 
        self.pitch_max = pitch_max 

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> VoicePrintVarianceSample:

        info = self._dataset[idx]
        text_grid = tgt.read_textgrid(info.text_path)
        phones_tier = text_grid.get_tier_by_name(PHONES_TIER)
        phoneme_ids = [
            self._phoneme_to_id[x.text] for x in phones_tier.get_copy_with_gaps_filled()
        ]
        ## loading number of frames
        ## TODO: Best to change to number of seconds and float32 in later versions
        ##durations_in_frames = np.load(info.duration_path)
        if (self.duration_type == "int"):
                durations_in_frames = np.array(
                        [
                                int(np.round(seconds_to_frame(x.end_time, 
                                    self.sample_rate, self.hop_size))) - int(np.round(seconds_to_frame(x.start_time, self.sample_rate, self.hop_size)))
                                for x in phones_tier.get_copy_with_gaps_filled()
                        ],
                        dtype=np.float32
                )
        elif (self.duration_type == "float"):
                durations_in_frames = np.array(
                        [
                                seconds_to_frame(x.end_time, self.sample_rate, self.hop_size) - seconds_to_frame(x.start_time, self.sample_rate, self.hop_size)
                                for x in phones_tier.get_copy_with_gaps_filled()
                        ],
                        dtype=np.float32
                )
        else:
                raise ValueError("Unknown value for duration_type: " + self.duration_type)


        mels: torch.Tensor = torch.Tensor(np.load(info.mel_path)).unsqueeze(0)
        mels = (mels - self.mels_mean) / self.mels_std

        pad_size = mels.shape[-1] - np.int64(durations_in_frames.sum())
        durations_in_frames[-1] += pad_size
        assert durations_in_frames[-1] >= 0

        durations_in_steps = durations_in_frames
        ## convert number of frames to number of decoder steps
        if (self.duration_type == "int"):
            if (self.frames_per_step > 1):
                durations_in_steps = np.ceil(durations_in_frames.astype('float32')/self.frames_per_step)
            durations_in_steps = durations_in_steps.astype(np.int32)
        elif (self.duration_type == "float"):
            if (self.frames_per_step > 1):
                durations_in_steps = durations_in_frames.astype('float32')/self.frames_per_step

        ## adjust mel length according to the number of decoder steps
        output_mel_length = np.ceil(durations_in_steps.sum()*self.frames_per_step)
        r_pad_size = output_mel_length - mels.shape[-1]
        assert (not (r_pad_size < 0))
        
        energy = np.load(info.energy_path)
        nonzero_idxs = np.where(energy != 0)[0]
        energy[nonzero_idxs] = np.log(energy[nonzero_idxs])
     
        pitch = np.load(info.pitch_path)
        nonzero_idxs = np.where(pitch != 0)[0]
        pitch[nonzero_idxs] = np.log(pitch[nonzero_idxs])
        
        speaker_embs: np.ndarray = np.load(str(info.speaker_path))
        speaker_embs_tensor = torch.from_numpy(speaker_embs)

        return VoicePrintVarianceSample(
            phonemes=phoneme_ids,
            num_phonemes=len(phoneme_ids),
            speaker_emb=speaker_embs_tensor,
            speaker_id=info.speaker_id,
            mels=mels,
            durations=durations_in_steps,
            energy=energy,
           pitch=pitch
        )



class VoicePrintVarianceFactory:

    _speaker_emb_ext = ".npy"

    def __init__(
        self,
        sample_rate: int,
        hop_size: int,
        frames_per_step: int,
        duration_type: str,
        n_mels: int,
        config: DatasetParams,
        phonemes_to_id: Dict[str, int],
        speakers_to_id: Dict[str, int],
        ignore_speakers: List[str],
        finetune: bool,
    ):
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.frames_per_step = frames_per_step
        self.duration_type = duration_type
        self.n_mels = n_mels
        self.finetune = finetune
        self._mels_dir = Path(config.mels_dir)
        self._duration_dir = Path(config.duration_dir)
        self._text_ext = config.text_ext
        self._text_dir = Path(config.text_dir)
        self._speaker_emb_dir = Path(config.speaker_emb_dir)
        self._energy_dir = Path(config.energy_dir)
        self._pitch_dir = Path(config.pitch_dir)
        self._mels_ext = config.mels_ext
        self.phoneme_to_id: Dict[str, int] = phonemes_to_id
        self.speaker_to_id: Dict[str, int] = speakers_to_id
        self.phoneme_to_id[PAD_TOKEN] = 0
        self.ignore_speakers = ignore_speakers
        if finetune:
            self.speaker_to_use = config.finetune_speakers
        else:
            self.speaker_to_use = [
                speaker.name
                for speaker in self._mels_dir.iterdir()
                if speaker not in config.finetune_speakers
            ]
        self._dataset: List[VoicePrintVarianceInfo] = self._build_dataset()
        self.mels_mean, self.mels_std = self._get_mean_and_std()

        self.energy_mean, self.energy_std = self._get_mean_and_std_scalar(self._energy_dir, self._mels_ext)
        self.energy_min, self.energy_max = self._get_min_max(self._energy_dir, self._mels_ext, self.energy_mean, self.energy_std)
        
        #self.energy_min = (self.energy_min - self.energy_mean) / self.energy_std
        #self.energy_max = (self.energy_max - self.energy_mean) / self.energy_std
        if self.energy_min != 0:
            self.energy_min = np.log(self.energy_min)
        self.energy_max = np.log(self.energy_max)

        self.pitch_mean, self.pitch_std = self._get_mean_and_std_scalar(self._pitch_dir, self._mels_ext)
        self.pitch_min, self.pitch_max = self._get_min_max(self._pitch_dir, self._mels_ext, self.pitch_mean, self.pitch_std)
        
        #self.pitch_min = (self.pitch_min - self.pitch_mean) / self.pitch_std
        #self.pitch_max = (self.pitch_max - self.pitch_mean) / self.pitch_std
        if self.pitch_min != 0:
            self.pitch_min = np.log(self.pitch_min)
        self.pitch_max = np.log(self.pitch_max)

    @staticmethod
    def add_to_mapping(mapping: Dict[str, int], token: str) -> None:
        if token not in mapping:
            mapping[token] = len(mapping)

    def split_train_valid(
        self, test_fraction: float
    ) -> Tuple[VoicePrintVarianceDataset, VoicePrintVarianceDataset]:
        speakers_to_data_id: Dict[int, List[int]] = defaultdict(list)
        ignore_speaker_ids = {
            self.speaker_to_id[speaker] for speaker in self.ignore_speakers
        }
        for i, sample in enumerate(self._dataset):
            speakers_to_data_id[sample.speaker_id].append(i)
        test_ids: List[int] = []
        for speaker, ids in speakers_to_data_id.items():
            test_size = int(len(ids) * test_fraction)
            if test_size > 0 and speaker not in ignore_speaker_ids:
                test_indexes = random.choices(ids, k=test_size)
                test_ids.extend(test_indexes)

        train_data = []
        test_data = []
        for i in range(len(self._dataset)):
            if i in test_ids:
                test_data.append(self._dataset[i])
            else:
                train_data.append(self._dataset[i])
        train_dataset = VoicePrintVarianceDataset(
            sample_rate=self.sample_rate,
            hop_size=self.hop_size,
            frames_per_step=self.frames_per_step,
            duration_type=self.duration_type,
            mels_mean=self.mels_mean,
            mels_std=self.mels_std,
            phoneme_to_ids=self.phoneme_to_id,
            data=train_data,
            energy_mean=self.energy_mean,
            energy_std=self.energy_std,
            pitch_mean=self.pitch_mean,
            pitch_std=self.pitch_std,
            energy_min=self.energy_min, 
            energy_max=self.energy_max,
            pitch_min=self.pitch_min, 
            pitch_max=self.pitch_max,
        )
        test_dataset = VoicePrintVarianceDataset(
            sample_rate=self.sample_rate,
            hop_size=self.hop_size,
            frames_per_step=self.frames_per_step,
            duration_type=self.duration_type,
            mels_mean=self.mels_mean,
            mels_std=self.mels_std,
            phoneme_to_ids=self.phoneme_to_id,
            data=test_data,
            energy_mean=self.energy_mean,
            energy_std=self.energy_std,
            pitch_mean=self.pitch_mean,
            pitch_std=self.pitch_std,
            energy_min=self.energy_min, 
            energy_max=self.energy_max,
            pitch_min=self.pitch_min, 
            pitch_max=self.pitch_max,
        )
        return train_dataset, test_dataset

    def _build_dataset(self) -> List[VoicePrintVarianceInfo]:

        dataset: List[VoicePrintVarianceInfo] = []

        mels_set = {
            Path(x.parent.name) / x.stem
            for x in self._mels_dir.rglob(f"*{self._mels_ext}")
        }
        speaker_emb_set = {
            Path(x.parent.name) / x.stem
            for x in self._speaker_emb_dir.rglob(f"*{self._speaker_emb_ext}")
        }
        duration_set = {
            Path(x.parent.name) / x.stem
            for x in self._duration_dir.rglob(f"*{self._mels_ext}")
        }
        texts_set = {
            Path(x.parent.name) / x.stem
            for x in self._text_dir.rglob(f"*{self._text_ext}")
        }
        enegry_set = {
            Path(x.parent.name) / x.stem
            for x in self._energy_dir.rglob(f"*{self._mels_ext}")
        }
        pitch_set = {
            Path(x.parent.name) / x.stem
            for x in self._pitch_dir.rglob(f"*{self._mels_ext}")
        }
        samples = list(mels_set & duration_set & speaker_emb_set & texts_set & pitch_set & enegry_set)
        for sample in tqdm(samples):
            if sample.parent.name in REMOVE_SPEAKERS:
                continue

            duration_path = (self._duration_dir / sample).with_suffix(self._mels_ext)
            tg_path = (self._text_dir / sample).with_suffix(self._text_ext)
            
            mels_path = (self._mels_dir / sample).with_suffix(self._mels_ext)
            speaker_emb_path = (self._speaker_emb_dir / sample).with_suffix(self._speaker_emb_ext)

            energy_path = (self._energy_dir / sample).with_suffix(self._mels_ext)
            pitch_path = (self._pitch_dir / sample).with_suffix(self._mels_ext)

            text_grid = tgt.read_textgrid(tg_path)

            self.add_to_mapping(self.speaker_to_id, sample.parent.name)
            speaker_id = self.speaker_to_id[sample.parent.name]
            
            if PHONES_TIER in text_grid.get_tier_names():

                phones_tier = text_grid.get_tier_by_name(PHONES_TIER)
                phonemes = [x.text for x in phones_tier.get_copy_with_gaps_filled()]
                if "spn" in phonemes:
                    continue
                for phoneme in phonemes:
                    self.add_to_mapping(self.phoneme_to_id, phoneme)


                if sample.parent.name in self.speaker_to_use:

                    dataset.append(
                        VoicePrintVarianceInfo(
                            text_path=tg_path,
                            duration_path=duration_path,
                            mel_path=mels_path,
                            phonemes_length=len(phonemes),
                            speaker_id=speaker_id,
                            speaker_path=speaker_emb_path,
                            energy_path=energy_path,
                            pitch_path=pitch_path
                        )
                    )

        return dataset

    def _get_mean_and_std(self) -> Tuple[torch.Tensor, torch.Tensor]:
        mel_sum = torch.zeros(self.n_mels, dtype=torch.float64)
        mel_squared_sum = torch.zeros(self.n_mels, dtype=torch.float64)
        counts = 0

        for mel_path in self._mels_dir.rglob(f"*{self._mels_ext}"):
            if mel_path.parent.name in REMOVE_SPEAKERS:
                continue
            mels: torch.Tensor = torch.Tensor(np.load(mel_path)).squeeze(0)
            mel_sum += mels.sum(dim=-1).squeeze(0)
            mel_squared_sum += (mels ** 2).sum(dim=-1).squeeze(0)
            counts += mels.shape[-1]

        mels_mean: torch.Tensor = mel_sum / counts
        mels_std: torch.Tensor = torch.sqrt(
            (mel_squared_sum - mel_sum * mel_sum / counts) / counts
        )

        return mels_mean.view(-1, 1), mels_std.view(-1, 1)

    def _remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values >= lower, values <= upper)
        return values[normal_indices]


    def _get_mean_and_std_scalar(self, scalars_path: Path, scalar_exts: str) -> Tuple[float, float]:
        scaler = StandardScaler()
        for scalar_path in scalars_path.rglob(f"*{scalar_exts}"):
            if scalar_path.parent.name in REMOVE_SPEAKERS:
                continue

            arr = self._remove_outlier(np.load(scalar_path))

            if (arr.shape[0] == 0):
                print(scalar_path)
                continue
                
            scaler.partial_fit(arr.reshape((-1, 1)))
        mean = scaler.mean_[0]
        std = scaler.scale_[0]
        return mean, std
    
    def _get_min_max(self, in_dir: Path, scalar_exts: str, mean, std):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for filename in in_dir.rglob(f"*{scalar_exts}"):
            values = self._remove_outlier(np.load(filename))

            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))

        return min_value, max_value


class VoicePrintVarianceCollate:
    """
    Zero-pads model inputs and targets based on max length in the batch
    """

    def __init__(self):
        pass

    def __call__(self, batch: List[VoicePrintVarianceSample]) -> VoicePrintVarianceBatch:
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
        speaker_emb_size = batch[0].speaker_emb.shape[0]

        input_speaker_ids = torch.LongTensor(
            [batch[i].speaker_id for i in ids_sorted_decreasing]
        )

        text_padded = torch.zeros((batch_size, max_input_len), dtype=torch.long)
        durations_padded = torch.zeros((batch_size, max_input_len), dtype=torch.float)
        speaker_emb_tensor = torch.zeros((batch_size, speaker_emb_size))

        energy_padded = torch.zeros((batch_size, max_input_len), dtype=torch.float)
        pitch_padded = torch.zeros((batch_size, max_input_len), dtype=torch.float)

        for i, idx in enumerate(ids_sorted_decreasing):
            text = batch[idx].phonemes
            text_padded[i, : len(text)] = torch.LongTensor(text)
            durations = batch[idx].durations
            durations_padded[i, : len(durations)] = torch.FloatTensor(durations)
            speaker_emb_tensor[i] = batch[idx].speaker_emb
            energy = batch[idx].energy
            energy_padded[i, : len(energy)] = torch.FloatTensor(energy)
            pitch = batch[idx].pitch
            pitch_padded[i, : len(pitch)] = torch.FloatTensor(pitch)

        num_mels = batch[0].mels.squeeze(0).size(0)
        max_target_len = max([x.mels.squeeze(0).size(1) for x in batch])

        # include mel padded and gate padded
        mel_padded = torch.zeros(
            (batch_size, num_mels, max_target_len), dtype=torch.float
        )
        for i, idx in enumerate(ids_sorted_decreasing):
            mel: torch.Tensor = batch[idx].mels.squeeze(0)
            mel_padded[i, :, : mel.shape[1]] = mel
        mel_padded = mel_padded.permute(0, 2, 1)

        return VoicePrintVarianceBatch(
            phonemes=text_padded,
            num_phonemes=input_lengths,
            speaker_embs=speaker_emb_tensor,
            durations=durations_padded,
            speaker_ids=input_speaker_ids,
            mels=mel_padded,
            energies=energy_padded,
            pitches=pitch_padded
        )
