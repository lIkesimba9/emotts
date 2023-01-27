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

from src.data_process.config import DatasetParams
from src.data_process.audio_utils import seconds_to_frame
from src.constants import REMOVE_SPEAKERS

NUMBER = Union[int, float]
PHONES_TIER = "phones"
PAD_TOKEN = "<PAD>"


@dataclass
class VoicePrintSample:

    phonemes: List[int]
    num_phonemes: int
    speaker_emb: torch.Tensor
    speaker_id: int
    durations: np.ndarray
    mels: torch.Tensor


@dataclass
class VoicePrintInfo:

    mel_path: Path
    speaker_path: Path
    speaker_id: int
    phonemes_length: int
    duration_path: Path
    text_path: Path


@dataclass
class VoicePrintBatch:

    phonemes: torch.Tensor
    num_phonemes: torch.Tensor
    speaker_embs: torch.Tensor
    speaker_ids: torch.Tensor
    durations: torch.Tensor
    mels: torch.Tensor


class VoicePrintDataset(Dataset[VoicePrintSample]):
    def __init__(
        self,
        sample_rate: int,
        hop_size: int,
        frames_per_step: int,
        duration_type: str,
        mels_mean: torch.Tensor,
        mels_std: torch.Tensor,
        phoneme_to_ids: Dict[str, int],
        data: List[VoicePrintInfo],
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

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> VoicePrintSample:

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
                                seconds_to_frame(x.end_time, 
                                    self.sample_rate, self.hop_size) - seconds_to_frame(x.start_time, self.sample_rate, self.hop_size)
                                for x in phones_tier.get_copy_with_gaps_filled()
                        ],
                        dtype=np.float32
                )
        else:
                raise ValueError("Unknown value for duration_type: " + self.duration_type)


        mels: torch.Tensor = torch.Tensor(np.load(info.mel_path)).unsqueeze(0)
        mels = (mels - self.mels_mean) / self.mels_std
        
        speaker_embs: np.ndarray = np.load(str(info.speaker_path))
        speaker_embs_tensor = torch.from_numpy(speaker_embs)

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
        ## NOTE: right now this adjustment also incorporates 
        ##            the accumulated error due to rounding of seconds into frames at each step
        ## TODO: get rid of conversion to int for Gaussian upsampling
        output_mel_length = np.ceil(durations_in_steps.sum()*self.frames_per_step)
        r_dur_pad_size = output_mel_length - mels.shape[-1]
        assert (not (r_dur_pad_size < 0))

        return VoicePrintSample(
            phonemes=phoneme_ids,
            num_phonemes=len(phoneme_ids),
            speaker_emb=speaker_embs_tensor,
            speaker_id=info.speaker_id,
            mels=mels,
            durations=durations_in_steps
        )



class VoicePrintFactory:

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
        self._dataset: List[VoicePrintInfo] = self._build_dataset()
        self.mels_mean, self.mels_std = self._get_mean_and_std()

    @staticmethod
    def add_to_mapping(mapping: Dict[str, int], token: str) -> None:
        if token not in mapping:
            mapping[token] = len(mapping)

    def split_train_valid(
        self, test_fraction: float
    ) -> Tuple[VoicePrintDataset, VoicePrintDataset]:
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
        train_dataset = VoicePrintDataset(
            sample_rate=self.sample_rate,
            hop_size=self.hop_size,
            frames_per_step=self.frames_per_step,
            duration_type=self.duration_type,
            mels_mean=self.mels_mean,
            mels_std=self.mels_std,
            phoneme_to_ids=self.phoneme_to_id,
            data=train_data,
        )
        test_dataset = VoicePrintDataset(
            sample_rate=self.sample_rate,
            hop_size=self.hop_size,
            frames_per_step=self.frames_per_step,
            duration_type=self.duration_type,
            mels_mean=self.mels_mean,
            mels_std=self.mels_std,
            phoneme_to_ids=self.phoneme_to_id,
            data=test_data,
        )
        return train_dataset, test_dataset

    def _build_dataset(self) -> List[VoicePrintInfo]:

        dataset: List[VoicePrintInfo] = []

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
        
        samples = list(mels_set & duration_set & speaker_emb_set & texts_set)
        for sample in tqdm(samples):
            if sample.parent.name in REMOVE_SPEAKERS:
                continue

            duration_path = (self._duration_dir / sample).with_suffix(self._mels_ext)
            tg_path = (self._text_dir / sample).with_suffix(self._text_ext)
            
            mels_path = (self._mels_dir / sample).with_suffix(self._mels_ext)
            speaker_emb_path = (self._speaker_emb_dir / sample).with_suffix(self._speaker_emb_ext)

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
                        VoicePrintInfo(
                            text_path=tg_path,
                            duration_path=duration_path,
                            mel_path=mels_path,
                            phonemes_length=len(phonemes),
                            speaker_id=speaker_id,
                            speaker_path=speaker_emb_path,
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


class VoicePrintCollate:
    """
    Zero-pads model inputs and targets based on max length in the batch
    """

    def __init__(self):
        pass

    def __call__(self, batch: List[VoicePrintSample]) -> VoicePrintBatch:
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
        for i, idx in enumerate(ids_sorted_decreasing):
            text = batch[idx].phonemes
            text_padded[i, : len(text)] = torch.LongTensor(text)
            durations = batch[idx].durations
            durations_padded[i, : len(durations)] = torch.FloatTensor(durations)
            speaker_emb_tensor[i] = batch[idx].speaker_emb

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

        return VoicePrintBatch(
            phonemes=text_padded,
            num_phonemes=input_lengths,
            speaker_embs=speaker_emb_tensor,
            durations=durations_padded,
            speaker_ids=input_speaker_ids,
            mels=mel_padded,
        )
