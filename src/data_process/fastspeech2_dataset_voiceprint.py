import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from src.data_process.config import DatasetParams
from src.constants import REMOVE_SPEAKERS

NUMBER = Union[int, float]
PHONES_TIER = "phones"
PAD_TOKEN = "<PAD>"


@dataclass
class FastSpeech2VoicePrintSample:

    phonemes: List[int]
    num_phonemes: int
    speaker_id: int
    duration: np.array
    mel: torch.Tensor
    energy: np.array
    pitch: np.array
    speaker_id_str: str
    wav_id: str
    speaker_emb: torch.Tensor



@dataclass
class FastSpeech2VoicePrintInfo:

    phonemes_path: Path
    mel_path: Path
    energy_path: Path
    duration_path: Path
    pitch_path: Path
    speaker_id: int
    phonemes_length: int
    speaker_path: Path


@dataclass
class FastSpeech2VoicePrintBatch:
    
    speaker_ids: torch.Tensor
    phonemes: torch.Tensor
    num_phonemes: torch.Tensor
    mels: torch.Tensor
    mels_lens: torch.Tensor
    energies: torch.Tensor
    pitches: torch.Tensor
    durations: torch.Tensor
    speaker_embs: torch.Tensor




class FastSpeech2VoicePrintDataset(Dataset[FastSpeech2VoicePrintSample]):
    def __init__(
        self,
        sample_rate: int,
        hop_size: int,
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
        data: List[FastSpeech2VoicePrintInfo],
    ):
        self._phoneme_to_id = phoneme_to_ids
        self._dataset = data
        self._dataset.sort(key=lambda x: x.phonemes_length)
        self.sample_rate = sample_rate
        self.hop_size = hop_size
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

    def is_outlier(self, x: np.array, p25, p75):
        """Check if value is an outlier."""
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)

        return x < lower or x > upper

    def normalize(self, array: np.array, mean: float, std: float):
        p25 = np.percentile(array, 25)
        p75 = np.percentile(array, 75)
        zero_idxs = np.where(array == 0.0)[0]
        indices_of_outliers = []
        for ind, value in enumerate(array):
            if self.is_outlier(value, p25, p75):
                indices_of_outliers.append(ind)
        array = (array - mean) / std
        array[indices_of_outliers] = 0.0
        array[zero_idxs] = 0.0
        # replace by mean f0.
        array[indices_of_outliers] = np.max(array)

        return array


    def __getitem__(self, idx: int) -> FastSpeech2VoicePrintSample:

        info = self._dataset[idx]
        phoneme_collection = open(info.phonemes_path).read().split(" ")
        phoneme_ids = [
            self._phoneme_to_id[phoneme] for phoneme in phoneme_collection
        ]

        duration: np.array = np.around(np.load(info.duration_path))

        mels: torch.Tensor = torch.Tensor(np.load(info.mel_path))
        mels = (mels - self.mels_mean) / self.mels_std
        

        energy = self.normalize(np.load(info.energy_path), self.energy_mean, self.energy_std)        

        pitch = np.load(info.pitch_path)
        nonzero_idxs = np.where(pitch != 0)[0]
        pitch[nonzero_idxs] = np.log(pitch[nonzero_idxs])

        speaker_embs: np.ndarray = np.load(str(info.speaker_path))
        speaker_embs_tensor = torch.from_numpy(speaker_embs)



        return FastSpeech2VoicePrintSample(
            phonemes=phoneme_ids,
            num_phonemes=len(phoneme_ids),
            speaker_id=info.speaker_id,
            mel=mels,
            duration=duration,
            energy=energy,
            pitch=pitch,
            speaker_id_str=info.pitch_path.parent.name,
            wav_id=info.pitch_path.stem,
            speaker_emb=speaker_embs_tensor,
        )



class FastSpeech2VoicePrintFactory:

    """Create VCTK Dataset

    Note:
        * All the speeches from speaker ``p315`` will be skipped due to the lack of the corresponding text files.
        * All the speeches from ``p280`` will be skipped for ``mic_id="mic2"`` due to the lack of the audio files.
        * Some of the speeches from speaker ``p362`` will be skipped due to the lack of  the audio files.
        * See Also: https://datashare.is.ed.ac.uk/handle/10283/3443
        * Make sure to put the files as the following structure:
            text
            ├── p225
            |   ├──p225_001.TextGrid
            |   ├──p225_002.TextGrid
            |   └──...
            └── pXXX
                ├──pXXX_YYY.TextGrid
                └──...
            mels
            ├── p225
            |   ├──p225_001.pkl
            |   ├──p225_002.pkl
            |   └──...
            └── pXXX
                ├──pXXX_YYY.pkl
                └──...
    """

    def __init__(
        self,
        sample_rate: int,
        hop_size: int,
        n_mels: int,
        config: DatasetParams,
        phonemes_to_id: Dict[str, int],
        speakers_to_id: Dict[str, int],
        ignore_speakers: List[str],
        finetune: bool,
    ):
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.n_mels = n_mels
        self.finetune = finetune
        self._mels_dir = Path(config.mels_dir)
        self._pitch_dir = Path(config.pitch_dir)
        self._phones_dir = Path(config.phones_dir)
        self._energy_dir = Path(config.energy_dir)
        self._duration_dir = Path(config.duration_dir)
        self._speaker_emb_dir = Path(config.speaker_emb_dir)
        self._fastspeech2_ext = config.fastspeech2_ext
        self._phones_ext = config.phones_ext
        self.phoneme_to_id: Dict[str, int] = phonemes_to_id
        self.phoneme_to_id[PAD_TOKEN] = 0
        self.phoneme_to_id[""] = 1
        self.speaker_to_id: Dict[str, int] = speakers_to_id
        self.ignore_speakers = ignore_speakers
        if finetune:
            self.speaker_to_use = config.finetune_speakers
        else:
            self.speaker_to_use = [
                speaker.name
                for speaker in self._mels_dir.iterdir()
                if speaker not in config.finetune_speakers
            ]
        self._dataset: List[FastSpeech2VoicePrintInfo] = self._build_dataset()
        self.mels_mean, self.mels_std = self._get_mean_and_std_mels()
        self.energy_mean, self.energy_std = self._get_mean_and_std_scalar(self._energy_dir, self._fastspeech2_ext)
        self.energy_min, self.energy_max = self._get_min_max(self._energy_dir, self._fastspeech2_ext, self.energy_mean, self.energy_std)
        
        #self.energy_min = (self.energy_min - self.energy_mean) / self.energy_std
        #self.energy_max = (self.energy_max - self.energy_mean) / self.energy_std
        
        self.pitch_mean, self.pitch_std = self._get_mean_and_std_scalar(self._pitch_dir, self._fastspeech2_ext)
        self.pitch_min, self.pitch_max = self._get_min_max(self._pitch_dir, self._fastspeech2_ext, self.pitch_mean, self.pitch_std)
        
        #self.pitch_min = (self.pitch_min - self.pitch_mean) / self.pitch_std
        #self.pitch_max = (self.pitch_max - self.pitch_mean) / self.pitch_std

    @staticmethod
    def add_to_mapping(mapping: Dict[str, int], token: str) -> None:
        if token not in mapping:
            mapping[token] = len(mapping)

    def split_train_valid(
        self, test_fraction: float
    ) -> Tuple[FastSpeech2VoicePrintDataset, FastSpeech2VoicePrintDataset]:
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
        train_dataset = FastSpeech2VoicePrintDataset(
            sample_rate=self.sample_rate,
            hop_size=self.hop_size,
            mels_mean=self.mels_mean,
            mels_std=self.mels_std,
            energy_mean=self.energy_mean,
            energy_std=self.energy_std,
            pitch_mean=self.pitch_mean,
            pitch_std=self.pitch_std,
            energy_min=self.energy_min, 
            energy_max=self.energy_max,
            pitch_min=self.pitch_min, 
            pitch_max=self.pitch_max,
            phoneme_to_ids=self.phoneme_to_id,
            data=train_data,
        )
        test_dataset = FastSpeech2VoicePrintDataset(
            sample_rate=self.sample_rate,
            hop_size=self.hop_size,
            mels_mean=self.mels_mean,
            mels_std=self.mels_std,
            energy_mean=self.energy_mean,
            energy_std=self.energy_std,
            pitch_mean=self.pitch_mean,
            pitch_std=self.pitch_std,
            energy_min=self.energy_min, 
            energy_max=self.energy_max,
            pitch_min=self.pitch_min, 
            pitch_max=self.pitch_max,
            phoneme_to_ids=self.phoneme_to_id,
            data=test_data,
        )
        return train_dataset, test_dataset

    def _build_dataset(self) -> List[FastSpeech2VoicePrintInfo]:

        dataset: List[FastSpeech2VoicePrintInfo] = []
        mels_set = {
            Path(x.parent.name) / x.stem
            for x in self._mels_dir.rglob(f"*{self._fastspeech2_ext}")
        }
        enegry_set = {
            Path(x.parent.name) / x.stem
            for x in self._energy_dir.rglob(f"*{self._fastspeech2_ext}")
        }
        pitch_set = {
            Path(x.parent.name) / x.stem
            for x in self._pitch_dir.rglob(f"*{self._fastspeech2_ext}")
        }
        duration_set = {
            Path(x.parent.name) / x.stem
            for x in self._duration_dir.rglob(f"*{self._fastspeech2_ext}")
        }
        phones_set = {
            Path(x.parent.name) / x.stem
            for x in self._phones_dir.rglob(f"*{self._phones_ext}")
        }
        speaker_emb_set = {
            Path(x.parent.name) / x.stem
            for x in self._speaker_emb_dir.rglob(f"*{self._fastspeech2_ext}")
        }       
        samples = list(mels_set & duration_set & pitch_set & enegry_set & phones_set & speaker_emb_set)
        for sample in tqdm(samples):
            if sample.parent.name in REMOVE_SPEAKERS:
                continue

            mels_path = (self._mels_dir / sample).with_suffix(self._fastspeech2_ext)
            energy_path = (self._energy_dir / sample).with_suffix(self._fastspeech2_ext)
            pitch_path = (self._pitch_dir / sample).with_suffix(self._fastspeech2_ext)
            duration_path = (self._duration_dir / sample).with_suffix(self._fastspeech2_ext)
            phonemes_path = (self._phones_dir / sample).with_suffix(self._phones_ext)
            speaker_emb_path = (self._speaker_emb_dir / sample).with_suffix(self._fastspeech2_ext)
            self.add_to_mapping(self.speaker_to_id, sample.parent.name)
            
            speaker_id = self.speaker_to_id[sample.parent.name]
            phonemes = open(phonemes_path).read().split(" ")
            if len(phonemes) > 0:

                for phoneme in phonemes:
                    self.add_to_mapping(self.phoneme_to_id, phoneme)

                if sample.parent.name in self.speaker_to_use:
                    dataset.append(
                        FastSpeech2VoicePrintInfo(
                            phonemes_length=len(phonemes),
                            phonemes_path=phonemes_path,
                            energy_path=energy_path,
                            duration_path=duration_path,
                            pitch_path=pitch_path,
                            mel_path=mels_path,
                            speaker_id=speaker_id,
                            speaker_path=speaker_emb_path,
                        )
                    )

        return dataset

    def _get_mean_and_std_mels(self) -> Tuple[torch.Tensor, torch.Tensor]:
        mel_sum = torch.zeros(self.n_mels, dtype=torch.float64)
        mel_squared_sum = torch.zeros(self.n_mels, dtype=torch.float64)
        counts = 0

        for mel_path in self._mels_dir.rglob(f"*{self._fastspeech2_ext}"):
            if mel_path.parent.name in REMOVE_SPEAKERS:
                continue
            mels: torch.Tensor = torch.Tensor(np.load(mel_path))
            mel_sum += mels.sum(dim=-1)
            mel_squared_sum += (mels ** 2).sum(dim=-1)
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





class FastSpeech2VoicePrintCollate:
    """
    Zero-pads model inputs and targets based on number of frames per setep
    """

    def __init__(self, n_frames_per_step: int = 1):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch: List[FastSpeech2VoicePrintSample]) -> FastSpeech2VoicePrintBatch:
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
            durations = batch[idx].duration
            durations_padded[i, : len(durations)] = torch.FloatTensor(durations)
            energy = batch[idx].energy
            energy_padded[i, : len(energy)] = torch.FloatTensor(energy)
            pitch = batch[idx].pitch
            pitch_padded[i, : len(pitch)] = torch.FloatTensor(pitch)
            speaker_emb_tensor[i] = batch[idx].speaker_emb

        num_mels = batch[0].mel.size(0)
        max_target_len = max([x.mel.size(1) for x in batch])
        mels_lens = torch.LongTensor([x.mel.size(1) for x in batch])


        # include mel padded and gate padded
        mel_padded = torch.zeros(
            (batch_size, num_mels, max_target_len), dtype=torch.float
        )
        for i, idx in enumerate(ids_sorted_decreasing):
            mel: torch.Tensor = batch[idx].mel
            mel_padded[i, :, : mel.shape[1]] = mel
        mel_padded = mel_padded.permute(0, 2, 1)

        return FastSpeech2VoicePrintBatch(
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

