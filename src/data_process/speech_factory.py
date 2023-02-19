import random
import numpy as np
import tgt
import torch


from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from operator import itemgetter

from src.data_process.config import DatasetParams
from .utils import dump_json, NUMBER, PHONES_TIER, PAD_TOKEN, NO_CONTEXT





class SpeechFactory:


    def __init__(
        self,
        sample_rate: int,
        hop_size: int,
        n_mels: int,
        config: DatasetParams,
        finetune: bool,
    ):
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.n_mels = n_mels
        self.finetune = finetune
        self._mels_dir = Path(config.mels_dir)
        self._duration_dir = Path(config.duration_dir)
        self._text_ext = config.text_ext
        self._text_dir = Path(config.text_dir)
        self._pitch_dir = Path(config.pitch_dir)
        self._energy_dir = Path(config.energy_dir)
        if config.sentence_emb_dir != None:
            self._sentence_emb_dir = Path(config.sentence_emb_dir)
        else:
            self._sentence_emb_dir = None
        self._speaker_emb_dir = Path(config.speaker_emb_dir)
        self._data_ext = config.data_ext
        self.phoneme_to_id: Dict[str, int] = {}
        self.speaker_to_id: Dict[str, int] = {}
        self.phoneme_to_id[PAD_TOKEN] = 0
        self.ignore_speakers = config.ignore_speakers
        self.context_lenght = config.context_lenght
        self.finetune_speakers  = config.finetune_speakers

        if finetune:
            self.speaker_to_use = [
                speaker.name
                for speaker in self._mels_dir.iterdir()
                if speaker.name in self.finetune_speakers and speaker.name not in self.ignore_speakers
            ]
        else:
            self.speaker_to_use = [
                speaker.name
                for speaker in self._mels_dir.iterdir()
                if speaker.name not in self.finetune_speakers and speaker.name not in self.ignore_speakers
            ]
        self.samples_collection = self.find_all_correct_samples()
        if self.context_lenght > 0:
            self.structure_of_book = self._build_structure_of_books()


        self._dataset: List = self._build_info_json()
        print("Calculate statistics...")
        self.mels_mean, self.mels_std = self._get_mean_and_std_mels()

        self.statistic_dict = self._calculate_mean_std_by_phonemes(list(self.phoneme_to_id.keys()))

        self.energy_min, self.energy_max, self.pitch_min, self.pitch_max = self._get_min_max_energy_pitch()
       

    def _calculate_mean_std_by_phonemes(self, list_of_all_phonemes: List[str]):
        list_of_all_phonemes.remove(PAD_TOKEN)
        energy_standart_scaller = {phoneme: StandardScaler() for phoneme in list_of_all_phonemes}
        pitch_standart_scaller =  {phoneme: StandardScaler() for phoneme in list_of_all_phonemes}
        for sample in tqdm(self.samples_collection):
            if sample.parent.name in self.ignore_speakers:
                    continue

            tg_path = (self._text_dir / sample).with_suffix(self._text_ext)

            text_grid = tgt.read_textgrid(tg_path)

            energy_path = (self._energy_dir / sample).with_suffix(self._data_ext)
            pitch_path = (self._pitch_dir / sample).with_suffix(self._data_ext)


            
            if PHONES_TIER in text_grid.get_tier_names():
                phones_tier = text_grid.get_tier_by_name(PHONES_TIER)
                phonemes = [x.text for x in phones_tier.get_copy_with_gaps_filled()]
                if "spn" in phonemes:
                    continue
                energy = np.load(energy_path)
                nonzero_idxs = np.where(energy != 0)[0]
                energy[nonzero_idxs] = np.log(energy[nonzero_idxs])
                
                pitch = np.load(pitch_path)
                nonzero_idxs = np.where(pitch != 0)[0]
                pitch[nonzero_idxs] = np.log(pitch[nonzero_idxs])
    
                for phoneme, energy_value, pitch_value in zip(phonemes, energy, pitch):
                    energy_standart_scaller[phoneme].partial_fit(np.array([energy_value]).reshape((-1, 1)))
                    pitch_standart_scaller[phoneme].partial_fit(np.array([pitch_value]).reshape((-1, 1)))
                    
        statistic_dict = {}
        for phoneme in list_of_all_phonemes:
            statistic_dict[phoneme] = {
                "pitch_mean": str(pitch_standart_scaller[phoneme].mean_[0]),
                "pitch_std": str(pitch_standart_scaller[phoneme].scale_[0]),
                "energy_mean": str(energy_standart_scaller[phoneme].mean_[0]),
                "energy_std": str(energy_standart_scaller[phoneme].scale_[0]),
            }
        return statistic_dict

    def find_all_correct_samples(self):
        
        mels_set = {
            Path(x.parent.name) / x.stem
            for x in self._mels_dir.rglob(f"*{self._data_ext}")
        }
        speaker_emb_set = {
            Path(x.parent.name) / x.stem
            for x in self._speaker_emb_dir.rglob(f"*{self._data_ext}")
        }
        duration_set = {
            Path(x.parent.name) / x.stem
            for x in self._duration_dir.rglob(f"*{self._data_ext}")
        }
        texts_set = {
            Path(x.parent.name) / x.stem
            for x in self._text_dir.rglob(f"*{self._text_ext}")
        }

        enegry_set = {
            Path(x.parent.name) / x.stem
            for x in self._energy_dir.rglob(f"*{self._data_ext}")
        }
        pitch_set = {
            Path(x.parent.name) / x.stem
            for x in self._pitch_dir.rglob(f"*{self._data_ext}")
        }
        if self._sentence_emb_dir != None:
            sentence_emb_set = {
                Path(x.parent.name) / x.stem
                for x in self._sentence_emb_dir.rglob(f"*{self._data_ext}")
            }
        samples_collection = mels_set & duration_set & speaker_emb_set & texts_set & enegry_set & pitch_set
        if self._sentence_emb_dir != None:
            samples_collection &= sentence_emb_set

        return list(samples_collection)

    def _build_structure_of_books(self):
        structure_of_book = defaultdict(dict)
        for sample in tqdm(self.samples_collection):
            if str(sample.parent.name) in self.finetune_speakers:
                subwords_array = sample.name.split('_')
                book_name = '_'.join(subwords_array[: -2])
                charter_num = int(subwords_array[-2])
                sentence_num = int(subwords_array[-1])
                if structure_of_book[book_name].get(charter_num) == None:
                    structure_of_book[book_name][charter_num] = [sentence_num]
                else:
                    structure_of_book[book_name][charter_num].append(sentence_num)
                    structure_of_book[book_name][charter_num].sort()
        
        return structure_of_book


    @staticmethod
    def add_to_mapping(mapping: Dict[str, int], token: str) -> None:
        if token not in mapping:
            mapping[token] = len(mapping)


    def create_train_test_json_files(
        self, test_fraction: float, train_path_file: Path, test_path_file: Path 
    ) -> None:
        speakers_to_data_id: Dict[int, List[int]] = defaultdict(list)

        for i, sample in enumerate(self._dataset):
            speakers_to_data_id[sample["speaker_id"]].append(i)
        test_ids: List[int] = []
        for speaker, ids in speakers_to_data_id.items():
            test_size = int(len(ids) * test_fraction)
            if test_size > 0 and speaker:
                test_indexes = random.choices(ids, k=test_size)
                test_ids.extend(test_indexes)

        train_data = []
        test_data = []
        for i in range(len(self._dataset)):
            if i in test_ids:
                test_data.append(self._dataset[i])
            else:
                train_data.append(self._dataset[i])
        dump_json(train_path_file, train_data)
        dump_json(test_path_file, test_data)
    

    @staticmethod
    def left_pad(context: List[str], context_lenght) -> None:
        while (len(context) != context_lenght):
            context.insert(0, NO_CONTEXT)


    @staticmethod
    def right_pad(context: List[str], context_lenght) -> None:
        while (len(context) != context_lenght):
            context.append(NO_CONTEXT)


    def get_path_of_context_sentence_and_mels(self, context: List[str], sample_name: str, path_to_sentence_emb: Path, path_to_mels: Path):
        path_context_of_sentence: List[str] = []
        path_context_of_mels: List[str] = []
        for idx in context:
            part_of_name = sample_name.split("_")
            part_of_name[-1] = str(idx).zfill(6)
            
            path_context_of_sentence.append(str((path_to_sentence_emb / '_'.join(part_of_name)).with_suffix(self._data_ext)))
            path_context_of_mels.append(str((path_to_mels / '_'.join(part_of_name)).with_suffix(self._data_ext)))
        return path_context_of_sentence, path_context_of_mels

    def add_context(self, row: Dict, sample_name: str, path_to_sentence_emb: Path, path_to_mels: Path):
        subwords_array = sample_name.split('_')
        book_name = '_'.join(subwords_array[: -2])
        charter_num = int(subwords_array[-2])
        sentence_num = int(subwords_array[-1])
        index = self.structure_of_book[book_name][charter_num].index(sentence_num)
        count_sentence_of_charter = len(self.structure_of_book[book_name][charter_num])


        left_context = self.structure_of_book[book_name][charter_num][max(index - self.context_lenght, 0) :index]
        left_context_of_sentence, left_context_of_mels = self.get_path_of_context_sentence_and_mels(left_context, sample_name, path_to_sentence_emb, path_to_mels)
        
        
        right_context = self.structure_of_book[book_name][charter_num][index + 1: min(index + 1 + self.context_lenght, count_sentence_of_charter)]
        right_context_of_sentence, right_context_of_mels = self.get_path_of_context_sentence_and_mels(right_context, sample_name, path_to_sentence_emb, path_to_mels)
        
        self.left_pad(left_context_of_sentence, self.context_lenght)
        self.left_pad(left_context_of_mels, self.context_lenght)

        self.right_pad(right_context_of_sentence, self.context_lenght)
        self.right_pad(right_context_of_mels, self.context_lenght)
        
        row["left_context_of_sentence"] = left_context_of_sentence
        row["left_context_of_mels"] = left_context_of_mels

        row["right_context_of_mels"] = right_context_of_mels
        row["right_context_of_sentence"] = right_context_of_sentence


    def _build_info_json(self) -> List:

        dataset_info: List = []

        for sample in tqdm(self.samples_collection):

            if sample.parent.name in self.ignore_speakers:
                continue

            duration_path = (self._duration_dir / sample).with_suffix(self._data_ext)
            tg_path = (self._text_dir / sample).with_suffix(self._text_ext)
            
            mel_path = (self._mels_dir / sample).with_suffix(self._data_ext)
            speaker_emb_path = (self._speaker_emb_dir / sample).with_suffix(self._data_ext)

            text_grid = tgt.read_textgrid(tg_path)

            energy_path = (self._energy_dir / sample).with_suffix(self._data_ext)
            pitch_path = (self._pitch_dir / sample).with_suffix(self._data_ext)
            if self._sentence_emb_dir != None:
                sentence_path = (self._sentence_emb_dir / sample).with_suffix(self._data_ext)

            self.add_to_mapping(self.speaker_to_id, sample.parent.name)
            speaker_id = self.speaker_to_id[sample.parent.name]
            
            if PHONES_TIER in text_grid.get_tier_names():

                phones_tier = text_grid.get_tier_by_name(PHONES_TIER)
                phonemes = [x.text for x in phones_tier.get_copy_with_gaps_filled()]
                if "spn" in phonemes:
                    continue
                for phoneme in phonemes:
                    self.add_to_mapping(self.phoneme_to_id, phoneme)
                row: Dict = {
                    "text_path": str(tg_path),
                    "mel_path": str(mel_path),
                    "duration_path": str(duration_path),
                    "speaker_id": speaker_id, 
                    "speaker_path": str(speaker_emb_path),
                    "phonemes_length": len(phonemes),
                    "energy_path" : str(energy_path),
                    "pitch_path": str(pitch_path),
                  
                }
                if self._sentence_emb_dir != None:
                    row["sentence_path"] = str(sentence_path)
                    row["left_context_of_sentence"] = [],
                    row["left_context_of_mels"] = []
                    row["right_context_of_mels"] = []
                    row["right_context_of_sentence"] = []

                    if self.context_lenght > 0 and sample.parent.name in self.finetune_speakers:
                        self.add_context(row, sample.name, sentence_path.parent, mel_path.parent)
                    
                if sample.parent.name in self.speaker_to_use:
                    dataset_info.append(row)

        return dataset_info


    def _get_mean_and_std_mels(self) -> Tuple[torch.Tensor, torch.Tensor]:
        mel_sum = torch.zeros(self.n_mels, dtype=torch.float64)
        mel_squared_sum = torch.zeros(self.n_mels, dtype=torch.float64)
        counts = 0

        for mel_path in self._mels_dir.rglob(f"*{self._data_ext}"):
            if mel_path.parent.name in self.ignore_speakers:
                continue
            mels: torch.Tensor = torch.Tensor(np.load(mel_path)).squeeze(0)
            mel_sum += mels.sum(dim=-1)
            mel_squared_sum += (mels ** 2).sum(dim=-1)
            counts += mels.shape[-1]

        mels_mean: torch.Tensor = mel_sum / counts
        mels_std: torch.Tensor = torch.sqrt(
            (mel_squared_sum - mel_sum * mel_sum / counts) / counts
        )

        return mels_mean.view(-1, 1), mels_std.view(-1, 1)

    def _get_min_max_energy_pitch(self):
        energy_max_value = np.finfo(np.float64).min
        energy_min_value = np.finfo(np.float64).max
        pitch_max_value = np.finfo(np.float64).min
        pitch_min_value = np.finfo(np.float64).max

        for sample in tqdm(self.samples_collection):
            if sample.parent.name in self.ignore_speakers:
                    continue

            tg_path = (self._text_dir / sample).with_suffix(self._text_ext)

            text_grid = tgt.read_textgrid(tg_path)

            energy_path = (self._energy_dir / sample).with_suffix(self._data_ext)
            pitch_path = (self._pitch_dir / sample).with_suffix(self._data_ext)


            
            if PHONES_TIER in text_grid.get_tier_names():
                phones_tier = text_grid.get_tier_by_name(PHONES_TIER)
                phonemes = [x.text for x in phones_tier.get_copy_with_gaps_filled()]
                if "spn" in phonemes:
                    continue
                energy = np.load(energy_path)
                nonzero_idxs = np.where(energy != 0)[0]
                energy[nonzero_idxs] = np.log(energy[nonzero_idxs])
                
                pitch = np.load(pitch_path)
                nonzero_idxs = np.where(pitch != 0)[0]
                pitch[nonzero_idxs] = np.log(pitch[nonzero_idxs])

                energy_max_value = max(energy_max_value, max(energy))
                energy_min_value = min(energy_min_value, min(energy))

                
                pitch_max_value = max(pitch_max_value, max(pitch))
                pitch_min_value = min(pitch_min_value, min(pitch))
        return energy_min_value, energy_max_value, pitch_min_value, pitch_max_value
