import argparse
import torch

from pathlib import Path

from src.data_process.speech_factory import SpeechFactory
from src.constants import DATASET_INFO_PATH, TEST_FILENAME, TRAIN_FILENAME, MELS_MEAN_FILENAME, MELS_STD_FILENAME, ENERGY_MIN_FILENAME, \
    ENERGY_MAX_FILENAME , PITCH_MIN_FILENAME, PITCH_MAX_FILENAME, PHONEMES_FILENAME, SPEAKERS_FILENAME, STATISTIC_FILENAME
from src.train_config import load_config
from src.data_process.utils import dump_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="configuration file path"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    factory = SpeechFactory(
        sample_rate=config.sample_rate,
        hop_size=config.hop_size,
        n_mels=config.n_mels,
        config=config.data,
        finetune=config.finetune,
    )
    data_info_dir = Path(DATASET_INFO_PATH) / Path(config.data.dataset_name)
    data_info_dir.mkdir(parents=True, exist_ok=True)

    factory.create_train_test_json_files(config.test_size, data_info_dir / TRAIN_FILENAME, data_info_dir / TEST_FILENAME)

    dump_json(data_info_dir / SPEAKERS_FILENAME, factory.speaker_to_id)
    dump_json(data_info_dir / PHONEMES_FILENAME, factory.phoneme_to_id)

    dump_json(data_info_dir / STATISTIC_FILENAME, factory.statistic_dict)

    torch.save(factory.mels_mean, data_info_dir / MELS_MEAN_FILENAME)
    torch.save(factory.mels_std, data_info_dir / MELS_STD_FILENAME)

    # std and mean after for log enegry and pitch
    torch.save(factory.energy_min, data_info_dir / ENERGY_MIN_FILENAME)
    torch.save(factory.energy_max, data_info_dir / ENERGY_MAX_FILENAME)

    torch.save(factory.pitch_min, data_info_dir / PITCH_MIN_FILENAME)
    torch.save(factory.pitch_max, data_info_dir / PITCH_MAX_FILENAME)

    

if __name__ == "__main__":
    main()
