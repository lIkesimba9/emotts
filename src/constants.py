from dataclasses import dataclass
from pathlib import Path
from typing import Union

from src.models.hifi_gan.hifi_config import HIFIParams


PATHLIKE = Union[str, Path]
FEATURE_MODEL_FILENAME = "feature_model.pth"
MELS_MEAN_FILENAME = "mels_mean.pth"
MELS_STD_FILENAME = "mels_std.pth"
PHONEMES_FILENAME = "phonemes.json"
SPEAKERS_FILENAME = "speakers.json"
CHECKPOINT_DIR = Path("checkpoints")
DATA_DIR = Path("checkpoints")
LOG_DIR = Path("logs")
MODEL_DIR = Path("models")


@dataclass
class TacoTronCheckpoint:
    path: Path
    model_file_name: str = FEATURE_MODEL_FILENAME
    phonemes_file_name: str = PHONEMES_FILENAME
    speakers_file_name: str = SPEAKERS_FILENAME
    mels_mean_filename: str = MELS_MEAN_FILENAME
    mels_std_filename: str = MELS_STD_FILENAME


@dataclass
class Language:
    name: str
    g2p_model_path: Path
    tacotron_checkpoint: TacoTronCheckpoint
    hifi_params: HIFIParams


@dataclass
class SupportedLanguages:
    english: Language = Language(
        name="English (en-EN)",
        g2p_model_path=Path("models/en/g2p/english_g2p.zip"),
        tacotron_checkpoint=TacoTronCheckpoint(path=Path("models/en/tacotron")),
        hifi_params=HIFIParams(dir_path="en/hifi", config_name="config.json", model_name="generator.hifi"),
    )
    russian: Language = Language(
        name="Russian (ru-RU)",
        g2p_model_path=Path("models/ru/g2p/english_g2p.zip"),
        tacotron_checkpoint=TacoTronCheckpoint(path=Path("models/ru/tacotron")),
        hifi_params=HIFIParams(dir_path="ru/hifi", config_name="config.json", model_name="generator.hifi"),
    )


@dataclass
class Emotion:
    name: str
    reference_audio_path: PATHLIKE
    ru_speaker_id: int
    en_speaker_id: int


@dataclass
class SupportedEmotions:
    angry: Emotion = Emotion(name="angry", reference_audio_path="", ru_speaker_id=10, en_speaker_id=0)
    happy: Emotion = Emotion(name="happy", reference_audio_path="", ru_speaker_id=21, en_speaker_id=0)
    sad: Emotion = Emotion(name="sad", reference_audio_path="", ru_speaker_id=40, en_speaker_id=0)
    very_angry: Emotion = Emotion(name="very_angry", reference_audio_path="", ru_speaker_id=41, en_speaker_id=0)
    very_happy: Emotion = Emotion(name="happy", reference_audio_path="", ru_speaker_id=12, en_speaker_id=0)
