from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DatasetParams:

    wav_dir: str
    feature_dir: str
    mels_dir: str
    duration_dir: str
    pitch_dir: str
    energy_dir: str
    ignore_speakers: List[str]
    text_ext: str = field(default=".TextGrid")
    mels_ext: str = field(default=".npy")
    fastspeech2_ext: str = field(default=".npy")
    phones_ext: str = field(default=".txt")
    finetune_speakers: List[str] = field(
        default_factory=lambda: [f"00{i}" for i in range(11, 21)]
    )
    speaker_emb_dir: Optional[str] = field(default=None)
    text_dir: Optional[str] = field(default=None)
