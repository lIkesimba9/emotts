from dataclasses import dataclass, field
from typing import List, Optional, Literal


@dataclass
class DatasetParams:

    wav_dir: str

    mels_dir: str
    duration_dir: str
    text_dir: str

    feature_dir: str
    ignore_speakers: List[str]

    dataset_name: str
    
    sentence_emb_dir: Optional[str] = field(default=None)
    speaker_emb_dir: Optional[str] = field(default=None)
    pitch_dir: Optional[str] = field(default=None)
    energy_dir: Optional[str] = field(default=None)
    pitch_norm: bool = field(default=True)
    energy_norm: bool = field(default=True)
    
    finetune_speakers: List[str] = field(
        default_factory=lambda: [f"00{i}" for i in range(11, 21)]
    )
    context_lenght: int = field(default=0)
    text_ext: str = field(default=".TextGrid")
    data_ext: str = field(default=".npy")
    speaker_emb_dir: Optional[str] = field(default=None)

        
