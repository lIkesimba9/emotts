import json
from dataclasses import dataclass
from pathlib import Path
from typing import Union


PATHLIKE = Union[str, Path]
FEATURE_MODEL_FILENAME = "feature_model.pth"
FASTSPEECH2_MODEL_FILENAME = "fastspeech2_model.pth"
MELS_MEAN_FILENAME = "mels_mean.pth"
MELS_STD_FILENAME = "mels_std.pth"
ENERGY_MIN_FILENAME = "energy_min.pth"
ENERGY_MAX_FILENAME = "energy_max.pth"
PITCH_MIN_FILENAME = "pitch_min.pth"
PITCH_MAX_FILENAME = "pitch_max.pth"
PHONEMES_FILENAME = "phonemes.json"
SPEAKERS_FILENAME = "speakers.json"
DATASET_INFO_PATH = "data_info"
STATISTIC_FILENAME = "statistic.json"
CHECKPOINT_DIR = Path("checkpoints")
HIFI_CHECKPOINT_NAME = "hifi"
FEATURE_CHECKPOINT_NAME = "feature"
FASTSPEECH2_CHECKPOINT_NAME = "fastspeech2"
DATA_DIR = Path("checkpoints")
LOG_DIR = Path("logs")
MODEL_DIR = Path("models")
REFERENCE_PATH = Path("references")
SPEAKER_PRINT_DIR = Path("speakers_prints")
DATASET_INFO_PATH = Path("data_info")
TRAIN_FILENAME = "train.json"
TEST_FILENAME = "test.json"


PHONEMES_ENG = [
    ' D UW1 Y UW1 R IY1 AH0 L AY2 Z W AH1 T T AY1 M IH1 T IH1 Z  ',
    ' HH IY1 K AH1 M Z B AE1 K T UW1 DH AH0 V AE1 L IY0  ',
    ' DH IH1 S D R EH1 S D AH1 Z N AA1 T L UH1 K W ER1 TH M AH1 CH  ',
    ' W AH1 T HH AE1 P AH0 N D T AH0 N AY1 T HH AE1 Z N AH1 TH IH0 NG T UW1 D UW1 W IH1 DH HH EH1 N R IY0  ',
    ' T AH0 D EY1  F AY1 V Y IH1 R Z L EY1 T ER0  W IY1 AA1 R F EY1 S IH0 NG AH0 S IH1 M AH0 L ER0 S IH2 CH UW0 EY1 SH AH0 N  ',
    ' W EH1 N AY1 S AO1 Y UW1 K IH1 S IH0 NG  Y UW1 L UH1 K T R IH1 L IY0 HH AE1 P IY0  ',
    ' OW1 N L IY0 W AH1 N V IY1 HH IH0 K AH0 L M EY1 B IY1 AH0 L AW1 D T UW1 P AA1 R K AE1 T EH1 N IY0 G IH1 V AH0 N T AY1 M  ',
    ' DH AH0 D EH1 D L AY2 N Z AA1 R IH2 N D IY1 D V EH1 R IY0 T AY1 T  ',
    ' AY1 EH1 M G L AE1 D Y UW1 EH2 N JH OY1 D Y ER0 S EH1 L F  ',
    ' W AH1 T AA1 R Y UW1 S T IH1 L D UW1 IH0 NG HH IY1 R  ',
    ' DH IH1 S IH1 Z AE1 N AE1 N AH0 M AH0 L DH AE1 T IH1 Z AH0 D M AY1 ER0 D F AO1 R IH1 T S W AY1 T N AH0 S AH0 N D K L EH1 N L IY0 N IH0 S  ',
    ' P ER0 HH AE1 P S DH EH1 R IH1 Z AH0 N AH1 DH ER0 W EY1 T UW1 P OW1 Z DH IY1 Z IH1 SH UW0 Z  ',
    ' Y AO1 R S T UW1 D AH0 N T S T EH1 S T S K AO1 R Z D R AA1 P L OW1 ER0 AH0 N D L OW1 ER0 EH1 V ER0 IY0 Y IH1 R  ',
    ' W EH0 R EH1 V ER0 HH ER1 T IH1 R Z F EH1 L  AH0 F R UW1 T T R IY1 G R UW1  ',
    ' AY1 W AA1 Z AH0 B AW1 T T UW1 HH EH1 D B AE1 K T UW1 M AY1 HH OW0 T EH1 L AH0 N D G OW1 T UW1 S L IY1 P  ',
    ' Y UW1 S EH1 D SH IY1 R IH1 L IY0 HH EH1 L P T L AE1 S T T AY1 M  ',
    ' M AY1 F EY1 V ER0 IH0 T S IY1 Z AH0 N  S P R IH1 NG  IH1 Z HH IY1 R  ',
    ' HH IY1  Z DH AH0 R IH1 CH G AY1 HH UW1 B IH1 L T DH IY0 EH1 R P L EY0 N Z  ',
    ' AA1 T OW2 AH0 N D IH0 L IH1 Z AH0 B AH0 TH G EY1 V IH1 T T UW1 AH1 S  F AO1 R DH AH0 W EH1 D IH0 NG  IH2 N K R EH1 D AH0 B L IY0 JH EH1 N ER0 AH0 S  ',
    ' L UH1 K  DH AH0 P AH0 L IY1 S S EH1 D DH AE1 T DH EH1 R W AA1 Z N AH1 TH IH0 NG S T OW1 L AH0 N F R AH1 M DH AH0 HH AW1 S  ',
    ' AH0 N D AY1 S AH0 P OW1 Z W IY1 K AE1 N TH AE1 NG K Y AO1 R B R AH1 DH ER0 F AO1 R DH AE1 T  ',
    ' DH AE1 T S EY0 P R IH1 T IY0 D EY1 N JH ER0 AH0 S TH IH1 NG Y UW1  R D UW1 IH0 NG  ',
    ' HH IY1 ER0 AY1 V D IH0 N JH AH0 P AE1 N F AO1 R DH AH0 F ER1 S T T AY1 M AE1 T DH IY0 EY1 JH AH1 V T W EH1 N T IY0 S IH1 K S  ',
    ' S AE1 M TH AO1 T W IY1 W ER1 HH AE1 V IH0 NG F AH1 N B IY1 IH0 NG T AH0 G EH1 DH ER0  ',
    ' W EH1 L  DH AH0 T R UW1 V AE1 L Y UW0 AH1 V S AH1 M TH IH0 NG IH1 S N T AO1 L W EY2 Z D IH0 T ER1 M AH0 N D B AY1 IH1 T S P R AY1 S  ',
    ' N OW1  IH1 T  S N AA1 T P AH0 L AY1 T T UW1 D IH0 S K AH1 S AH0 L EY1 D IY0 S EY1 JH  ',
    ' JH AH1 S T AH0 N AH1 DH ER0 K W AO1 R T ER0 M AY1 L AH0 N D AY1 D AA1 N T HH AE1 V T UW1 B IY1 T AA1 L ER0 AH0 N T EH1 V ER0 AH0 G EH1 N  ',
    ' B AH1 T JH OW1 N Z AH0 P AA1 R T M AH0 N T HH AE1 D OW1 N L IY0 B IH1 N R EH1 N T IH0 D AW1 T F AO1 R AH0 W IY1 K  ',
    ' W AH1 T Y AO1 R P ER1 F IH1 K T D EY1 W UH1 D HH AE1 V B IH1 N L AY1 K  ',
    ' N AA1 T AH0 V EH1 R IY0 Y UW1 S F AH0 L S K IH1 L  AH0 S P EH1 SH L IY0 W EH1 N DH AH0 M AH1 N IY0 R AH1 N Z AW1 T  ',
]

PHONEMES_CHI = [
    ['q', 'i1', 'ng', 'ch', 'u1', 'v2', 'l', 'a2', 'n', 'e2', 'r', 'sh', 'e4', 'ng', 'v2', 'l', 'a2', 'n'],
    ['t', 'ia1', 'n', 'd', 'ao4', 'ch', 'ou2', 'q', 'i2', 'n'],
    ['j', 'iou3', 't', 'ia1', 'n', 'l', 'a3', 'n', 've4'],
    ['s', 'ai1', 'ue1', 'n', 'sh', 'ii1', 'm', 'a3', 'ia1', 'n', 'zh', 'ii1', 'f', 'ei1', 'f', 'u2'],
    ['i1', 'm', 'i2', 'ng', 'j', 'i1', 'ng', 'r', 'e2', 'n'],
    ['i1', 's', 'ii1', 'b', 'u4', 'g', 'ou3'],
    ['i1', 'j', 'ia4', 'n', 'sh', 'ua1', 'ng', 'd', 'iao1'],
    ['sh', 'a1', 'n', 'v3', 'v4', 'l', 'ai2', 'f', 'e1', 'ng', 'm', 'a3', 'n', 'l', 'ou2'],
    ['m', 'a2', 'q', 've4', 's', 'uei1', 'x', 'iao3', 'u3', 'z', 'a4', 'ng', 'j', 'v4', 'q', 'va2', 'n'],
    ['q', 'ia2', 'ng', 'l', 'o2', 'ng', 'n', 'a2', 'n', 'ia1', 'd', 'i4', 't', 'ou2', 'sh', 'e2'],
    ['q', 'ia2', 'n', 'p', 'a4', 'l', 'a2', 'ng', 'h', 'ou4', 'p', 'a4', 'h', 'u3'],
    ['d', 'a4', 'zh', 'ii4', 'r', 'uo4', 'v2']
]
