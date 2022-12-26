#!/usr/bin/env python
from pathlib import Path

import click
import torch
import torchaudio
from librosa.filters import mel as librosa_mel
from tqdm import tqdm
from typing import List, Tuple

import tgt
import librosa
import numpy as np
import pyworld as pw
from scipy.interpolate import interp1d

import torch
import torchaudio
from librosa.filters import mel as librosa_mel
from tqdm import tqdm
import numpy as np
from tqdm import tqdm

import audio as Audio

# Using the same parameters as in HiFiGAN
F_MIN = 0
F_MAX = 8000
HOP_SIZE = 256
WIN_SIZE = 1024
N_FFT = 1024
N_MELS = 80
SAMPLE_RATE = 22050
FILTER_LENGTH = 1024





def _convert_to_continuous_f0(f0: np.array) -> np.array:
    if (f0 == 0).all():
        return f0

    # padding start and end of f0 sequence
    start_f0 = f0[f0 != 0][0]
    end_f0 = f0[f0 != 0][-1]
    start_idx = np.where(f0 == start_f0)[0][0]
    end_idx = np.where(f0 == end_f0)[0][-1]
    f0[:start_idx] = start_f0
    f0[end_idx:] = end_f0

    # get non-zero frame index
    nonzero_idxs = np.where(f0 != 0)[0]

    # perform linear interpolation
    interp_fn = interp1d(nonzero_idxs, f0[nonzero_idxs])
    f0 = interp_fn(np.arange(0, f0.shape[0]))
    
    return f0


@click.command()
@click.option("--input-audio-dir", type=Path, required=True,
              help="Directory with audios to process.") #data/processed/esd/english/resampled
@click.option("--input-textgrid-dir", type=Path, required=True,
              help="Directory with textgrid to process.")#data/processed/esd/english/mfa_outputs
@click.option("--output-dir", type=Path, required=True,
              help="Directory for mels, energy, pitch.")#data/processed/esd/english/fastspeech2
@click.option("--audio-ext", type=str, default="flac", required=True,
              help="Extension of audio files.")
def main(input_audio_dir: Path, input_textgrid_dir: Path, output_dir: Path, audio_ext: str) -> None:
    output_dir.mkdir(exist_ok=True, parents=True)

    stft = Audio.stft.TacotronSTFT(
            FILTER_LENGTH, 
            HOP_SIZE, 
            WIN_SIZE, 
            N_MELS, 
            SAMPLE_RATE, 
            F_MIN, 
            F_MAX
        )


    # Compute pitch, energy, duration, and mel-spectrogram
    textgrid_collection = list(input_textgrid_dir.rglob(f"*.TextGrid")) #{audio_ext}
    print(f"Number of TextGrid files found: {len(textgrid_collection)}")
    print("Compute pitch, energy, duration, and mel-spectrogram...")

    for tg_path in tqdm(textgrid_collection):
        audio_path = input_audio_dir / Path(tg_path.parent.stem) /  Path(tg_path.stem + f".{audio_ext}")
        output = process_utterance(audio_path, tg_path, stft)
        if output == None:
            continue
        duration, pitch, energy, mels = output

        # [size],           [size] [size]  [n_mels x time]
        new_mel_path = output_dir / Path("mels") / Path(tg_path.parent.stem)
        new_pitch_path = output_dir / Path("pitch") / Path(tg_path.parent.stem)
        new_energy_path = output_dir / Path("energy") / Path(tg_path.parent.stem)
        new_duration_path = output_dir / Path("duration") / Path(tg_path.parent.stem)
        #new_phones_path = output_dir / Path("phones") / Path(tg_path.parent.stem)
        
        new_mel_path.mkdir(exist_ok=True, parents=True)
        new_pitch_path.mkdir(exist_ok=True, parents=True)
        new_energy_path.mkdir(exist_ok=True, parents=True)
        new_duration_path.mkdir(exist_ok=True, parents=True)
        #new_phones_path.mkdir(exist_ok=True, parents=True)

        np.save((new_duration_path / tg_path.stem).with_suffix(".npy"), duration)
        np.save((new_pitch_path / tg_path.stem).with_suffix(".npy"), pitch)
        np.save((new_energy_path / tg_path.stem).with_suffix(".npy"), energy)
        np.save((new_mel_path / tg_path.stem).with_suffix(".npy"), mels)
        #open((new_phones_path / tg_path.stem).with_suffix(".txt"), 'wt').write((" ".join(phones)).rstrip())

    print("Finished successfully.")
    print(f"Processed files are located at {output_dir}")

def seconds_to_frame(seconds: float) -> float:
    return seconds * SAMPLE_RATE / HOP_SIZE

def process_utterance(audio_path: Path, textgrid_path: Path,
    stft: Audio.stft.TacotronSTFT
) -> Tuple[List, List, List, List]:
    textgrid = tgt.io.read_textgrid(textgrid_path)
    durations = np.array(
        [
            int(np.round(seconds_to_frame(x.end_time)) - np.round(seconds_to_frame(x.start_time)))
            for x in textgrid.get_tier_by_name("phones").get_copy_with_gaps_filled()
        ],
        dtype=np.int32,
    )

    # Read and trim wav filesmel_spec
    wav, _ = librosa.load(audio_path)

    # Compute fundamental frequency
    pitch, t = pw.dio(
        wav.astype(np.float64),
        SAMPLE_RATE,
        frame_period=HOP_SIZE / SAMPLE_RATE * 1000,
    )
    pitch = pw.stonemask(wav.astype(np.float64), pitch, t, SAMPLE_RATE)
    pitch = pitch[: sum(durations)]
    if np.sum(pitch != 0) <= 1:
        return None

    # Compute mel-scale spectrogram and energy
    mels, energy = Audio.tools.get_mel_from_wav(wav, stft)
    
    mels = mels[:, : sum(durations)]

    energy = energy[: sum(durations)]



    pitch = _convert_to_continuous_f0(pitch)


    # Phoneme-level average
    pos = 0
    for i, d in enumerate(durations):
        if d > 0:
            pitch[i] = np.mean(pitch[pos : pos + d])
        else:
            pitch[i] = 0
        pos += d
    pitch = pitch[: len(durations)]

    # Phoneme-level average
    pos = 0
    for i, d in enumerate(durations):
        if d > 0:
            energy[i] = np.mean(energy[pos : pos + np.round(d + 1).astype(np.int32)])
        else:
            energy[i] = 0
        pos += np.round(d).astype(np.int32)
    energy = energy[: len(durations)]

    return durations, pitch, energy, mels


if __name__ == "__main__":
    main()
