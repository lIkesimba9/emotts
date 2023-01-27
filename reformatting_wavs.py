import torchaudio
import soundfile as sf

import numpy as np
import pandas as pd
import os
from glob import glob
from tqdm import tqdm
import torch


def gather_files_from_folder(PATH, ext='wav'):
    return [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], f'*.{ext}'))]


input_wavs_dir = '/media/public/emotts/data/preprocessed/resampled_16'
wav_files = gather_files_from_folder(input_wavs_dir, ext='wav')


TARGET_SR = 22050
#wav_path1 = 'data/preprocessed/resampled_16/0016/0016_000001.wav'
#wav_path2 = 'data/preprocessed/resampled/0019_001251.wav'
#wav_path3 = 'data/preprocessed/resampled/0019_001251_updated.wav'

def remix_resample_write(f_wav, f_out):
    audio, sr = torchaudio.load(f_wav)
    audio = audio.squeeze(0).cpu().detach().numpy()
    sf.write(f_out, audio, TARGET_SR, subtype='PCM_16')
    return f_out

for filename in tqdm(wav_files):
    remix_resample_write(filename, filename)