# MADE 2020-2021 Emotional Text-to-Speech Synthesis

## Basic Architecture
- feature extractor:
    - Non-Attentive Tacotron; implementation based on [NAT repo](https://github.com/Garvit-32/Non-Attentive-Tacotron/);
    - FastSpeech2; implementation based on [FastSpeech2 repo](https://github.com/ming024/FastSpeech2);
    ```
    Input features: phoneme embedding, style embedding, speaker embedding.
    ```

- vocoder:
    - pretrained HiFi-GAN (better performance, but needs fine-tuning); [HiFiGAN repo](https://github.com/jik876/hifi-gan);

      NOTE: pretrained models work with 22.05 kHz (sample rate) audio only);
    ```
    Input features: 80-dim mel spectral features.
    ```


## Data Preprocessing
1. Pausation cutting with VAD ([Silero VAD](https://github.com/snakers4/silero-vad));
2. Resampling audio and converting stereo to mono;
3. Normalization:
    - converting to lowercase;
    - expanding numbers and abbreviations:
        - 123 -> hundred and twenty-three, 10/10/2021 -> tenth October two thousand twenty one;
    - collapsing whitespace;
4. MFA Alignment (external forced aligner):
    - output in TextGrid format (convert to durations: `npy`-array of durations in seconds / in frames);

5. Feature extraction: pitch, energy, duration, mel-spectrogram(80-dim) with Z-normalization and speaker embeddings;

## Style Encoding
- GST-based (global style tokens): prosody transfer model;

## Speaker Encoding
- lookup embedding;
- embedding from pretrained VoicePrint encoder; [source code](https://github.com/CorentinJ/Real-Time-Voice-Cloning/wiki/Pretrained-models);

## Useful Links

### Repositories
- [Tacotron 2](https://github.com/NVIDIA/tacotron2)
- [Non-Annentive Tacotron](https://github.com/Garvit-32/Non-Attentive-Tacotron/)
- [FastSpeech2](https://github.com/ming024/FastSpeech2)
- [HiFiGAN](https://github.com/jik876/hifi-gan)
- [VoicePrint encoder](https://github.com/CorentinJ/Real-Time-Voice-Cloning/wiki/Pretrained-models)
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [MFA](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner)


### Datasets
- [VCTK](https://datashare.ed.ac.uk/handle/10283/3443)
- [FreeST](https://openslr.elda.org/resources/38/ST-CMDS-20170001_1-OS.tar.gz)
- [ESD](https://github.com/HLTSingapore/Emotional-Speech-Data)

### Articles
- [28 Nov 2017] [Emotional End-to-End Neural Speech synthesizer](https://arxiv.org/pdf/1711.05447.pdf)
- [16 Feb 2018] [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://arxiv.org/pdf/1712.05884)
- [06 Aug 2019] [Robust Sequence-to-Sequence Acoustic Modeling with Stepwise Monotonic Attention for Neural TTS](https://arxiv.org/pdf/1906.00672)
- [23 Oct 2020] [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis](https://arxiv.org/pdf/2010.05646)
- [11 May 2021] [Non-Attentive Tacotron: Robust and Controllable Neural TTS Synthesis Including Unsupervised Duration Modeling](https://arxiv.org/pdf/2010.04301)

## Experiments

### Preprocessing

#### Preprocessing VCTK
- Move ```vctk.zip``` file to ```vctk_preprocessing/data/zip/vctk.zip``` and run
```
bash vctk_preprocessing/vctk_preprocessing.sh
```

#### Preprocessing ESD
- Move ```Emotional Speech Dataset (ESD).zip``` file to ```esd_preprocessing/data/zip/Emotional Speech Dataset (ESD).zip``` and run
```
bash esd_preprocessing/esd_preprocessing.sh
```

#### Merge VCTK and ESD
- You need to merge ```data/processed/vctk``` and ```data/processed/esd``` in ```data/preprocessed/``` with such subdirectories
    - ```data/preprocessed/mels```
    - ```data/preprocessed/resampled```
    - ```data/preprocessed/duration ```
    - ```data/preprocessed/pitch ```
    - ```data/preprocessed/energy```
    - ```data/preprocessed/phones ```
    - ```data/preprocessed/embeddings```
- Open config file from config/ and set path to data.
- Also download hifi and set up ```pretrained_hifi: /path/to/models/hifi```

### Training Base Tacotron
#### Training
```
conda env create -f environment.yml
conda activate emmots
python train_non_attentive_voiceprint.py --config configs/nat_inflated/nat_inflated.yml
```

#### Change last checkpoint for tuning
Rename ```checkpoints/nat_inflated/feature/500000_feature_model.pth``` into
```checkpoints/nat_inflated/feature/feature_model.pth```
#### Tuning
- Check ```data``` and ```pretrained_hifi``` in ```nat_inflated_tune.yml``` (same as in the ```nat_inflated.yml```)
```
python train_non_attentive_voiceprint.py --config configs/nat_inflated/nat_inflated_tune.yml
```

### Tensorboard
```
tensorboard --logdir logs/
```

## NOTE:
- Gentle (claimes to be able to align non-verbal emotion expression) (NOTE: for English only);
