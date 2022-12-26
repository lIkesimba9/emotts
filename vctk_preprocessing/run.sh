#!/bin/bash
conda activate emotts
cd repo
export OUTPUT_DIR=data
: '
cd repo/data

export OUTPUT_DIR=data

# Unzip dataset and reorganize folders

[ -d "raw/" ] && rm -rf raw/
[ -d "processed/vctk/" ] && rm -rf processed/vctk/

unzip -q zip/vctk.zip txt/* wav48_silence_trimmed/*

mkdir -p raw/text
mv txt/* raw/text
mkdir -p raw/audio
mv wav48_silence_trimmed/* raw/audio
rm -rf txt wav48_silence_trimmed

cd ..

echo -e "\n1. Selecting only one mic per speaker"
python src/preprocessing/preprocessing.py --input-dir $OUTPUT_DIR/raw/audio --output-dir $OUTPUT_DIR/processed/vctk/audio_single_mic --audio-ext flac

conda deactivate
conda activate pausation

echo -e "\n2. Pausation cutting with VAD"
python src/preprocessing/pausation_cutting.py --input-dir $OUTPUT_DIR/processed/vctk/audio_single_mic --output-dir $OUTPUT_DIR/processed/vctk/no_pause --target-sr 48000

conda deactivate
conda activate emotts

echo -e "\n3. Resampling"
python src/preprocessing/resampling.py --input-dir $OUTPUT_DIR/processed/vctk/no_pause --output-dir $OUTPUT_DIR/processed/vctk/resampled --resample-rate 22050
'
[ -d "$OUTPUT_DIR/processed/vctk/mfa_inputs/" ] && rm -rf $OUTPUT_DIR/processed/vctk/mfa_inputs/
[ -d "$OUTPUT_DIR/processed/vctk/mfa_outputs/" ] && rm -rf $OUTPUT_DIR/processed/vctk/mfa_outputs/

echo -e "\n4. Text normalization"
python src/preprocessing/text_normalization.py --input-dir $OUTPUT_DIR/raw/text --output-dir $OUTPUT_DIR/processed/vctk/mfa_inputs

echo -e "\n5. MFA Alignment setup"

# download a pretrained english acoustic model, and english lexicon
mkdir -p models
wget -q --show-progress https://github.com/MontrealCorpusTools/mfa-models/raw/main/acoustic/english.zip -P models

LEXICON_PATH="models/vctk.h-arpabet-lexicon.txt"
if [ ! -f "models/vctk.h-arpabet-lexicon.txt" ]; then
    echo "don't exist $LEXICON_PATH then lexicon download from http://www.openslr.org/resources/11/librispeech-lexicon.txt ."
    wget -q --show-progress http://www.openslr.org/resources/11/librispeech-lexicon.txt -P models
    LEXICON_PATH="models/librispeech-lexicon.txt"
fi

echo "You use this lexicon: $LEXICON_PATH"

conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib  # link to libopenblas
conda deactivate
conda activate emotts
: '
echo -e "\n6. MFA Preprocessing"
python src/preprocessing/mfa_preprocessing.py --input-dir $OUTPUT_DIR/processed/vctk/resampled --output-dir $OUTPUT_DIR/processed/vctk/mfa_inputs
'
# FINALLY, align phonemes and speech
echo -e "\n7. MFA Alignment"

mfa align -t ./temp --clean -j 4 $OUTPUT_DIR/processed/vctk/mfa_inputs $LEXICON_PATH models/english.zip $OUTPUT_DIR/processed/vctk/mfa_outputs
rm -rf temp

echo -e "\n8. MFA Postprocessing"
# Aggregate mels by speakers
python src/preprocessing/mfa_postprocessing.py --input-dir $OUTPUT_DIR/processed/vctk/resampled

[ -d "$OUTPUT_DIR/processed/vctk/fastspeech2/" ] && rm -rf $OUTPUT_DIR/processed/vctk/fastspeech2/

echo -e "\n9. Compute pitch, mels, energy, duration for fastspeech2"

python src/preprocessing/enrgy_mel_pitch_for_fastspeech2.py --input-audio-dir $OUTPUT_DIR/processed/vctk/resampled --input-textgrid-dir $OUTPUT_DIR/processed/vctk/mfa_outputs  --output-dir $OUTPUT_DIR/processed/vctk/fastspeech2 --audio-ext flac
: '
gdown --fuzzy  https://drive.google.com/file/d/1q8mEGwCkFy23KZsinbuvdKAQLqNKbYf1/view?usp=sharing -O models/encoder.pt
echo -e "\n10. Compute embeddings"
python src/preprocessing/generate_speaker_embegigs.py --input-dir $OUTPUT_DIR/processed/vctk/resampled --output-dir $OUTPUT_DIR/processed/vctk/embeddings  --audio-ext flac --model-path models/encoder.pt

'