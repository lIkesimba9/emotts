#!/bin/bash
conda activate emotts
cd repo
export OUTPUT_DIR=data


echo -e "\n1. Rename audio by chapter"
python src/preprocessing/blizzard_rename_audio.py --input-dir $OUTPUT_DIR/books/segments --output-dir $OUTPUT_DIR/processed/blizzard/audio

echo -e "\n2. Split text by chapter"
python src/preprocessing/blizzard_split_text.py --input-dir $OUTPUT_DIR/books/transcripts --output-dir $OUTPUT_DIR/processed/blizzard/text



echo -e "\n3. Resampling"
python src/preprocessing/resampling.py --input-dir $OUTPUT_DIR/processed/blizzard/audio --output-dir $OUTPUT_DIR/processed/blizzard/resampled --resample-rate 22050 --audio-ext wav


echo -e "\n4. Text normalization"
python src/preprocessing/text_normalization.py --input-dir $OUTPUT_DIR/processed/blizzard/text --output-dir $OUTPUT_DIR/processed/blizzard/mfa_inputs

echo -e "\n5. MFA Alignment setup"

# download a pretrained english acoustic model, and english lexicon
mkdir -p models
wget -q --show-progress https://github.com/MontrealCorpusTools/mfa-models/raw/main/acoustic/english.zip -P models

wget -q --show-progress http://www.openslr.org/resources/11/librispeech-lexicon.txt -P models
LEXICON_PATH="models/librispeech-lexicon.txt"

echo "You use this lexicon: $LEXICON_PATH"

conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib  # link to libopenblas
conda deactivate
conda activate emotts

echo -e "\n6. MFA Preprocessing"
python src/preprocessing/mfa_preprocessing.py --input-dir $OUTPUT_DIR/processed/blizzard/resampled --output-dir $OUTPUT_DIR/processed/blizzard/mfa_inputs

# FINALLY, align phonemes and speech
echo -e "\n7. MFA Alignment"

mfa align -t ./temp --clean -j 4 $OUTPUT_DIR/processed/blizzard/mfa_inputs $LEXICON_PATH models/english.zip $OUTPUT_DIR/processed/blizzard/mfa_outputs
rm -rf temp


[ -d "$OUTPUT_DIR/processed/blizzard/fastspeech2/" ] && rm -rf $OUTPUT_DIR/processed/blizzard/fastspeech2/

echo -e "\n9. Compute pitch, mels, energy, duration for fastspeech2"

python src/preprocessing/enrgy_mel_pitch_for_fastspeech2.py --input-audio-dir $OUTPUT_DIR/processed/blizzard/resampled --input-textgrid-dir $OUTPUT_DIR/processed/blizzard/mfa_outputs  --output-dir $OUTPUT_DIR/processed/blizzard/fastspeech2 --audio-ext wav

gdown --fuzzy  https://drive.google.com/file/d/1q8mEGwCkFy23KZsinbuvdKAQLqNKbYf1/view?usp=sharing -O models/encoder.pt

echo -e "\n10. Compute embeddings"
python src/preprocessing/generate_speaker_embegigs.py --input-dir $OUTPUT_DIR/processed/blizzard/resampled --output-dir $OUTPUT_DIR/processed/blizzard/embeddings  --audio-ext wav --model-path models/encoder.pt
