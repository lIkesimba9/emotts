#!/bin/bash
conda activate emotts
cd repo

export OUTPUT_DIR=data

[ -d "$OUTPUT_DIR/processed/freest/" ] && rm -rf $OUTPUT_DIR/processed/freest/

tar -xf data/zip/ST-CMDS-20170001_1-OS.tar.gz

echo -e "\n1. Sorting"
python src/preprocessing/sorting_freest.py --input-dir ST-CMDS-20170001_1-OS --output-dir $OUTPUT_DIR/processed/freest

conda deactivate
conda activate pausation

echo -e "\n2. Pausation cutting with VAD"
python src/preprocessing/pausation_cutting.py --input-dir $OUTPUT_DIR/processed/freest/audio --output-dir $OUTPUT_DIR/processed/freest/no_pause --target-sr 48000 --audio-ext wav

conda deactivate
conda activate emotts

echo -e "\n3. Resampling"
python src/preprocessing/resampling.py --input-dir $OUTPUT_DIR/processed/freest/no_pause --output-dir $OUTPUT_DIR/processed/freest/resampled --resample-rate 22050 --audio-ext wav

echo -e "\n4. Audio to Mel"
python src/preprocessing/wav_to_mel.py --input-dir $OUTPUT_DIR/processed/freest/resampled --output-dir $OUTPUT_DIR/processed/freest/mels  --audio-ext wav

echo -e "\n5. Text normalization"
python src/preprocessing/text_normalization.py --input-dir $OUTPUT_DIR/processed/freest/text --output-dir $OUTPUT_DIR/processed/freest/mfa_inputs --language chinese 

echo -e "\n6. MFA Alignment setup"

# download a pretrained english acoustic model, and english lexicon
mkdir -p models
wget -q --show-progress https://github.com/lIkesimba9/FreeST_mfa_align/raw/main/model/freest.zip -P models
wget -q --show-progress https://raw.githubusercontent.com/lIkesimba9/FreeST_mfa_align/main/pinyin-lexicon_with_tab.txt -P models

conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib  # link to libopenblas
conda deactivate
conda activate emotts

echo -e "\n7. MFA Preprocessing"
python src/preprocessing/mfa_preprocessing.py --input-dir $OUTPUT_DIR/processed/freest/resampled --output-dir $OUTPUT_DIR/processed/freest/mfa_inputs

# FINALLY, align phonemes and speech
echo -e "\n8. MFA Alignment"

mfa align -t ./temp --clean -j 4 $OUTPUT_DIR/processed/freest/mfa_inputs models/pinyin-lexicon_with_tab.txt models/freest.zip $OUTPUT_DIR/processed/freest/mfa_outputs
rm -rf temp

echo -e "\n9. MFA Postprocessing"
# Aggregate mels by speakers
python src/preprocessing/mfa_postprocessing.py --input-dir $OUTPUT_DIR/processed/freest/mels
