#!/bin/bash
conda activate emotts
cd repo

export OUTPUT_DIR=data
:'
[ -d "$OUTPUT_DIR/processed/esd/" ] && rm -rf $OUTPUT_DIR/processed/esd/

echo -e "\n1. Extract from zip"
python src/preprocessing/extract_esd.py --input-esd-zip $OUTPUT_DIR/zip/"Emotional Speech Dataset (ESD).zip" --output-dir $OUTPUT_DIR/processed/esd --language $language

conda deactivate
conda activate pausation

echo -e "\n2. Pausation cutting with VAD"
python src/preprocessing/pausation_cutting.py --input-dir $OUTPUT_DIR/processed/esd/$language/audio --output-dir $OUTPUT_DIR/processed/esd/$language/no_pause --target-sr 48000 --audio-ext wav

conda deactivate
conda activate emotts

echo -e "\n3. Resampling"
python src/preprocessing/resampling.py --input-dir $OUTPUT_DIR/processed/esd/$language/no_pause --output-dir $OUTPUT_DIR/processed/esd/$language/resampled --resample-rate 22050 --audio-ext wav


[ -d "$OUTPUT_DIR/processed/esd/mfa_inputs/" ] && rm -rf $OUTPUT_DIR/processed/esd/chinese/mfa_inputs/
[ -d "$OUTPUT_DIR/processed/esd/mfa_outputs/" ] && rm -rf $OUTPUT_DIR/processed/esd/chinese/mfa_outputs/

echo -e "\n5. Text normalization"
python src/preprocessing/text_normalization.py --input-dir $OUTPUT_DIR/processed/esd/$language/text --output-dir $OUTPUT_DIR/processed/esd/$language/mfa_inputs --language $language 
'

echo -e "\n6. MFA Alignment setup"

# download a pretrained english acoustic model, and english lexicon
mkdir -p models
[ $language == "chinese" ] && wget -q --show-progress https://github.com/lIkesimba9/FreeST_mfa_align/raw/main/model/freest.zip -P models
[ $language == "chinese" ] && wget -q --show-progress https://raw.githubusercontent.com/lIkesimba9/FreeST_mfa_align/main/model/pinyin-lexicon_with_tab.dict -P models

[ $language == "english" ] && wget -q --show-progress https://github.com/MontrealCorpusTools/mfa-models/raw/main/acoustic/english.zip -P models
[ $language == "english" ] && wget -q --show-progress http://www.openslr.org/resources/11/librispeech-lexicon.txt -P models

conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib  # link to libopenblas
conda deactivate
conda activate emotts
:'
echo -e "\n7. MFA Preprocessing"
python src/preprocessing/mfa_preprocessing.py --input-dir $OUTPUT_DIR/processed/esd/$language/resampled --output-dir $OUTPUT_DIR/processed/esd/$language/mfa_inputs
'

# FINALLY, align phonemes and speech

echo -e "\n8. MFA Alignment"

[ $language == "english" ] && mfa align -t ./temp --clean -j 4 $OUTPUT_DIR/processed/esd/$language/mfa_inputs models/librispeech-lexicon.txt models/english.zip $OUTPUT_DIR/processed/esd/$language/mfa_outputs
[ $language == "chinese" ] && mfa align -t ./temp --clean -j 4 $OUTPUT_DIR/processed/esd/$language/mfa_inputs models/pinyin-lexicon_with_tab.dict models/freest.zip $OUTPUT_DIR/processed/esd/$language/mfa_outputs

rm -rf temp
:'
echo -e "\n9. MFA Postprocessing"
# Aggregate mels by speakers
python src/preprocessing/mfa_postprocessing.py --input-dir $OUTPUT_DIR/processed/esd/$language/resampled
'

echo -e "\n10. Compute pitch, mels, energy, duration for fastspeech2"

python src/preprocessing/enrgy_mel_pitch_for_fastspeech2.py --input-audio-dir $OUTPUT_DIR/processed/esd/$language/resampled --input-textgrid-dir $OUTPUT_DIR/processed/esd/$language/mfa_outputs  --output-dir $OUTPUT_DIR/processed/esd/$language/fastspeech2 --audio-ext wav

:'
gdown --fuzzy  https://drive.google.com/file/d/1q8mEGwCkFy23KZsinbuvdKAQLqNKbYf1/view?usp=sharing -O models/encoder.pt
echo -e "\n11. Compute embeddings"
python src/preprocessing/generate_speaker_embegigs.py --input-dir $OUTPUT_DIR/processed/esd/$language/resampled --output-dir $OUTPUT_DIR/processed/esd/$language/embeddings  --audio-ext wav --model-path models/encoder.pt
'
