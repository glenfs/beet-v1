# Beethove_v3 Repo Guide

## What This Project Is
Beethove_v3 is a SATB chorale generation pipeline built with `music21` and `tensorflow`.

Core workflow:
1. Preprocess MIDI/MusicXML into tokenized SATB training data.
2. Train separate sequence models for Bass, Alto, and Tenor generation.
3. Run inference from a soprano input and export generated score/MIDI.

Main entry points:
- `preprocess_satb.py`
- `train_model_foundational_v3.py`
- `inference_v2.py`
- `inference_v3_duration.py` (alternative inference with duration-focused decoding)

## How To Run (Build/Test/Train/Infer)
This repo does not have a separate compile/build step.

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Preprocess dataset
```bash
python preprocess_satb.py
```

### 3. Train models
```bash
python train_model_foundational_v3.py
```

### 4. Run inference
```bash
python inference_v2.py
```

Optional duration-focused inference:
```bash
python inference_v3_duration.py
```

### 5. Run smoke test
```bash
python smoke_test_preprocess.py
```

## Coding Rules
- Keep token format consistent with existing pipeline: SATB voices plus rhythm/control channels separated by `^`.
- Preserve sequence length assumptions (`SEQUENCE_LENGTH`) across preprocessing, training, and inference.
- Use integer token IDs that match `beethovenAi_bach_v1.2.json`; do not silently remap IDs.
- Keep optional feature flags (`USE_MODE_ENCODING`, `USE_POS_ENCODING`) backward-compatible in new code.
- Prefer explicit shape/range checks before model training/inference to avoid silent failures.
- When editing inference, keep output export stable (MusicXML + MIDI) and preserve key/time signature restoration behavior.

## Tokenization Logic (How It Works)
Tokenization is time-step based (16th-note resolution by default).

### Time step and duration expansion
- `TIME_STEP = 0.25` quarter lengths (in `preprocess_satb.py`).
- Each note/rest is expanded into multiple tokens:
1. First step uses the event symbol.
2. Remaining steps use `_` (hold/continuation).

Example:
- Quarter note C4 (MIDI 60) -> `60 _ _ _`
- Half rest -> `r _ _ _ _ _ _ _`

### Symbol vocabulary
- Note-on: MIDI integer token (e.g., `60`)
- Rest: `r`
- Hold/continue: `_`
- Delimiter in single-file dataset: `/`
- Voice/channel separator in saved song strings: `^`

### Voice layout in encoded song text
Base encoded song uses SATB order:
1. Soprano
2. Alto
3. Tenor
4. Bass

Serialized as:
`soprano ^ alto ^ tenor ^ bass`

### Extra aligned channels
Additional per-time-step channels are appended after SATB:
- Beats
- Fermata
- Barlines
- Optional Mode (if `USE_MODE_ENCODING=True`)
- Optional Position (if `USE_POS_ENCODING=True`)

So final serialized sample is:
- Without optional channels:  
`S ^ A ^ T ^ B ^ beats ^ fermata ^ barlines`
- With mode and position:  
`S ^ A ^ T ^ B ^ beats ^ fermata ^ barlines ^ mode ^ position`

### Alignment rule (critical)
- All channels for a sample must have identical time-step length.
- This alignment is required for windowing (`SEQUENCE_LENGTH`) and model input tensors.

### Training/inference implications
- `_` is usually the most frequent token due to duration expansion.
- Raw token accuracy can be misleadingly high from predicting `_`.
- `masked_accuracy_no_hold_rest` is used to evaluate note-event quality by ignoring `_` and `r`.

## Where Config Lives
Primary config file:
- `constants.py`

Important config groups in `constants.py`:
- Dataset paths and output folders (`MIDI_DATASET_PATH`, `SAVE_DIR`, `VAL_DIR`)
- Mapping and dataset files (`MAPPING_PATH`, `SINGLE_FILE_DATASET`)
- Model save paths (`SAVE_MODEL_DIR`, `SAVE_MODEL_PATH`, etc.)
- Sequence settings (`SEQUENCE_LENGTH`)
- Optional feature toggles (`USE_MODE_ENCODING`, `USE_POS_ENCODING`, bins)
- Training loss weights (`HOLD_WEIGHT`, `REST_WEIGHT`)

Other operational defaults:
- Training hyperparameters in `train_model_foundational_v3.py` (`BATCH_SIZE`, `EPOCHS`, `STEP_SIZE`)
- Inference input folder and decoding settings in `inference_v2.py` / `inference_v3_duration.py`
