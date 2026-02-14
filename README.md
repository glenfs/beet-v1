# Beethove_v3

SATB preprocessing, training, and inference pipeline for chorale-style generation.

**Entry Points**
1. `preprocess_satb.py`  
   Builds the dataset, mapping, and training sequences.
2. `train_model_foundational_v3.py`  
   Trains bass, alto, and tenor models.
3. `inference_v2.py`  
   Generates full SATB parts from a soprano line.

**Pipeline (Typical)**
1. Preprocess data:
   - `python preprocess_satb.py`
2. Train models:
   - `python train_model_foundational_v3.py`
3. Run inference:
   - `python inference_v2.py`

**Notes**
- Paths and dataset names are configured in `constants.py`.
- `SAVE_DIR` and `VAL_DIR` must contain preprocessed files in the expected SATB+beats+fermata format.
- Fermata encoding currently uses score annotations when present.
- Barline encoding adds a binary channel marking the final time-step of each measure.
- Normalized bar position is optional (see `USE_POS_ENCODING` in `constants.py`).
