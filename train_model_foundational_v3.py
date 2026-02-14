import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from satb_models import (
    build_bass_model,
    build_alto_model,
    build_tenor_model
)
from preprocess_satb import midi_to_chorale_data, get_mapping
from constants import (
    SAVE_DIR,
    SEQUENCE_LENGTH,
    SAVE_MODEL_DIR,
    VAL_DIR,
    USE_POS_ENCODING,
    HOLD_WEIGHT,
    REST_WEIGHT,
    USE_MODE_ENCODING,
)

# ======================================================
# CONFIG
# ======================================================

BATCH_SIZE = 256
EPOCHS = 125
STEP_SIZE = 16

os.makedirs(SAVE_DIR, exist_ok=True)


# ======================================================
# DATA WINDOWING
# ======================================================

def _window_chorales(chorales, sequence_length):
    S, B, A, T = [], [], [], []
    beats, fermata, barlines, modes, pos = [], [], [], [], []

    for chorale in chorales:
        song_len = len(chorale[0])
        if song_len < sequence_length:
            continue

        for i in range(0, song_len - sequence_length, STEP_SIZE):
            s = chorale[0][i:i + sequence_length]
            a = chorale[1][i:i + sequence_length]
            t = chorale[2][i:i + sequence_length]
            b = chorale[3][i:i + sequence_length]
            bt = chorale[4][i:i + sequence_length]
            f = chorale[5][i:i + sequence_length]
            bl = chorale[6][i:i + sequence_length]

            idx = 7
            m = None
            if USE_MODE_ENCODING:
                m = chorale[idx][i:i + sequence_length]
                idx += 1
            p = chorale[idx][i:i + sequence_length] if USE_POS_ENCODING else None

            required = [s, a, t, b, bt, f, bl]
            if USE_MODE_ENCODING:
                required.append(m)
            if USE_POS_ENCODING:
                required.append(p)

            if all(len(x) == sequence_length for x in required):
                S.append(s)
                B.append(b)
                A.append(a)
                T.append(t)
                beats.append(bt)
                fermata.append(f)
                barlines.append(bl)
                if USE_MODE_ENCODING:
                    modes.append(m)
                if USE_POS_ENCODING:
                    pos.append(p)

    data = (
        np.array(S, dtype="int32"),
        np.array(B, dtype="int32"),
        np.array(A, dtype="int32"),
        np.array(T, dtype="int32"),
        np.array(beats, dtype="int32"),
        np.array(fermata, dtype="int32"),
        np.array(barlines, dtype="int32"),
    )

    if USE_MODE_ENCODING:
        data = (*data, np.array(modes, dtype="int32"))
    if USE_POS_ENCODING:
        data = (*data, np.array(pos, dtype="int32"))

    return data

def _load_dataset_from_dir(dataset_dir):
    print("ðŸŽ¼ Loading mappings and chorales...")
    mappings = get_mapping()
    num_classes = len(mappings)

    chorale_data = midi_to_chorale_data(dataset_dir, mappings)

    print("ðŸ”¹ Windowing sequences...")
    data = _window_chorales(chorale_data, SEQUENCE_LENGTH)

    print("ðŸ”¹ Converting to numpy...")
    return (*data, num_classes)


def load_dataset():
    return _load_dataset_from_dir(SAVE_DIR)


def load_val_dataset():
    return _load_dataset_from_dir(VAL_DIR)


# ======================================================
# TRAIN HELPERS
# ======================================================

def _get_special_token_ids(mappings):
    hold_id = mappings.get("_")
    rest_id = mappings.get("r")
    ignore_ids = [i for i in [hold_id, rest_id] if i is not None]
    return hold_id, rest_id, ignore_ids


def _masked_accuracy(ignore_ids):
    ignore_ids_tf = tf.constant(ignore_ids, dtype=tf.int32) if ignore_ids else None

    def metric(y_true, y_pred):
        y_true_int = tf.cast(y_true, tf.int32)
        y_pred_ids = tf.argmax(y_pred, axis=-1, output_type=tf.int32)

        if ignore_ids_tf is None:
            matches = tf.cast(tf.equal(y_true_int, y_pred_ids), tf.float32)
            return tf.reduce_mean(matches)

        mask = tf.logical_not(tf.reduce_any(tf.equal(
            tf.expand_dims(y_true_int, -1), ignore_ids_tf
        ), axis=-1))
        mask_f = tf.cast(mask, tf.float32)
        matches = tf.cast(tf.equal(y_true_int, y_pred_ids), tf.float32) * mask_f
        return tf.math.divide_no_nan(tf.reduce_sum(matches), tf.reduce_sum(mask_f))

    metric.__name__ = "masked_accuracy_no_hold_rest"
    return metric


def _make_sample_weights(labels, hold_id, rest_id, hold_weight=HOLD_WEIGHT, rest_weight=REST_WEIGHT):
    if hold_id is None and rest_id is None:
        return None
    weights = np.ones_like(labels, dtype="float32")
    if hold_id is not None:
        weights[labels == hold_id] = hold_weight
    if rest_id is not None:
        weights[labels == rest_id] = rest_weight
    return weights


def _log_token_stats(name, arr, hold_id, rest_id):
    flat = arr.reshape(-1)
    total = flat.size
    hold_count = int(np.sum(flat == hold_id)) if hold_id is not None else 0
    rest_count = int(np.sum(flat == rest_id)) if rest_id is not None else 0
    other_count = total - hold_count - rest_count
    print(f"{name} token stats:")
    print(f"  total={total} hold(_ )={hold_count} rest(r)={rest_count} other={other_count}")
    if total > 0:
        print(f"  pct_hold={hold_count / total:.3f} pct_rest={rest_count / total:.3f}")


def _assert_training_tensors(inputs, labels, name, num_classes):
    n = labels.shape[0]
    if n == 0:
        raise ValueError(
            f"{name}: no training windows were created (labels shape={labels.shape}). "
            "Check dataset paths, preprocessing output, and SEQUENCE_LENGTH/STEP_SIZE."
        )

    for i, x in enumerate(inputs):
        if x.shape[0] != n:
            raise ValueError(
                f"{name}: input[{i}] batch size {x.shape[0]} does not match labels batch size {n}."
            )
        if x.size == 0:
            raise ValueError(f"{name}: input[{i}] is empty (shape={x.shape}).")

    if labels.size == 0:
        raise ValueError(f"{name}: labels are empty (shape={labels.shape}).")

    if np.any(labels < 0) or np.any(labels >= num_classes):
        min_id = int(np.min(labels))
        max_id = int(np.max(labels))
        raise ValueError(
            f"{name}: label ids out of range [0, {num_classes - 1}] "
            f"(min={min_id}, max={max_id})."
        )

    if not np.isfinite(labels).all():
        raise ValueError(f"{name}: labels contain NaN/Inf.")


def compile_and_train(model, inputs, labels, inputs_v, labels_v, save_name, hold_id, rest_id, ignore_ids):
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=11,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=SAVE_MODEL_DIR + "best_" + save_name,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]

    num_classes = model.output_shape[-1]
    _assert_training_tensors(inputs, labels, f"{save_name} train", num_classes)
    _assert_training_tensors(inputs_v, labels_v, f"{save_name} val", num_classes)
    print(
        f"{save_name}: train_n={labels.shape[0]} val_n={labels_v.shape[0]} "
        f"seq_len={labels.shape[1]}"
    )

    model.compile(
        optimizer=Adam(3e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", _masked_accuracy(ignore_ids)],
        weighted_metrics=[]
    )

    sample_weights = _make_sample_weights(labels, hold_id, rest_id)
    sample_weights_v = _make_sample_weights(labels_v, hold_id, rest_id)

    model.fit(
        inputs,
        labels,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(inputs_v, labels_v, sample_weights_v) if sample_weights_v is not None else (inputs_v, labels_v),
        callbacks=callbacks,
        sample_weight=sample_weights
    )

    model.save(SAVE_MODEL_DIR + save_name)


# ======================================================
# MAIN
# ======================================================

def main():
    if USE_MODE_ENCODING and USE_POS_ENCODING:
        S, B, A, T, beats, fermata, barlines, mode, pos, num_classes = load_dataset()
        Sv, Bv, Av, Tv, beatsv, fermatav, barlinesv, modev, posv, _ = load_val_dataset()
    elif USE_MODE_ENCODING:
        S, B, A, T, beats, fermata, barlines, mode, num_classes = load_dataset()
        Sv, Bv, Av, Tv, beatsv, fermatav, barlinesv, modev, _ = load_val_dataset()
    elif USE_POS_ENCODING:
        S, B, A, T, beats, fermata, barlines, pos, num_classes = load_dataset()
        Sv, Bv, Av, Tv, beatsv, fermatav, barlinesv, posv, _ = load_val_dataset()
    else:
        S, B, A, T, beats, fermata, barlines, num_classes = load_dataset()
        Sv, Bv, Av, Tv, beatsv, fermatav, barlinesv, _ = load_val_dataset()

    mappings = get_mapping()
    hold_id, rest_id, ignore_ids = _get_special_token_ids(mappings)

    _log_token_stats("Soprano", S, hold_id, rest_id)
    _log_token_stats("Bass", B, hold_id, rest_id)
    _log_token_stats("Alto", A, hold_id, rest_id)
    _log_token_stats("Tenor", T, hold_id, rest_id)

    print("\n==============================")
    print("Training Bass model (S â†’ B)")
    print("==============================")

    bass_model = build_bass_model(SEQUENCE_LENGTH, num_classes)
    bass_inputs = [S, beats, fermata, barlines]
    bass_inputs_v = [Sv, beatsv, fermatav, barlinesv]
    if USE_MODE_ENCODING:
        bass_inputs.append(mode)
        bass_inputs_v.append(modev)
    if USE_POS_ENCODING:
        bass_inputs.append(pos)
        bass_inputs_v.append(posv)

    compile_and_train(
        bass_model,
        bass_inputs,
        B,
        bass_inputs_v,
        Bv,
        "model_bass.h5",
        hold_id,
        rest_id,
        ignore_ids
    )

    print("\n==============================")
    print("Training Alto model (S+B â†’ A)")
    print("==============================")

    alto_model = build_alto_model(SEQUENCE_LENGTH, num_classes)
    alto_inputs = [S, B, beats, fermata, barlines]
    alto_inputs_v = [Sv, Bv, beatsv, fermatav, barlinesv]
    if USE_MODE_ENCODING:
        alto_inputs.append(mode)
        alto_inputs_v.append(modev)
    if USE_POS_ENCODING:
        alto_inputs.append(pos)
        alto_inputs_v.append(posv)

    compile_and_train(
        alto_model,
        alto_inputs,
        A,
        alto_inputs_v,
        Av,
        "model_alto.h5",
        hold_id,
        rest_id,
        ignore_ids
    )

    print("\n==============================")
    print("Training Tenor model (S+B+A â†’ T)")
    print("==============================")

    tenor_model = build_tenor_model(SEQUENCE_LENGTH, num_classes)
    tenor_inputs = [S, B, A, beats, fermata, barlines]
    tenor_inputs_v = [Sv, Bv, Av, beatsv, fermatav, barlinesv]
    if USE_MODE_ENCODING:
        tenor_inputs.append(mode)
        tenor_inputs_v.append(modev)
    if USE_POS_ENCODING:
        tenor_inputs.append(pos)
        tenor_inputs_v.append(posv)

    compile_and_train(
        tenor_model,
        tenor_inputs,
        T,
        tenor_inputs_v,
        Tv,
        "model_tenor.h5",
        hold_id,
        rest_id,
        ignore_ids
    )

    print("\nâœ… All models trained and saved!")


if __name__ == "__main__":
    main()

