from pathlib import Path
import re
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

from feature_engine import extract_from_path, FEATURE_NAMES


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

CREMA_DIR = ROOT / "data" / "CREMA-D" / "AudioWAV"
MODEL_DIR = ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "acoustic_state_model.joblib"


# ============================================================
# CREMA-D LABELS
# ============================================================

EMOTION_MAP = {
    "ANG": "anger",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}


def parse_cremad_filename(filename):
    """
    CREMA-D filename format contains:

        ActorID_Sentence_Emotion_Level.wav

    Example:

        1001_DFA_ANG_XX.wav

    Returns:
        speaker_id, emotion
    """

    stem = Path(filename).stem
    parts = stem.split("_")

    if len(parts) < 3:
        return None, None

    speaker_id = parts[0]
    emotion_code = parts[2]

    emotion = EMOTION_MAP.get(emotion_code)

    return speaker_id, emotion


# ============================================================
# LOAD DATASET
# ============================================================

def load_cremad():

    if not CREMA_DIR.exists():
        raise FileNotFoundError(
            f"\nCREMA-D folder not found:\n{CREMA_DIR}\n\n"
            "Expected structure:\n"
            "data/CREMA-D/AudioWAV/*.wav"
        )

    audio_files = sorted(CREMA_DIR.glob("*.wav"))

    if len(audio_files) == 0:
        raise RuntimeError(
            f"No WAV files found in:\n{CREMA_DIR}"
        )

    print(f"Found {len(audio_files)} CREMA-D audio files.")

    features = []
    labels = []
    speakers = []

    for i, audio_path in enumerate(audio_files, start=1):

        speaker_id, emotion = parse_cremad_filename(
            audio_path.name
        )

        if emotion is None:
            continue

        try:
            feature_vector = extract_from_path(audio_path)

            if not np.all(np.isfinite(feature_vector)):
                continue

            features.append(feature_vector)
            labels.append(emotion)
            speakers.append(speaker_id)

        except Exception as e:
            print(
                f"Skipping {audio_path.name}: {e}"
            )

        if i % 250 == 0:
            print(
                f"Processed {i}/{len(audio_files)} files..."
            )

    X = np.asarray(features)
    y = np.asarray(labels)
    groups = np.asarray(speakers)

    print("\nDataset loaded.")
    print(f"Samples: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Speakers: {len(np.unique(groups))}")

    return X, y, groups


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(X, y, groups):

    # --------------------------------------------------------
    # Speaker-independent split
    # --------------------------------------------------------

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups)
    )

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    print("\nSpeaker-independent split:")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")

    # --------------------------------------------------------
    # Pipeline
    # --------------------------------------------------------

    model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),

        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=14,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced"
            )
        )
    ])

    print("\nTraining model...")

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\n==============================")
    print("MODEL EVALUATION")
    print("==============================")

    print(
        f"\nSpeaker-independent accuracy: "
        f"{accuracy:.3f}"
    )

    print("\nClassification report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    artifact = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "classes": model.classes_.tolist(),
        "dataset": "CREMA-D",
        "task": "acoustic_emotional_state_classification",
    }

    joblib.dump(
        artifact,
        MODEL_PATH
    )

    print("\nModel saved to:")

    print(MODEL_PATH)

    return model


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("======================================")
    print("VOCAL INTELLIGENCE AGENT")
    print("Acoustic Model Training")
    print("======================================")

    X, y, groups = load_cremad()

    train_model(
        X,
        y,
        groups
    )

    print("\nTraining complete.")