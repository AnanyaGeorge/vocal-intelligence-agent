import numpy as np
import librosa


FEATURE_NAMES = [
    "pitch_mean",
    "pitch_std",
    "pitch_range",
    "rms_mean",
    "rms_std",
    "zcr_mean",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff",
    "mfcc_1",
    "mfcc_2",
    "mfcc_3",
    "mfcc_4",
    "mfcc_5",
    "mfcc_6",
    "mfcc_7",
    "mfcc_8",
    "mfcc_9",
    "mfcc_10",
    "mfcc_11",
    "mfcc_12",
    "mfcc_13",
]


def load_audio(path, sr=16000):
    """Load an audio file as mono speech audio."""
    audio, sample_rate = librosa.load(
        path,
        sr=sr,
        mono=True
    )
    return audio, sample_rate


def _safe_pitch(audio, sr):
    """Estimate fundamental frequency while safely handling silence."""
    if len(audio) < 512:
        return np.array([0.0])

    try:
        pitch = librosa.yin(
            audio,
            fmin=70,
            fmax=400,
            sr=sr
        )

        pitch = pitch[np.isfinite(pitch)]

        if len(pitch) == 0:
            return np.array([0.0])

        return pitch

    except Exception:
        return np.array([0.0])


def extract_features(audio, sr=16000):
    """
    Extract explainable acoustic features from one audio segment.

    Returns:
        numpy array containing the feature vector.
    """

    if audio is None or len(audio) == 0:
        return np.zeros(len(FEATURE_NAMES), dtype=float)

    audio = np.asarray(audio, dtype=float)

    # Remove DC offset
    audio = audio - np.mean(audio)

    # Prevent extremely quiet recordings from causing numerical issues
    if np.max(np.abs(audio)) > 0:
        audio = audio / (np.max(np.abs(audio)) + 1e-8)

    # -------------------------
    # Pitch
    # -------------------------
    pitch = _safe_pitch(audio, sr)

    valid_pitch = pitch[
        (pitch >= 70) &
        (pitch <= 400)
    ]

    if len(valid_pitch) == 0:
        pitch_mean = 0.0
        pitch_std = 0.0
        pitch_range = 0.0
    else:
        pitch_mean = float(np.mean(valid_pitch))
        pitch_std = float(np.std(valid_pitch))
        pitch_range = float(
            np.max(valid_pitch) - np.min(valid_pitch)
        )

    # -------------------------
    # Energy
    # -------------------------
    rms = librosa.feature.rms(y=audio)[0]

    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))

    # -------------------------
    # Zero Crossing Rate
    # -------------------------
    zcr = librosa.feature.zero_crossing_rate(audio)[0]

    zcr_mean = float(np.mean(zcr))

    # -------------------------
    # Spectral features
    # -------------------------
    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sr
    )[0]

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sr
    )[0]

    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sr
    )[0]

    # -------------------------
    # MFCC
    # -------------------------
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=13
    )

    mfcc_means = np.mean(mfcc, axis=1)

    features = [
        pitch_mean,
        pitch_std,
        pitch_range,
        rms_mean,
        rms_std,
        zcr_mean,
        float(np.mean(spectral_centroid)),
        float(np.mean(spectral_bandwidth)),
        float(np.mean(spectral_rolloff)),
    ]

    features.extend(mfcc_means.tolist())

    return np.asarray(features, dtype=float)


def extract_from_path(path, sr=16000):
    """Load an audio file and extract its acoustic feature vector."""
    audio, sample_rate = load_audio(path, sr=sr)
    return extract_features(audio, sr=sample_rate)