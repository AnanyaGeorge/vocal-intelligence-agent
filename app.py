from pathlib import Path
import time
import json

import joblib
import numpy as np
import streamlit as st
from scipy.signal import resample_poly
from streamlit_webrtc import webrtc_streamer, WebRtcMode

from feature_engine import extract_features


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "acoustic_state_model.joblib"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

SAMPLE_RATE = 16000
WINDOW_SECONDS = 2.0
CALIBRATION_SECONDS = 10.0


st.set_page_config(
    page_title="Vocal Intelligence Agent",
    page_icon="🎤",
    layout="wide",
)

st.title("🎤 VOCAL INTELLIGENCE AGENT")
st.caption("Real-time acoustic analysis with speaker-specific calibration")

st.warning(
    "Research prototype: this system reports acoustic patterns and "
    "speaker-relative deviations. It is not a direct lie detector."
)


@st.cache_resource
def load_artifact():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


artifact = load_artifact()

if artifact is None:
    st.error("Model not found. Run `python train_model.py` first.")
    st.stop()

pipeline = artifact["model"]
classes = artifact["classes"]


# ------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------

defaults = {
    "chunks": [],
    "running": False,
    "start_time": None,
    "baseline": None,
    "latest_features": None,
    "latest_probabilities": None,
    "latest_deviation": 0.0,
    "finished": False,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ------------------------------------------------------------
# AUDIO FUNCTIONS
# ------------------------------------------------------------

def frame_to_mono(frame):
    audio = frame.to_ndarray()

    if audio.ndim == 2:
        audio = np.mean(audio, axis=0)

    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        denom = max(abs(info.min), info.max)
        audio = audio.astype(np.float32) / float(denom)
    else:
        audio = audio.astype(np.float32)

    return audio, int(frame.sample_rate)


def resample_audio(audio, source_sr):
    if source_sr == SAMPLE_RATE:
        return audio

    gcd = np.gcd(source_sr, SAMPLE_RATE)

    up = SAMPLE_RATE // gcd
    down = source_sr // gcd

    return resample_poly(
        audio,
        up,
        down
    ).astype(np.float32)


def predict_window(audio):
    features = extract_features(
        audio,
        sr=SAMPLE_RATE
    )

    probabilities = pipeline.predict_proba(
        features.reshape(1, -1)
    )[0]

    return features, probabilities


# ------------------------------------------------------------
# SPEAKER DEVIATION
# ------------------------------------------------------------

def deviation_score(current, baseline):

    if baseline is None:
        return 0.0

    # Main tracked speaker-relative features:
    # pitch mean, pitch std, pitch range,
    # RMS mean, RMS std, zero-crossing rate

    indices = [0, 1, 2, 3, 4, 5]

    current = np.asarray(
        current,
        dtype=float
    )

    baseline = np.asarray(
        baseline,
        dtype=float
    )

    scale = np.maximum(
        np.abs(baseline[indices]) * 0.10,
        1e-3
    )

    z = np.abs(
        current[indices] - baseline[indices]
    ) / scale

    score = np.mean(z) * 25.0

    return float(
        np.clip(score, 0.0, 100.0)
    )


def evidence_text(current, baseline):

    if baseline is None:
        return ["Waiting for calibration..."]

    notes = []

    if abs(current[0] - baseline[0]) > max(
        abs(baseline[0]) * 0.10,
        2.0
    ):
        notes.append(
            "Pitch level differs from calibration."
        )

    if abs(current[1] - baseline[1]) > max(
        abs(baseline[1]) * 0.20,
        2.0
    ):
        notes.append(
            "Pitch variability differs from calibration."
        )

    if abs(current[3] - baseline[3]) > max(
        abs(baseline[3]) * 0.20,
        0.02
    ):
        notes.append(
            "Vocal energy differs from calibration."
        )

    if abs(current[5] - baseline[5]) > max(
        abs(baseline[5]) * 0.20,
        0.01
    ):
        notes.append(
            "Speech spectral activity differs from calibration."
        )

    if not notes:
        notes.append(
            "No major tracked acoustic deviation "
            "in this window."
        )

    return notes


# ------------------------------------------------------------
# FINAL SESSION ANALYSIS
# ------------------------------------------------------------

def final_analysis(audio):

    if len(audio) < int(
        SAMPLE_RATE * 4
    ):
        return None

    split = int(
        len(audio) * 0.25
    )

    calibration_audio = audio[:split]
    evaluation_audio = audio[split:]

    window = int(
        SAMPLE_RATE * WINDOW_SECONDS
    )

    # -------------------------
    # Build calibration baseline
    # -------------------------

    baseline_vectors = []

    for start in range(
        0,
        len(calibration_audio) - window + 1,
        window
    ):

        chunk = calibration_audio[
            start:start + window
        ]

        try:

            features = extract_features(
                chunk,
                sr=SAMPLE_RATE
            )

            if np.all(
                np.isfinite(features)
            ):
                baseline_vectors.append(
                    features
                )

        except Exception:
            continue

    if not baseline_vectors:
        return None

    baseline = np.mean(
        np.vstack(baseline_vectors),
        axis=0
    )

    # -------------------------
    # Evaluate remaining audio
    # -------------------------

    probabilities = []
    deviations = []

    for start in range(
        0,
        len(evaluation_audio) - window + 1,
        window
    ):

        chunk = evaluation_audio[
            start:start + window
        ]

        try:

            features, probs = predict_window(
                chunk
            )

            if not np.all(
                np.isfinite(features)
            ):
                continue

            probabilities.append(
                probs
            )

            deviations.append(
                deviation_score(
                    features,
                    baseline
                )
            )

        except Exception:
            continue

    average_probabilities = None

    if probabilities:
        average_probabilities = np.mean(
            np.vstack(probabilities),
            axis=0
        )

    average_deviation = 0.0

    if deviations:
        average_deviation = float(
            np.mean(deviations)
        )

    return {
        "baseline": baseline,
        "probabilities": average_probabilities,
        "average_deviation": average_deviation,
    }


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------

with st.sidebar:

    st.header("System")

    st.write("Model: Random Forest")
    st.write("Training data: CREMA-D")
    st.write("Sampling rate: 16 kHz")
    st.write("Live window: 2 seconds")
    st.write("Speaker calibration: 10 seconds")
    st.write("Final calibration: first 25%")

    st.divider()

    st.info(
        "CREMA-D provides supervised acoustic/emotional-state "
        "labels. Live calibration adapts analysis to the "
        "current speaker."
    )


# ------------------------------------------------------------
# DASHBOARD
# ------------------------------------------------------------

status = st.empty()

col1, col2, col3 = st.columns(3)

duration_box = col1.empty()
calibration_box = col2.empty()
deviation_box = col3.empty()

st.divider()

left, right = st.columns(2)

with left:

    st.subheader(
        "Speaker Calibration"
    )

    baseline_box = st.empty()

with right:

    st.subheader(
        "Current Acoustic State"
    )

    probability_box = st.empty()

st.subheader(
    "Acoustic Evidence"
)

evidence_box = st.empty()


# ------------------------------------------------------------
# MICROPHONE
# ------------------------------------------------------------

ctx = webrtc_streamer(
    key="vocal-agent",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=256,
    media_stream_constraints={
        "audio": True,
        "video": False,
    },
)


# ------------------------------------------------------------
# LIVE PROCESSING
# ------------------------------------------------------------

if ctx.state.playing:

    if not st.session_state.running:

        st.session_state.running = True
        st.session_state.finished = False

        st.session_state.chunks = []

        st.session_state.baseline = None

        st.session_state.latest_features = None
        st.session_state.latest_probabilities = None
        st.session_state.latest_deviation = 0.0

        st.session_state.start_time = time.time()

    status.success(
        "🟢 MICROPHONE ACTIVE — listening..."
    )

    rolling = np.array(
        [],
        dtype=np.float32
    )

    last_analysis = 0.0

    while ctx.state.playing:

        try:

            frames = ctx.audio_receiver.get_frames(
                timeout=1
            )

        except Exception:

            time.sleep(0.05)
            continue

        for frame in frames:

            try:

                audio, frame_sr = frame_to_mono(
                    frame
                )

                audio = resample_audio(
                    audio,
                    frame_sr
                )

                if len(audio):

                    st.session_state.chunks.append(
                        audio
                    )

                    rolling = np.concatenate(
                        (
                            rolling,
                            audio
                        )
                    )

                    max_samples = int(
                        SAMPLE_RATE *
                        WINDOW_SECONDS
                    )

                    if len(rolling) > max_samples:

                        rolling = rolling[
                            -max_samples:
                        ]

            except Exception:

                continue

        elapsed = (
            time.time()
            - st.session_state.start_time
        )

        duration_box.metric(
            "Session Duration",
            f"{elapsed:.1f} s"
        )

        # -------------------------
        # CALIBRATION
        # -------------------------

        if elapsed < CALIBRATION_SECONDS:

            remaining = (
                CALIBRATION_SECONDS
                - elapsed
            )

            calibration_box.metric(
                "Speaker Calibration",
                f"{elapsed:.1f} / "
                f"{CALIBRATION_SECONDS:.0f} s"
            )

            status.warning(
                f"🟡 CALIBRATING SPEAKER — "
                f"{remaining:.1f} s remaining"
            )

        # -------------------------
        # LIVE ANALYSIS
        # -------------------------

        elif len(rolling) >= int(
            SAMPLE_RATE * WINDOW_SECONDS
        ):

            # Create speaker baseline once.
            if st.session_state.baseline is None:

                captured = np.concatenate(
                    st.session_state.chunks
                )

                calibration_len = int(
                    SAMPLE_RATE *
                    CALIBRATION_SECONDS
                )

                captured = captured[
                    :calibration_len
                ]

                baseline_vectors = []

                window = int(
                    SAMPLE_RATE *
                    WINDOW_SECONDS
                )

                for start in range(
                    0,
                    len(captured) - window + 1,
                    window
                ):

                    chunk = captured[
                        start:start + window
                    ]

                    try:

                        f = extract_features(
                            chunk,
                            sr=SAMPLE_RATE
                        )

                        if np.all(
                            np.isfinite(f)
                        ):

                            baseline_vectors.append(
                                f
                            )

                    except Exception:

                        continue

                if baseline_vectors:

                    st.session_state.baseline = np.mean(
                        np.vstack(
                            baseline_vectors
                        ),
                        axis=0
                    )

            # Analyze about once per second.
            if time.time() - last_analysis > 1.0:

                try:

                    features, probs = predict_window(
                        rolling
                    )

                    st.session_state.latest_features = (
                        features
                    )

                    st.session_state.latest_probabilities = (
                        probs
                    )

                    st.session_state.latest_deviation = (
                        deviation_score(
                            features,
                            st.session_state.baseline
                        )
                    )

                    last_analysis = time.time()

                    calibration_box.metric(
                        "Calibration Status",
                        "✅ Complete"
                    )

                    deviation_box.metric(
                        "Acoustic Deviation",
                        f"{st.session_state.latest_deviation:.1f} / 100"
                    )

                    # -------------------------
                    # Probabilities
                    # -------------------------

                    order = np.argsort(
                        probs
                    )[::-1]

                    with probability_box.container():

                        for idx in order:

                            label = classes[idx]

                            value = float(
                                probs[idx]
                            )

                            st.write(
                                f"**{label.title()}** — "
                                f"{value * 100:.1f}%"
                            )

                            st.progress(
                                value
                            )

                    # -------------------------
                    # Evidence
                    # -------------------------

                    notes = evidence_text(
                        features,
                        st.session_state.baseline
                    )

                    evidence_box.markdown(
                        "\n\n".join(
                            f"• {note}"
                            for note in notes
                        )
                    )

                    # -------------------------
                    # Baseline table
                    # -------------------------

                    baseline = (
                        st.session_state.baseline
                    )

                    baseline_box.table(
                        {
                            "Feature": [
                                "Pitch mean",
                                "Pitch variability",
                                "Pitch range",
                                "Vocal energy",
                                "Energy variability",
                                "Zero-crossing rate",
                            ],
                            "Baseline": [
                                round(
                                    float(baseline[0]),
                                    4
                                ),
                                round(
                                    float(baseline[1]),
                                    4
                                ),
                                round(
                                    float(baseline[2]),
                                    4
                                ),
                                round(
                                    float(baseline[3]),
                                    4
                                ),
                                round(
                                    float(baseline[4]),
                                    4
                                ),
                                round(
                                    float(baseline[5]),
                                    4
                                ),
                            ],
                        }
                    )

                    status.success(
                        "🟢 ANALYZING — comparing speech "
                        "with speaker baseline"
                    )

                except Exception as exc:

                    status.error(
                        f"Live analysis error: {exc}"
                    )

        time.sleep(0.05)


# ------------------------------------------------------------
# AFTER STOP
# ------------------------------------------------------------

if (
    st.session_state.running
    and not ctx.state.playing
    and not st.session_state.finished
):

    st.session_state.running = False
    st.session_state.finished = True

    if st.session_state.chunks:

        full_audio = np.concatenate(
            st.session_state.chunks
        )

        status.info(
            "⏳ Recording stopped. Running final "
            "25% calibration analysis..."
        )

        result = final_analysis(
            full_audio
        )

        if result is None:

            status.error(
                "Not enough usable audio "
                "for final analysis."
            )

        else:

            status.success(
                "✅ FINAL ANALYSIS COMPLETE"
            )

            total_seconds = (
                len(full_audio)
                / SAMPLE_RATE
            )

            st.divider()

            st.subheader(
                "📊 Final Session Analysis"
            )

            a, b, c = st.columns(3)

            a.metric(
                "Total Session",
                f"{total_seconds:.1f} s"
            )

            b.metric(
                "Calibration",
                f"{total_seconds * 0.25:.1f} s"
            )

            c.metric(
                "Evaluation",
                f"{total_seconds * 0.75:.1f} s"
            )

            st.divider()

            if result["probabilities"] is not None:

                st.markdown(
                    "### Average Acoustic State Distribution"
                )

                probs = result[
                    "probabilities"
                ]

                for idx in np.argsort(
                    probs
                )[::-1]:

                    value = float(
                        probs[idx]
                    )

                    st.write(
                        f"**{classes[idx].title()}** — "
                        f"{value * 100:.1f}%"
                    )

                    st.progress(
                        value
                    )

            st.metric(
                "Average Acoustic Deviation",
                f"{result['average_deviation']:.1f} / 100"
            )

            st.caption(
                "This is a speaker-relative acoustic "
                "deviation measure, not a deception probability."
            )

            summary = {
                "sample_rate": SAMPLE_RATE,
                "window_seconds": WINDOW_SECONDS,
                "total_duration_seconds": total_seconds,
                "calibration_ratio": 0.25,
                "average_acoustic_deviation":
                    result["average_deviation"],
                "model": "Random Forest",
                "training_dataset": "CREMA-D",
            }

            summary_path = (
                OUTPUT_DIR
                / "latest_session_summary.json"
            )

            with open(
                summary_path,
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    summary,
                    file,
                    indent=4
                )

            st.info(
                f"Saved session summary to "
                f"`{summary_path}`"
            )