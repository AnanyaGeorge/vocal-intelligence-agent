# Multimodal Sensor Fusion for Cognitive Load and Stress Detection

> **Status: In progress.** The current milestone is a working voice-processing prototype with a Streamlit interface. RGB-video processing, thermal processing, and multimodal fusion have not yet been implemented.

## Overview

This final-year project explores a multimodal approach to cognitive-load and stress detection using three intended input modalities:

- **Audio / voice:** the currently implemented component.
- **RGB video:** planned.
- **Thermal data:** planned.

The intended system will process each modality and combine relevant outputs through a fusion layer. The complete multimodal system and its performance evaluation remain future work.

## Current Progress

| Component | Status | Notes |
|---|---|---|
| Voice-processing module | Implemented to current milestone | A working prototype is available through a Streamlit interface. |
| Streamlit interface | Demonstrated | The current interface supports the voice prototype. |
| RGB-video module | Not yet implemented | Planned development. |
| Thermal-data module | Not yet implemented | Planned development. |
| Multimodal fusion | Planned | To be developed after the modality-specific components. |
| Integrated evaluation | Pending | Report results only after experiments have been run. |

## Repository Structure

```text
vocal-intelligence-agent/
├── app.py
├── feature_engine.py
├── train_model.py
├── requirements.txt
├── data/       # Keep datasets out of Git unless sharing is explicitly permitted
├── models/     # Check model size, licensing, and sensitivity before publishing
└── outputs/    # Check for recordings, predictions, logs, or other sensitive files
```

## Getting Started

### 1. Create and activate a virtual environment

Use Python 3.12 if that matches the environment in which the current prototype was tested.

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```powershell
streamlit run app.py
```

These instructions assume the dependencies in `requirements.txt` are complete and that any required model assets are available locally. Update this section if your actual setup differs.

## Responsible Use and Limitations

- This is an academic prototype under development, not a validated diagnostic or operational tool.
- Predictions may be affected by recording quality, language, accent, environment, dataset composition, and model limitations.
- Do not use outputs to make high-stakes decisions about individuals.
- Do not publish private recordings, personal information, credentials, or datasets unless you have the necessary permissions.
- The repository describes the current implementation honestly; planned components should not be interpreted as completed or evaluated.

## Planned Work

1. Continue evaluating and documenting the voice module.
2. Implement the RGB-video processing component.
3. Implement the thermal-data processing component.
4. Design and integrate the multimodal fusion layer.
5. Evaluate the integrated system using appropriate metrics and clearly documented experimental conditions.

## Acknowledgements

Add the names of project contributors, supervisor, and permitted dataset/model citations here before publishing, if appropriate.

## License

No license has been selected yet. Unless a license is added, others should not assume they have permission to reuse this code. Confirm your team's and institution's requirements before choosing a license.
