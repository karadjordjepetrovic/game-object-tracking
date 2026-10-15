# Game Object Tracking

Multi-object tracking on gameplay recordings: detect characters and other objects in every frame, keep a stable ID on each one over time, and make it measurably accurate and fast on the GPU.

Follow-up to [realtime-depth-optimization](https://github.com/karadjordjepetrovic/realtime-depth-optimization) and game-text-recognition (HUD mask).

> **Status:** work in progress. Phase 0 (setup) done.

<!-- Demo GIF goes here: baseline vs final tracker, side by side, on a fast camera turn. -->

## Problem

Gameplay footage breaks assumptions that generic trackers make:

- **Camera motion:** fast mouse turns move every object at once, so motion prediction fails and IDs are lost.
- **HUD:** minimap icons, kill-feed portraits and markers produce false detections that never move.
- **Visual noise:** muzzle flash, explosions and motion blur cause missed or flickering detections.
- **Domain gap:** detectors trained on real photos don't know the game's art style or object classes.

<!-- Replace with your own observations from the Phase 1 baseline, with timestamps. -->

## Approach

Tracking-by-detection: a detector finds boxes in each frame, and a tracker (ByteTrack / BoT-SORT) links them into tracks using motion prediction and box matching.

Game-specific additions:

1. HUD masking using the mask from game-text-recognition
2. Camera motion compensation
3. A detector fine-tuned on auto-labelled, hand-corrected game frames
4. TensorRT inference and a GPU-side pipeline

## Results

Scored on a hand-labelled test clip with [TrackEval](https://github.com/JonathonLuiten/TrackEval). The test clip is never used for training.

| Config | HOTA | IDF1 | IDSW | FPS |
| --- | --- | --- | --- | --- |
| Baseline (YOLO + ByteTrack) | | | | |
| + HUD mask | | | | |
| + camera motion compensation | | | | |
| + fine-tuned detector | | | | |

## Performance

| Detector | Precision | Detect ms | Total FPS | HOTA |
| --- | --- | --- | --- | --- |
| PyTorch | FP32 | | | |
| TensorRT | FP16 | | | |
| TensorRT | INT8 | | | |

<!-- Add: per-stage timing table, Nsight Systems before/after screenshot, HOTA-vs-FPS curve for the detector-skip experiment. -->

## Limitations

<!-- Honest notes, e.g. masking can hide characters under the crosshair; one test clip is a small sample. -->

## Setup

Tested on Windows, NVIDIA GeForce RTX 3070 (driver 576.52), Python 3.12, PyTorch with CUDA 12.8.

```powershell
git clone https://github.com/karadjordjepetrovic/game-object-tracking
cd game-object-tracking
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install ultralytics opencv-python numpy pandas matplotlib
```

Check the GPU is visible:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Usage

```powershell
python src/track_baseline.py data/clips/clip01.mp4
```

Outputs a tracked video and a MOT-format track file in `outputs/`.

## Repository layout

```
data/clips/      gameplay clips (not committed)
data/gt/         hand-labelled ground truth
data/dataset/    training images and labels (not committed)
configs/         tracker settings
src/             scripts
outputs/         result videos, track files, scores (not committed)
```

## Recording clips

Clips are recorded with Xbox Game Bar (1080p, 60 fps, high quality) and are not committed.
Game Bar records with a variable frame rate, so every clip is converted to a constant 60 fps when it is cut:

```powershell
ffmpeg -ss 00:01:00 -i data\clips\mw2_raw.mp4 -t 45 -r 60 -fps_mode cfr -c:v libx264 -crf 18 -an data\clips\clip01.mp4
```

A constant frame rate keeps frame numbers consistent between ground truth, tracker output and evaluation.