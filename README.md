# Game Object Tracking

Multi-object tracking on gameplay recordings: detect characters in every frame, keep a stable ID on each one over time, and make the pipeline fast on the GPU without losing accuracy.

Follow-up to [realtime-depth-optimization](https://github.com/karadjordjepetrovic/realtime-depth-optimization) and game-text-recognition (HUD mask).

> **Status:** work in progress. Baseline, clean timing and reference comparison done; next step is TensorRT.

<!-- Demo GIF goes here: reference vs optimized tracker, side by side. -->

## Approach

Tracking-by-detection: a YOLO detector finds character boxes in each frame, and ByteTrack links them into tracks using motion prediction and box matching.

**How accuracy is measured.** Instead of hand-labelled ground truth, every run is compared with a **reference run**: the slowest, most careful configuration (YOLO11x at 1280 px input, detector on every frame). The scores say how closely a faster configuration agrees with that reference, which is the question that matters when optimizing: *does the fast version give the same answer as the careful one?*

- **Precision / recall / F1:** do the run's boxes match the reference's boxes (IoU ≥ 0.5)?
- **IDF1 / ID switches:** do the track IDs stay consistent with the reference?

The reference is not ground truth: mistakes it makes itself (e.g. boxes on the player's own weapon) are not penalised in faster runs.

## Results

Test clip: 41 s of Call of Duty: Modern Warfare 2 campaign gameplay, 1080p, 2,458 frames at a constant 60 fps. Timing excludes the first 20 frames (CUDA warm-up) and video writing.

| Run | Model | Input | FPS | Precision | Recall | F1 | IDF1 | ID switches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reference | YOLO11x | 1280 | 15.8 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| nano | YOLO11n | 640 | 48.3 | 0.792 | 0.397 | 0.529 | 0.425 | 64 |
| nano_1280 | YOLO11n | 1280 | 50.0 | 0.862 | 0.593 | 0.703 | 0.625 | 54 |
| small | YOLO11s | 640 | 55.5 | 0.865 | 0.603 | 0.711 | 0.599 | 69 |
| **medium** | **YOLO11m** | **640** | **47.3** | **0.936** | **0.716** | **0.811** | **0.734** | **66** |

### Per-stage timing (ms per frame)

| Run | Decode | Preprocess | Inference | Postprocess | Tracker | Total |
| --- | --- | --- | --- | --- | --- | --- |
| reference | 3.29 | 4.79 | 51.66 | 1.85 | 1.79 | 63.38 |
| nano | 3.26 | 1.75 | 12.30 | 1.70 | 1.69 | 20.70 |
| nano_1280 | 2.99 | 3.59 | 10.36 | 1.60 | 1.46 | 19.99 |
| small | 2.97 | 1.05 | 10.86 | 1.64 | 1.50 | 18.02 |
| medium | 2.99 | 1.09 | 13.85 | 1.67 | 1.54 | 21.14 |

### Observations

- **Small models are not GPU-bound.** Going from YOLO11n to YOLO11m roughly triples the network size but adds only a few milliseconds: inference stays at 10–14 ms, and decode, pre/postprocessing and the tracker add another ~7 ms on the CPU regardless of model. Picking a smaller model barely buys speed; the overhead around the model does.
- **Input resolution matters for distant characters.** YOLO11n gains 0.17 F1 going from 640 to 1280 px input, so many of the misses are small characters that disappear when the 1080p frame is shrunk.
- **YOLO11m at 640 is the configuration to optimize:** the best agreement with the reference (F1 0.81) at nearly the speed of the smallest model.
- **Timings vary by about ±1–2 ms between runs** (e.g. nano at 1280 measured faster than at 640), so differences that small are within noise. Repeat runs before drawing conclusions from them.
- **ID switches stay high (54–69) in every run,** pointing at the tracker settings and camera motion rather than the detector.

## Next steps

1. TensorRT FP16 and INT8 for YOLO11m, scored against the reference
2. Nsight Systems profiling of the CPU overhead around the model
3. GPU video decode (NVDEC) and GPU preprocessing
4. Detector-skip experiment: FPS vs agreement curve
5. Optional: HUD masking and camera motion compensation to reduce ID switches

## Setup

Tested on Windows, NVIDIA GeForce RTX 3070 (driver 576.52), Python 3.12, PyTorch with CUDA 12.8.

```powershell
git clone https://github.com/karadjordjepetrovic/game-object-tracking
cd game-object-tracking
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install ultralytics lap scipy opencv-python numpy pandas matplotlib
```

Check the GPU is visible:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Recording clips

Clips are recorded with Xbox Game Bar (1080p, 60 fps, high quality) and are not committed.
Game Bar records with a variable frame rate, so every clip is converted to a constant 60 fps when it is cut:

```powershell
ffmpeg -ss 00:00:00 -i data\clips\mw2_raw.mp4 -t 45 -r 60 -fps_mode cfr -c:v libx264 -crf 18 -an data\clips\clip01.mp4
```

A constant frame rate keeps frame numbers consistent between runs.

## Usage

Track a clip and time each stage (writes into `outputs/<clip>/`):

```powershell
# reference: slow and careful, with a video of the result
python src/track.py data/clips/clip01.mp4 --model yolo11x.pt --imgsz 1280 --tag reference --video

# a faster configuration
python src/track.py data/clips/clip01.mp4 --model yolo11m.pt --tag medium
```

Compare a run with the reference (also appends a row to `outputs/<clip>/results.csv`):

```powershell
python src/compare.py outputs/clip01/reference.txt outputs/clip01/medium.txt
```

## Repository layout

```
data/clips/      gameplay clips (not committed)
data/gt/         labelling rules (hand-labelled ground truth, if added later)
configs/         tracker settings
src/             track.py (tracking + timing), compare.py (scoring vs reference)
outputs/         track files, timing, result videos, results.csv (not committed)
```
