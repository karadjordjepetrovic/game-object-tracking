"""Track objects in a clip and time each stage of the pipeline.

Writes, into outputs/<clip name>/:
  <tag>.txt          tracks in MOT format (one line per box per frame)
  <tag>_timing.json  average milliseconds per stage and FPS
  <tag>.mp4          video with boxes drawn (only with --video)

Examples:
  python src/track.py data/clips/clip01.mp4 --model yolo11x.pt --tag reference --video
  python src/track.py data/clips/clip01.mp4 --model yolo11n.pt --tag nano
"""
import argparse
import json
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument('clip', help='path to the video clip')
parser.add_argument('--model', default='yolo11n.pt', help='model weights (.pt) or TensorRT engine (.engine)')
parser.add_argument('--tracker', default='bytetrack.yaml', help='tracker config (.yaml)')
parser.add_argument('--tag', help='name for the output files (default: the model file name)')
parser.add_argument('--classes', type=int, nargs='*', default=[0],
                    help='class ids to keep (0 = person). Use --classes with no numbers to keep all')
parser.add_argument('--imgsz', type=int, default=640, help='model input size in pixels')
parser.add_argument('--warmup', type=int, default=20, help='first N frames are left out of the timing')
parser.add_argument('--video', action='store_true', help='also write a video with boxes drawn (slow)')
args = parser.parse_args()

clip = Path(args.clip)
tag = args.tag or Path(args.model).stem
out_dir = Path('outputs') / clip.stem
out_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(clip))
if not cap.isOpened():
    raise SystemExit(f'Could not open {clip}')

model = YOLO(args.model)

writer = None
if args.video:
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = cv2.VideoWriter(str(out_dir / f'{tag}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

stages = ['decode', 'preprocess', 'inference', 'postprocess', 'tracker']
totals = dict.fromkeys(stages, 0.0)
timed_frames = 0
frame_idx = 0

with open(out_dir / f'{tag}.txt', 'w') as track_file:
    while True:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        t1 = time.perf_counter()
        if not ok:
            break
        frame_idx += 1  # MOT format counts frames from 1

        result = model.track(frame, persist=True, tracker=args.tracker, classes=args.classes or None,
                             imgsz=args.imgsz, verbose=False)[0]
        t2 = time.perf_counter()

        boxes = result.boxes
        if boxes.id is not None:
            for (x1, y1, x2, y2), tid, conf in zip(boxes.xyxy.tolist(),
                                                   boxes.id.int().tolist(),
                                                   boxes.conf.tolist()):
                # frame, id, left, top, width, height, confidence, -1, -1, -1
                track_file.write(f'{frame_idx},{tid},{x1:.1f},{y1:.1f},'
                                 f'{x2 - x1:.1f},{y2 - y1:.1f},{conf:.3f},-1,-1,-1\n')

        # Drawing and writing video happen after t2, so they are not part of the timing.
        if writer:
            writer.write(result.plot())

        if frame_idx > args.warmup:
            speed = result.speed  # milliseconds, measured inside Ultralytics
            model_ms = speed['preprocess'] + speed['inference'] + speed['postprocess']
            totals['decode'] += (t1 - t0) * 1000
            totals['preprocess'] += speed['preprocess']
            totals['inference'] += speed['inference']
            totals['postprocess'] += speed['postprocess']
            totals['tracker'] += max((t2 - t1) * 1000 - model_ms, 0.0)  # tracker + small overheads
            timed_frames += 1

cap.release()
if writer:
    writer.release()

if frame_idx == 0:
    raise SystemExit('No frames were read - check the clip with ffprobe')
if timed_frames == 0:
    raise SystemExit(f'Clip has only {frame_idx} frames, fewer than --warmup {args.warmup}')

avg = {stage: totals[stage] / timed_frames for stage in stages}
pipeline_ms = sum(avg.values())
fps = 1000 / pipeline_ms

print(f'\n{tag}: {frame_idx} frames ({timed_frames} timed, first {args.warmup} skipped)')
for stage in stages:
    print(f'  {stage:>11}: {avg[stage]:7.2f} ms/frame')
print(f'  {"total":>11}: {pipeline_ms:7.2f} ms/frame = {fps:.1f} FPS')

timing = {
    'clip': clip.stem, 'tag': tag, 'model': args.model, 'tracker': args.tracker,
    'imgsz': args.imgsz, 'frames': frame_idx, 'timed_frames': timed_frames,
    'ms_per_frame': {k: round(v, 3) for k, v in avg.items()},
    'total_ms': round(pipeline_ms, 3), 'fps': round(fps, 2),
}
(out_dir / f'{tag}_timing.json').write_text(json.dumps(timing, indent=2))
print(f'\nSaved {out_dir / (tag + ".txt")} and {tag}_timing.json')
