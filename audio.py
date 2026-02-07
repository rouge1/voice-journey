#!/usr/bin/env python3
"""Offline audio processor for Voice Journey.

Performs speaker diarization and speech transcription using cached models.
Run setup.py first to download models.
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*torch.load.*weights_only.*")

import sys
import os

# Handle --list before argparse (audio_file is required for normal use)
if "--list" in sys.argv:
    from models import list_models
    list_models()
    print("Usage: python audio.py <audio_file> [options]")
    print("Setup: python setup.py")
    sys.exit(0)

import argparse

parser = argparse.ArgumentParser(
    description="Process audio file with speaker diarization and transcription."
)
parser.add_argument("audio_file", help="Path to the audio file to process")
parser.add_argument(
    "--model",
    default="medium",
    choices=["tiny", "small", "medium", "large", "turbo"],
    help="Whisper model size (default: medium)",
)
parser.add_argument(
    "--translate",
    action="store_true",
    help="Translate audio to English (default: transcribe in original language)",
)
args = parser.parse_args()

# Don't force offline mode - let huggingface_hub use cache automatically

# Pre-flight: check models are cached before heavy imports
from models import is_diarization_cached, is_whisper_cached, _get_whisper_model_name

if not is_diarization_cached():
    print("Error: Diarization model not found in cache.")
    print("Run 'python setup.py' to download models first.")
    sys.exit(1)

if not is_whisper_cached(args.model):
    print(f"Error: Whisper {args.model} model not found in cache.")
    print(f"Run 'python setup.py --whisper-sizes {args.model}' to download it.")
    sys.exit(1)

if not os.path.exists(args.audio_file):
    print(f"Error: Audio file '{args.audio_file}' not found!")
    sys.exit(1)

# Don't set HF_HUB_OFFLINE - huggingface_hub will use cached models automatically
# and fall back gracefully if there's no internet connection

# ==================== SETUP ====================
import torch
import torchaudio
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel
from datetime import timedelta

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

# Load diarization pipeline
print("\nLoading pyannote speaker diarization pipeline...")
diarizer = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarizer.to(torch.device("cuda"))

# Load Whisper transcription model
device = "cpu"
compute_type = "int8"
actual_model = _get_whisper_model_name(args.model)
print(f"Loading Whisper {args.model} transcription model on {device}...")
transcriber = WhisperModel(actual_model, device=device, compute_type=compute_type)

# ==================== PROCESS AUDIO ====================

print(f"\nLoading audio: {args.audio_file}")

print("Running diarization...")
diarization = diarizer(args.audio_file)

# Get audio duration from diarization results
timeline_extent = diarization.speaker_diarization.get_timeline().extent()
duration_seconds = timeline_extent.duration
duration = str(timedelta(seconds=int(duration_seconds)))
print(f"Audio duration: {duration} ({duration_seconds:.2f} seconds)")

# Clear GPU memory after diarization
torch.cuda.empty_cache()

# Build speaker map for timeline and alignment
speaker_map = {}
for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
    for t in range(int(turn.start * 10), int(turn.end * 10)):
        speaker_map[round(t / 10, 2)] = speaker

# Create timeline visualization
print("\nGenerating timeline visualization...")
timeline = []
for t in range(0, int(duration_seconds)):
    has_speaker = any(
        speaker_map.get(round(sub_t / 10, 2), None)
        for sub_t in range(t * 10, (t + 1) * 10)
        if speaker_map.get(round(sub_t / 10, 2), None) not in [None, "UNKNOWN"]
    )
    timeline.append("x" if has_speaker else "_")
timeline_str = "".join(timeline)
print(f"Timeline (1 char = 1 sec): {timeline_str}")

print("Running transcription...")
try:
    task = "translate" if args.translate else "transcribe"
    segments, info = transcriber.transcribe(
        args.audio_file, beam_size=1, vad_filter=True, task=task
    )
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print(
            "GPU out of memory! Try a smaller model (--model tiny or small) or set device='cpu'"
        )
    raise

# Clear GPU memory after transcription
torch.cuda.empty_cache()

# ==================== ALIGN & PRINT ====================

print("\n" + "=" * 60)
print("SPEAKER-LABELED TRANSCRIPT")
print("=" * 60)

current_speaker = None
for seg in segments:
    start = seg.start
    end = seg.end
    text = seg.text.strip()

    # Find dominant speaker in this segment
    times = range(int(start * 10), int(end * 10))
    speakers_in_seg = [speaker_map.get(round(t / 10, 2), "UNKNOWN") for t in times]
    seg_speaker = (
        max(set(speakers_in_seg), key=speakers_in_seg.count)
        if speakers_in_seg
        else "UNKNOWN"
    )

    if seg_speaker != current_speaker:
        timestamp = str(timedelta(seconds=int(start)))
        print(f"\n[{timestamp}] {seg_speaker}:")
        current_speaker = seg_speaker

    start_time = str(timedelta(seconds=int(start)))
    end_time = str(timedelta(seconds=int(end)))
    print(f"[{start_time} - {end_time}] {text}")

if args.translate:
    print(f"\nTranslated from: {info.language} → en")
else:
    print(f"\nTranscription language: {info.language}")

# Clean up
del diarizer
del transcriber
torch.cuda.empty_cache()

print("\nProcessing complete!")
