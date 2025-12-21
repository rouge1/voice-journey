import warnings
# Suppress warnings before imports
warnings.filterwarnings("ignore", category=UserWarning)

import sys
import os
import torch
import torchaudio
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel
from datetime import timedelta

# Enable TF32 for better performance (slight precision trade-off)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ==================== CONFIGURATION ====================
# Audio file to process (pass as command-line argument or set default)
if len(sys.argv) > 1:
    audio_file = sys.argv[1]
else:
    audio_file = "audio.wav"  # Default audio file

# Model size (optional second argument: tiny, small, medium, large-v3)
if len(sys.argv) > 2:
    model_size = sys.argv[2]
    valid_sizes = ["tiny", "small", "medium", "large-v3"]
    if model_size not in valid_sizes:
        print(f"Error: Invalid model size '{model_size}'. Choose from: {', '.join(valid_sizes)}")
        sys.exit(1)
else:
    model_size = "medium"  # Default model size

# Validate audio file exists
if not os.path.exists(audio_file):
    print(f"Error: Audio file '{audio_file}' not found!")
    print("Usage: python audio.py <path_to_audio_file> [model_size]")
    print("Model sizes: tiny, small, medium, large-v3")
    sys.exit(1)

# ==================== SETUP ====================
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

# Load diarization pipeline (from cache or prompt for token)
print("\nLoading pyannote speaker diarization pipeline...")
try:
    diarizer = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
except Exception as e:
    if "401" in str(e) or "403" in str(e) or "gated" in str(e).lower():
        print("\n" + "="*60)
        print("Model not cached. Hugging Face token required for first download.")
        print("Get your token at: https://hf.co/settings/tokens")
        print("Make sure you've accepted the model terms at:")
        print("  https://hf.co/pyannote/speaker-diarization-3.1")
        print("="*60)
        hf_token = input("\nEnter your Hugging Face token: ").strip()
        if not hf_token:
            print("Error: Token is required for first-time download.")
            sys.exit(1)
        print("Downloading model (this only needs to happen once)...")
        diarizer = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=hf_token)
        print("Model cached successfully! Token won't be needed again.")
    else:
        raise e
diarizer.to(torch.device("cuda"))

# Load Whisper transcription model
# Options: "large-v3" (best accuracy, high VRAM), "medium", "small", "tiny" (fastest, least accurate)
device = "cpu"  # Use CPU for transcription to avoid cuDNN issues
compute_type = "int8"  # Use int8 for CPU
print(f"Loading Whisper {model_size} transcription model on {device}...")
transcriber = WhisperModel(model_size, device=device, compute_type=compute_type)

# ==================== PROCESS AUDIO ====================

print(f"\nLoading audio: {audio_file}")
waveform, sample_rate = torchaudio.load(audio_file)

# Convert to mono if needed
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

audio_in_memory = {"waveform": waveform, "sample_rate": sample_rate}

print("Running diarization...")
diarization = diarizer(audio_in_memory)

print("Running transcription...")
try:
    segments, info = transcriber.transcribe(audio_file, beam_size=1, vad_filter=True, language="en")  # beam_size=1 saves memory
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print("GPU out of memory! Try using a smaller model (change model_size to 'small' or 'tiny') or set device='cpu'")
        raise e
    else:
        raise e

# ==================== ALIGN & PRINT ====================
# Build a speaker map per time segment
speaker_map = {}
for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
    for t in range(int(turn.start * 10), int(turn.end * 10)):  # 100ms steps
        speaker_map[round(t / 10, 2)] = speaker

print("\n" + "="*60)
print("SPEAKER-LABELED TRANSCRIPT")
print("="*60)

current_speaker = None
for seg in segments:
    start = seg.start
    end = seg.end
    text = seg.text.strip()

    # Find dominant speaker in this segment
    times = range(int(start * 10), int(end * 10))
    speakers_in_seg = [speaker_map.get(round(t / 10, 2), "UNKNOWN") for t in times]
    seg_speaker = max(set(speakers_in_seg), key=speakers_in_seg.count) if speakers_in_seg else "UNKNOWN"

    if seg_speaker != current_speaker:
        timestamp = str(timedelta(seconds=int(start)))
        print(f"\n[{timestamp}] {seg_speaker}:")
        current_speaker = seg_speaker

    start_time = str(timedelta(seconds=int(start)))
    end_time = str(timedelta(seconds=int(end)))
    print(f"[{start_time} - {end_time}] {text}")

print("\nTranscription language:", info.language)
print("Processing complete! 🚀")