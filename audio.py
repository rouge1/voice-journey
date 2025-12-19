import torch
import torchaudio
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel
from datetime import timedelta
import warnings

# Enable TF32 for better performance (slight precision trade-off)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Suppress the TF32 warning from pyannote.audio
warnings.filterwarnings("ignore", message="TensorFloat-32.*has been disabled", category=UserWarning)

# ==================== SETUP ====================
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

# Load diarization pipeline (from cache)
print("\nLoading pyannote speaker diarization pipeline...")
diarizer = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarizer.to(torch.device("cuda"))

# Load Whisper transcription model
# Options: "large-v3" (best accuracy, high VRAM), "medium", "small", "tiny" (fastest, least accurate)
model_size = "medium"  # Change to "large-v3" for best accuracy if you have enough VRAM
device = "cuda"  # Change to "cpu" if GPU runs out of memory
print(f"Loading Whisper {model_size} transcription model on {device}...")
transcriber = WhisperModel(model_size, device=device, compute_type="float16")

# ==================== PROCESS AUDIO ====================
audio_file = "meet_kevin1.wav"  # ← Change to your actual file

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
    for t in torch.arange(turn.start, turn.end, 0.1):  # 100ms steps
        speaker_map[round(t.item(), 2)] = speaker

print("\n" + "="*60)
print("SPEAKER-LABELED TRANSCRIPT")
print("="*60)

current_speaker = None
for seg in segments:
    start = seg.start
    end = seg.end
    text = seg.text.strip()

    # Find dominant speaker in this segment
    times = torch.arange(start, end, 0.1)
    speakers_in_seg = [speaker_map.get(round(t.item(), 2), "UNKNOWN") for t in times]
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