import warnings
# Suppress warnings before imports
warnings.filterwarnings("ignore", category=UserWarning)

import sys
import os
import argparse
import socket

# Check network connectivity when --update is used
def check_network_connectivity():
    """Quick check if we can reach the internet."""
    try:
        # Try to connect to Google DNS with short timeout
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False

# Parse command-line arguments early to set offline mode before importing models
parser = argparse.ArgumentParser(description='Process audio file with speaker diarization and transcription.')
parser.add_argument('audio_file', nargs='?', help='Path to the audio file to process (optional with --update)')
parser.add_argument('--model_size', default='medium', choices=['tiny', 'small', 'medium', 'large-v3'], 
                    help='Whisper model size (default: medium)')
parser.add_argument('--update', action='store_true', 
                    help='Allow online updates for models (checks connectivity first)')
parser.add_argument('--list', action='store_true', 
                    help='List available models and their cache status')
parser.add_argument('--no-speaker-analysis', action='store_true',
                    help='Disable Voxtral speaker characteristics analysis')

args = parser.parse_args()

# Check network connectivity if --update is requested
if args.update and not check_network_connectivity():
    print("\n" + "="*50)
    print("No internet connection detected.")
    print("Cannot check for model updates.")
    if not args.audio_file:
        print("Exiting - no audio file to process.")
        sys.exit(0)
    print("Using cached models only.")
    print("Connect to internet and try again to check for updates.")
    print("="*50)
    # Force offline mode
    os.environ["HF_HUB_OFFLINE"] = "1"

# Handle --list (exit early)
if args.list:
    import os
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    
    print("Available models and cache status:")
    print()
    
    # Check pyannote model
    pyannote_path = os.path.join(cache_dir, "models--pyannote--speaker-diarization-3.1")
    pyannote_status = "✅ Cached" if os.path.exists(pyannote_path) else "❌ Not cached"
    print(f"Speaker Diarization: {pyannote_status}")
    print("  pyannote/speaker-diarization-3.1")
    print()
    
    # Check Voxtral model
    voxtral_path = os.path.join(cache_dir, "models--mistralai--Voxtral-Mini-3B-2507")
    voxtral_status = "✅ Cached" if os.path.exists(voxtral_path) else "❌ Not cached"
    print(f"Speaker Analysis: {voxtral_status}")
    print("  mistralai/Voxtral-Mini-3B-2507")
    print()
    
    # Check Whisper models
    print("Whisper Transcription Models:")
    whisper_sizes = ['tiny', 'small', 'medium', 'large-v3']
    size_info = {
        'tiny': 'Fastest, least accurate (~39 MB)',
        'small': 'Balanced speed/accuracy (~100 MB)', 
        'medium': 'Default, good quality (~484 MB)',
        'large-v3': 'Best accuracy, high memory (~1.5 GB)'
    }
    
    for size in whisper_sizes:
        model_path = os.path.join(cache_dir, f"models--Systran--faster-whisper-{size}")
        status = "✅ Cached" if os.path.exists(model_path) else "❌ Not cached"
        print(f"  {size:<8} - {size_info[size]:<35} {status}")
    
    print()
    print("Usage: python audio.py <audio_file> [options]")
    print("Options:")
    print("  --model_size <size>    Whisper model size (tiny/small/medium/large-v3)")
    print("  --no-speaker-analysis  Skip Voxtral speaker characteristics analysis")
    print("  --update              Download missing models")
    sys.exit(0)

# Set offline mode by default, unless --update is specified OR Voxtral analysis is enabled
if not args.update and args.no_speaker_analysis:
    os.environ["HF_HUB_OFFLINE"] = "1"  # Force offline mode to use cached models

import torch
import torchaudio
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel
from datetime import timedelta

# Voxtral imports for speaker analysis (default enabled)
from transformers import AutoProcessor, VoxtralForConditionalGeneration
import soundfile as sf
import tempfile

# Set variables from parsed args
audio_file = args.audio_file
model_size = args.model_size

# Handle update-only mode
if args.update and not audio_file:
    print("Checking for model updates...")
elif not audio_file:
    print("Error: Audio file required. Usage: python audio.py <audio_file> [--model_size SIZE] [--update]")
    sys.exit(1)
elif not os.path.exists(audio_file):
    print(f"Error: Audio file '{audio_file}' not found!")
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
    elif args.update and ("Temporary failure in name resolution" in str(e) or 
                          "Max retries exceeded" in str(e) or 
                          "Connection refused" in str(e) or
                          "Name resolution failure" in str(e) or
                          "MaxRetryError" in str(type(e))):
        print("\n" + "="*50)
        print("Network connection error while checking for updates.")
        print("This is expected if you're offline.")
        print("Using cached models - no updates downloaded.")
        print("Run with internet connection to check for updates.")
        print("="*50)
        # Try to load from cache
        try:
            diarizer = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
        except Exception as cache_e:
            print(f"Error loading cached model: {cache_e}")
            sys.exit(1)
    else:
        raise e
diarizer.to(torch.device("cuda"))

# Load Whisper transcription model
# Options: "large-v3" (best accuracy, high VRAM), "medium", "small", "tiny" (fastest, least accurate)
device = "cpu"  # Use CPU for transcription to avoid cuDNN issues
compute_type = "int8"  # Use int8 for CPU
print(f"Loading Whisper {model_size} transcription model on {device}...")
try:
    transcriber = WhisperModel(model_size, device=device, compute_type=compute_type)
except Exception as e:
    if args.update and ("Temporary failure in name resolution" in str(e) or 
                        "Max retries exceeded" in str(e) or 
                        "Connection refused" in str(e) or
                        "Name resolution failure" in str(e) or
                        "MaxRetryError" in str(type(e))):
        print("\n" + "="*50)
        print("Network connection error while checking for Whisper updates.")
        print("This is expected if you're offline.")
        print("Using cached model - no updates downloaded.")
        print("Run with internet connection to check for updates.")
        print("="*50)
        # Try to load from cache - WhisperModel should handle this automatically
        transcriber = WhisperModel(model_size, device=device, compute_type=compute_type)
    else:
        raise e

# Load Voxtral model for speaker analysis (default enabled)
voxtral_processor = None
voxtral_model = None
if not args.no_speaker_analysis:
    print("\nLoading Voxtral model for speaker analysis...")
    model_id = "mistralai/Voxtral-Mini-3B-2507"
    
    try:
        voxtral_processor = AutoProcessor.from_pretrained(model_id)
        voxtral_model = VoxtralForConditionalGeneration.from_pretrained(
            model_id,
            dtype=torch.float16,
            device_map="auto"
        )
        print("✓ Voxtral model loaded")
    except Exception as e:
        if args.update and ("Temporary failure in name resolution" in str(e) or 
                            "Max retries exceeded" in str(e) or 
                            "Connection refused" in str(e) or
                            "OfflineModeIsEnabled" in str(type(e).__name__)):
            print("\n" + "="*50)
            print("Network connection error or offline mode enabled.")
            print("Voxtral model not cached. Run with internet and --update to download.")
            print("Continuing without speaker analysis...")
            print("="*50)
            voxtral_processor = None
            voxtral_model = None
        else:
            raise e

def analyze_speaker_segment(audio_file_path, voxtral_processor, voxtral_model):
    """Analyze a speaker segment for gender, age, and emotional state."""
    try:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "path": audio_file_path},
                    {"type": "text", "text": "Briefly estimate the speaker's gender, approximate age, and emotional state. Be concise."}
                ]
            }
        ]
        
        inputs = voxtral_processor.apply_chat_template(conversation, return_tensors="pt", return_dict=True)
        inputs = {k: v.to(voxtral_model.device) for k, v in inputs.items()}
        
        outputs = voxtral_model.generate(**inputs, max_new_tokens=256)
        response = voxtral_processor.decode(outputs[0], skip_special_tokens=True)
        return response
    except Exception as e:
        return f"Error analyzing segment: {str(e)}"

# If update-only mode, exit after loading models
if args.update and not audio_file:
    print("Model update check complete! All models are loaded and ready.")
    sys.exit(0)

# ==================== PROCESS AUDIO ====================

print(f"\nLoading audio: {audio_file}")
waveform, sample_rate = torchaudio.load(audio_file)

# Convert to mono if needed
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

audio_in_memory = {"waveform": waveform, "sample_rate": sample_rate}

# Calculate and display audio duration
duration_seconds = waveform.shape[1] / sample_rate
duration = str(timedelta(seconds=int(duration_seconds)))
print(f"Audio duration: {duration}")

print("Running diarization...")
diarization = diarizer(audio_in_memory)

# Clear GPU memory after diarization
torch.cuda.empty_cache()

# Note: Keeping diarizer in memory as its result is used later for speaker analysis

# Build speaker map for timeline and alignment
speaker_map = {}
for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
    for t in range(int(turn.start * 10), int(turn.end * 10)):  # 100ms steps
        speaker_map[round(t / 10, 2)] = speaker

# Create timeline visualization
print("\nGenerating timeline visualization...")
timeline = []
for t in range(0, int(duration_seconds)):
    has_speaker = any(speaker_map.get(round(sub_t / 10, 2), None) 
                     for sub_t in range(t*10, (t+1)*10) 
                     if speaker_map.get(round(sub_t / 10, 2), None) not in [None, "UNKNOWN"])
    timeline.append("x" if has_speaker else "_")
timeline_str = "".join(timeline)
print(f"Timeline (1 char = 1 sec): {timeline_str}")

print("Running transcription...")
try:
    segments, info = transcriber.transcribe(audio_file, beam_size=1, vad_filter=True, language="en")  # beam_size=1 saves memory
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print("GPU out of memory! Try using a smaller model (change model_size to 'small' or 'tiny') or set device='cpu'")
        raise e
    else:
        raise e

# Clear GPU memory after transcription (Whisper uses CPU, but good practice)
torch.cuda.empty_cache()

# ==================== ALIGN & PRINT ====================
# Speaker map already built above

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

# ==================== SPEAKER ANALYSIS ====================
if voxtral_processor and voxtral_model:
    print("\n" + "="*60)
    print("SPEAKER CHARACTERISTICS ANALYSIS")
    print("="*60)
    
    # Get unique speakers and their segments
    speakers = {}
    for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
        if speaker not in speakers:
            speakers[speaker] = []
        speakers[speaker].append((turn.start, turn.end))
    
    # Analyze each speaker
    for speaker, turns in speakers.items():
        print(f"\n{speaker}:")
        
        # Use the first substantial segment (at least 3 seconds) or the longest one
        suitable_segments = [(s, e) for s, e in turns if e - s >= 3.0]
        if not suitable_segments:
            suitable_segments = turns
        
        start, end = max(suitable_segments, key=lambda x: x[1] - x[0])
        
        # Extract audio segment
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        segment_waveform = waveform[:, start_sample:end_sample]
        
        # Save temporary file for analysis
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            torchaudio.save(tmp_path, segment_waveform, sample_rate)
        
        try:
            print(f"  Analyzing segment from {timedelta(seconds=int(start))} to {timedelta(seconds=int(end))}...")
            analysis = analyze_speaker_segment(tmp_path, voxtral_processor, voxtral_model)
            print(f"  Analysis: {analysis}")
        finally:
            # Clean up temp file
            os.remove(tmp_path)

# Clear GPU memory after speaker analysis
torch.cuda.empty_cache()

# Clean up Voxtral models to free memory
if voxtral_processor:
    del voxtral_processor
if voxtral_model:
    del voxtral_model

# Clean up remaining models
del diarizer
del transcriber
torch.cuda.empty_cache()

print("Processing complete! 🚀")