# Voice Journey - AI Coding Agent Instructions

## Project Overview
Voice Journey is a Python CLI tool that combines speaker diarization (identifying who spoke when) with speech transcription. It uses pyannote.audio for GPU-accelerated diarization and faster-whisper for CPU-based transcription, designed for offline-first operation with cached ML models.

## Architecture & Key Patterns

### Processing Pipeline
```
Audio Input → Pyannote Diarization (GPU) → Speaker Timeline
                    ↓
         Whisper Transcription (CPU) → Transcript Alignment
```
- **GPU/CPU Separation**: Diarization runs on CUDA, transcription on CPU with int8 quantization to avoid cuDNN conflicts
- **Memory Management**: Explicit `torch.cuda.empty_cache()` between stages (see `audio.py:180,220`)
- **Offline-First**: Models cached in `~/.cache/huggingface/hub/`, internet only for first download or `--update`

### Early Argument Parsing Pattern
Arguments parsed before heavy imports to set environment variables:
```python
# Parse args first (audio.py:15-25)
args = parser.parse_args()
if not args.update:
    os.environ["HF_HUB_OFFLINE"] = "1"  # Force offline mode
# Then import heavy libraries (audio.py:75+)
```

### Interactive Token Handling
For gated Hugging Face models, prompt user interactively on first run:
```python
try:
    diarizer = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
except Exception as e:
    if "401" in str(e):  # Token required
        hf_token = input("Enter your Hugging Face token: ")
        diarizer = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=hf_token)
```

## Critical Workflows

### Environment Setup
```bash
conda create -n audio_llm python=3.12
conda activate audio_llm
conda install --file conda_packages.txt  # PyTorch + CUDA
pip install -r requirements.txt          # Remaining deps
```

### Command Line Interface
```bash
usage: audio.py [-h] [--model_size {tiny,small,medium,large-v3,turbo}] [--update] [--list]
                [--translate]
                [audio_file]

Process audio file with speaker diarization and transcription.

positional arguments:
  audio_file            Path to the audio file to process (optional with --update)

options:
  -h, --help            show this help message and exit
  --model_size {tiny,small,medium,large-v3,turbo}
                        Whisper model size (default: medium)
  --update              Allow online updates for models (checks connectivity first)
  --list                List available models and their cache status
  --translate           Translate audio to English (default: transcribe in original language)
```

### Model Management
- **Cache Location**: `~/.cache/huggingface/hub/models--*`
- **Check Status**: `python audio.py --list`
- **Force Updates**: `python audio.py --update` (requires internet)
- **Network Check**: Automatic connectivity detection before downloads

### Hardware Requirements
- GPU: 16GB+ VRAM (RTX 3080 tested)
- RAM: 62GB+ minimum
- CUDA: 12.4+ compatible drivers

## Development Conventions

### Error Handling Patterns
- Network errors during `--update` → Graceful fallback to cached models
- Missing audio files → Clear error messages with usage hints
- GPU memory issues → Suggest smaller models (`--model_size tiny`)

### Model Selection Trade-offs
| Model | Size | Use Case | Example Command |
|-------|------|----------|-----------------|
| tiny | 39MB | Testing/fast | `python audio.py file.wav --model_size tiny` |
| small | 484MB | Balanced | `python audio.py file.wav --model_size small` |
| medium | 1.5GB | Default | `python audio.py file.wav` |
| large-v3 | 2.9GB | Max accuracy | `python audio.py file.wav --model_size large-v3` |
| turbo | 809MB | Fast v3 | `python audio.py file.wav --model_size turbo` |

### Testing
- Sample audio files in `sessions/` directory
- Test with: `python audio.py sessions/test_clip.wav --model_size tiny`

## Integration Points

### External Dependencies
- **Hugging Face**: Gated models require token acceptance at hf.co/pyannote/speaker-diarization-3.1
- **CUDA**: GPU acceleration via PyTorch (conda-installed for compatibility)
- **Audio Processing**: torchaudio for loading, soundfile for I/O

### Configuration
- No config files - all settings via CLI args
- Environment variables set programmatically (HF_HUB_OFFLINE, etc.)
- Model paths hardcoded to Hugging Face hub locations

## Key Files Reference
- `audio.py`: Main script with CLI parsing, model loading, processing pipeline
- `requirements.txt`: Python dependencies (pip install)
- `conda_packages.txt`: Conda environment with CUDA PyTorch
- `sessions/`: Test audio files for development
- `CLAUDE.md`: Additional AI assistant guidance (complements this file)</content>
<parameter name="filePath">/data/python/voice-journey/.github/copilot-instructions.md