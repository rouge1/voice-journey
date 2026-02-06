# Voice Journey - AI Coding Agent Instructions

## Project Overview
Voice Journey is a Python CLI tool that combines speaker diarization (identifying who spoke when) with speech transcription. It uses pyannote.audio for GPU-accelerated diarization and faster-whisper for CPU-based transcription, designed for offline-first operation with cached ML models.

## Architecture & Key Patterns

### File Structure
- `models.py` — Shared model constants, cache paths, `list_models()` utility
- `setup.py` — Online: downloads models, guides HF token setup
- `audio.py` — Offline: processes audio, assumes models are cached

### Processing Pipeline
```
Audio Input → Pyannote Diarization (GPU) → Speaker Timeline
                    ↓
         Whisper Transcription (CPU) → Transcript Alignment
```
- **Online/Offline Separation**: `setup.py` handles downloads, `audio.py` runs fully offline
- **GPU/CPU Separation**: Diarization runs on CUDA, transcription on CPU with int8 quantization to avoid cuDNN conflicts
- **Memory Management**: Explicit `torch.cuda.empty_cache()` between stages
- **Pre-flight Cache Checks**: `audio.py` verifies models exist before heavy imports
- **Early `--list` Handling**: Checked via `sys.argv` before argparse (since `audio_file` is required)

### Pyannote API Usage
The diarization pipeline returns an `Annotation` directly:
```python
diarization = diarizer(audio_file)
# Correct:
diarization.itertracks(yield_label=True)
diarization.get_timeline().extent()
# Wrong (old API / never existed):
# diarization.speaker_diarization.itertracks(...)
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
# Model setup (online, run once)
python setup.py                                # Download default models
python setup.py --whisper-sizes medium large-v3 # Download specific sizes
python setup.py --token YOUR_TOKEN             # Non-interactive token

# Audio processing (offline)
python audio.py audio.wav                      # Process with default medium model
python audio.py audio.wav --model_size tiny    # Use smaller model
python audio.py audio.wav --translate          # Translate to English
python audio.py --list                         # Show model cache status
```

### Model Management
- **Cache Location**: `~/.cache/huggingface/hub/models--*`
- **Check Status**: `python audio.py --list` or `python setup.py --list`
- **Download Models**: `python setup.py` (requires internet + HF token for diarization)

### Hardware Requirements
- GPU: 16GB+ VRAM (RTX 3080 tested)
- RAM: 62GB+ minimum
- CUDA: 12.4+ compatible drivers

## Development Conventions

### Error Handling Patterns
- Missing models → Actionable message: "Run `python setup.py --whisper-sizes <size>`"
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
- `models.py`: Shared model constants, cache paths, `list_models()` utility
- `setup.py`: Online model setup — downloads, HF token auth
- `audio.py`: Offline audio processing — diarization + transcription
- `requirements.txt`: Python dependencies (pip install)
- `conda_packages.txt`: Conda environment with CUDA PyTorch
- `sessions/`: Test audio files for development
- `CLAUDE.md`: Additional AI assistant guidance (complements this file)

## User interactions
- **After making changes**: Run some unit tests to make sure we caught silly things like check
ing imports, checking file structure
- **Remind the user**: Remind the user how to see the changes that were just made. For example
, restart the app.py or refresh the webpage with (ctrl + shift + R). 

</content>
<parameter name="filePath">/data/python/voice-journey/.github/copilot-instructions.md