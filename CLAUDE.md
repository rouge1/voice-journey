# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Voice Journey is a Python CLI for speaker diarization and speech transcription. It combines pyannote.audio (GPU diarization) with faster-whisper (CPU transcription) for offline-first operation with cached ML models.

## Commands

```bash
# Environment setup
conda create -n audio_llm python=3.12 && conda activate audio_llm
conda install --file conda_packages.txt
pip install -r requirements.txt

# First-time model setup (requires internet)
python setup.py                                     # Download default models (diarization + medium whisper)
python setup.py --update medium large      # Download specific whisper sizes
python setup.py --whisper-sizes all                  # Download all whisper sizes
python setup.py --token YOUR_TOKEN                   # Pass HF token non-interactively
python setup.py --list                               # Show models and cache status

# Audio processing (offline, uses cached models)
python audio.py audio.wav                            # Process with default medium model
python audio.py audio.wav --model tiny          # Use smaller/faster model
python audio.py audio.wav --translate                # Translate to English
python audio.py --list                               # Show models and cache status
```

## Architecture

**Files:**
- `models.py` — Shared model constants, cache paths, `list_models()` utility
- `setup.py` — Online: downloads models, guides HF token setup
- `audio.py` — Offline: processes audio, assumes models are cached

**Processing Pipeline:**
```
Audio → Pyannote Diarization (CUDA GPU) → Speaker Timeline
              ↓
     Whisper Transcription (CPU int8) → Aligned Transcript
```

**Key Design Patterns:**
- Online/offline separation: `setup.py` handles downloads, `audio.py` runs fully offline
- GPU/CPU separation: Diarization on CUDA, transcription on CPU with int8 to avoid cuDNN conflicts
- Pre-flight cache checks: `audio.py` verifies models exist before heavy imports
- Early `--list` handling: Checked via `sys.argv` before argparse (since `audio_file` is required)
- Memory management: `torch.cuda.empty_cache()` between pipeline stages

**Model Cache:** `~/.cache/huggingface/hub/models--*`

## Common Pitfalls

- **`diarization.speaker_diarization` doesn't exist**: Pyannote's `Pipeline.__call__()` returns an `Annotation` directly. Use `diarization.itertracks()` and `diarization.get_timeline()` — not `diarization.speaker_diarization.itertracks()`.
- **Whisper turbo cache path differs**: The turbo model is `mobiuslabsgmbh/faster-whisper-large-v3-turbo`, not `Systran/faster-whisper-turbo`. See `models.py:_whisper_cache_path()`.
- **Heavy imports are slow**: `torch`, `pyannote.audio`, `faster_whisper` take ~10s to import. Always validate inputs and check cache *before* importing them.
- **`audio_file` is a required positional arg**: To support `--list` without requiring a file, check `sys.argv` before `argparse.parse_args()`.
- **cuDNN conflicts**: Running both diarization (GPU) and whisper (GPU) can cause cuDNN errors. Whisper runs on CPU with int8 to avoid this.

## Hardware Requirements

- GPU: 16GB+ VRAM with CUDA 12.4+
- RAM: 62GB+ minimum
- First run requires internet for model downloads via `setup.py`; `audio.py` works offline
