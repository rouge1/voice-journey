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

# Basic usage
python audio.py audio.wav                           # Process with default medium model
python audio.py audio.wav --model_size tiny         # Use smaller/faster model
python audio.py audio.wav --translate               # Translate to English

# Model management
python audio.py --list                              # Show models and cache status
python audio.py --update                            # Check for model updates (needs internet)
```

## Architecture

**Processing Pipeline:**
```
Audio → Pyannote Diarization (CUDA GPU) → Speaker Timeline
              ↓
     Whisper Transcription (CPU int8) → Aligned Transcript
```

**Key Design Patterns:**
- GPU/CPU separation: Diarization on CUDA, transcription on CPU with int8 to avoid cuDNN conflicts
- Early arg parsing: Arguments parsed before heavy imports to set `HF_HUB_OFFLINE=1` for offline mode
- Interactive token handling: Prompts for Hugging Face token on first run (gated model access)
- Memory management: `torch.cuda.empty_cache()` between pipeline stages

**Model Cache:** `~/.cache/huggingface/hub/models--*`

## Hardware Requirements

- GPU: 16GB+ VRAM with CUDA 12.4+
- RAM: 62GB+ minimum
- First run requires internet for model downloads; subsequent runs work offline
