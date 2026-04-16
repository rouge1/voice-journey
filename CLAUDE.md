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
python setup.py --update medium large turbo         # Download specific whisper sizes
python setup.py --update all                        # Download all whisper sizes
python setup.py --token YOUR_TOKEN                  # Pass HF token non-interactively
python setup.py --list                              # Show models and cache status

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
- `wav_tags.py` — WAV metadata utility: `get_wav_real_duration()`, `format_duration()`, `read_riff_info()`, `RIFF_INFO_NAMES`; also a standalone CLI (`python wav_tags.py <file.wav>` or `python wav_tags.py <dir>`) for inspecting a single file or scanning a directory

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

- **Pyannote returns `DiarizeOutput`, not `Annotation`**: `Pipeline.__call__()` returns a `DiarizeOutput` dataclass. Access the annotation via `diarization.speaker_diarization.itertracks()` and `diarization.speaker_diarization.get_timeline()`.
- **Don't set `HF_HUB_OFFLINE=1`**: It breaks pyannote's `from_pretrained` cache resolution. Let huggingface_hub use cached models automatically without forcing offline mode.
- **Whisper turbo cache path differs**: The turbo model is `mobiuslabsgmbh/faster-whisper-large-v3-turbo`, not `Systran/faster-whisper-turbo`. See `models.py:_whisper_cache_path()`.
- **Heavy imports are slow**: `torch`, `pyannote.audio`, `faster_whisper` take ~10s to import. Always validate inputs and check cache *before* importing them.
- **`audio_file` is a required positional arg**: To support `--list` without requiring a file, check `sys.argv` before `argparse.parse_args()`.
- **cuDNN conflicts**: Running both diarization (GPU) and whisper (GPU) can cause cuDNN errors. Whisper runs on CPU with int8 to avoid this.
- **WAV streaming-mode duration is wrong**: SDR/intercept tools often write `0x00000000` or `0xFFFFFFFF` as the `data` chunk size (streaming mode), so `mutagen` reports 0:00 or thousands of hours. Use `get_wav_real_duration()` from `wav_tags.py`, which ignores the header's stated size and computes duration from actual file size minus the data-chunk offset.
- **`wav_tags` import in `audio.py` is lazy/optional**: The metadata block is wrapped in `try/except Exception: pass` so `audio.py` still runs if `mutagen` isn't installed or the file isn't a standard WAV.
- **mutagen WAVE `.tags` only returns ID3 tags, not RIFF INFO**: For WAV files with a RIFF LIST/INFO chunk but no embedded ID3 (e.g. files encoded by ffmpeg/lavf), `mutagen.File().tags` is `None`. RIFF INFO tags like `ISFT` (Software) are invisible to mutagen. Use `read_riff_info()` from `wav_tags.py` to parse them directly from the binary structure.
- **`format_duration()` truncates sub-second files to `0:00`**: Using `int(seconds)` before building a `timedelta` drops any fractional part, so a 0.8s file displays as `0:00`. Use `round(seconds)` instead. Files under 1 second are shown as `0:00 (0.42s)` to distinguish genuinely-short clips from broken headers.
- **Don't trust diarization's timeline extent for file duration**: `diarization.speaker_diarization.get_timeline().extent()` only spans detected-speech regions, so silent head/tail gets clipped. Load the waveform with `torchaudio.load()` and compute `waveform.shape[1] / sample_rate` for true file duration.
- **Short audio (<1s) needs deferred timeline rendering**: The 1-char-per-second timeline renders empty for sub-second clips. `audio.py` sets `is_short_audio = file_duration < 1.0` and defers timeline output until after transcription, then prints `"x"` if Whisper found speech or `"_"` otherwise. Also wrap `range(0, int(duration_seconds))` with `max(1, ...)` so a zero-int-duration file still produces a one-cell timeline.
- **`list_models_with_updates()` must fall back gracefully when offline**: If the HF update check fails (no network), don't surface a raw error — check `is_diarization_cached()` / `is_whisper_cached(size)` and print `"✅ Cached (⚠️ update check failed, offline?)"` so `--list` stays useful without internet.

## Hardware Requirements

- GPU: 16GB+ VRAM with CUDA 12.4+
- RAM: 62GB+ minimum
- First run requires internet for model downloads via `setup.py`; `audio.py` works offline
