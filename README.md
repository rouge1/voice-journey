# Voice Journey

A project for speaker diarization using PyTorch and the pyannote.audio library. This setup enables automatic speaker identification and segmentation in audio recordings.

## Features

- Speaker diarization with state-of-the-art pyannote.audio models
- Speech transcription using OpenAI Whisper (via faster-whisper)
- GPU acceleration support with CUDA
- Offline-first operation with cached models
- Separate setup/processing workflow (online setup, offline processing)
- Pre-flight cache validation with actionable error messages
- Flexible command-line interface with multiple options

## Hardware Requirements

- **GPU**: NVIDIA GeForce RTX 3080 Laptop GPU (16GB VRAM) or equivalent with CUDA support
- **CPU**: 11th Gen Intel Core i9-11980HK (8 cores, 8 threads) or similar
- **Memory**: 62 GiB RAM minimum
- **Storage**: Sufficient space for model downloads (~10GB)
- **OS**: Linux (tested on Ubuntu-based systems)

## Software Requirements

- Python 3.12
- Conda/Mamba for environment management
- CUDA 12.9+ compatible drivers
- Hugging Face account with access to pyannote models

## Installation

### 1. Create Conda Environment

```bash
conda create -n audio_llm python=3.12
conda activate audio_llm
```

### 2. Install Dependencies

```bash
# Install PyTorch and core dependencies via conda
conda install --file conda_packages.txt

# Install remaining Python packages via pip
pip install -r requirements.txt
```

This ensures proper CUDA support and all required dependencies are installed.

### 3. Set Up Hugging Face Access

The pyannote models are gated and require accepting the license terms:

1. Visit [Hugging Face](https://hf.co/settings/tokens) and create an access token
2. Request access to the following models (accept the license terms):
   - [pyannote/speaker-diarization-3.1](https://hf.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://hf.co/pyannote/segmentation-3.0)

### 4. Download Models

```bash
# Download default models (diarization + medium whisper)
python setup.py

# Or download specific whisper sizes
python setup.py --whisper-sizes medium large-v3

# Or download all whisper sizes
python setup.py --whisper-sizes all

# Pass token non-interactively
python setup.py --token YOUR_TOKEN
```

On first run, `setup.py` will prompt for your Hugging Face token. Once models are downloaded, the token is never needed again.

### 5. Verify Installation

```bash
python audio.py --list
```

This will show all available models and their cache status.

## Usage

### Command Line Options

```bash
# Model setup (online, run once)
python setup.py [OPTIONS]

Options:
  --list                  Show model cache status and exit
  --token TOKEN           Hugging Face token (or set HF_TOKEN env var)
  --whisper-sizes SIZE    Whisper sizes to download (default: medium, use "all" for all)

# Audio processing (offline)
python audio.py AUDIO_FILE [OPTIONS]

Options:
  AUDIO_FILE              Path to audio file to process (required)
  --model_size SIZE       Whisper model size: tiny, small, medium, large-v3, turbo (default: medium)
  --translate             Translate audio to English (auto-detects source language)
  --list                  Show available models and cache status
  -h, --help             Show help message

Examples:
  # First-time setup
  python setup.py
  python setup.py --whisper-sizes medium large-v3

  # Basic usage (offline)
  python audio.py audio.wav

  # Specify model size
  python audio.py audio.wav --model_size large-v3
  python audio.py audio.wav --model_size turbo

  # List models and cache status
  python audio.py --list

  # Translate Spanish/French/etc. audio to English
  python audio.py spanish_audio.wav --translate
```

### Model Sizes

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| `tiny` | ~39 MB | Fastest | Lowest | Testing, quick results |
| `small` | ~484 MB | Fast | Good | Balance of speed/quality |
| `medium` | ~1.5 GB | Medium | Better | Default, good quality |
| `large-v3` | ~2.9 GB | Slowest | Best | Maximum accuracy |
| `turbo` | ~809 MB | Fast | High | Optimized v3 performance |

### Offline vs Online Operation

- **`setup.py` (Online)**: Downloads models, handles HF authentication
- **`audio.py` (Offline)**: Processes audio using cached models, no internet required
- Pre-flight checks verify models are cached before loading heavy libraries

### Cache Status

Use `python audio.py --list` or `python setup.py --list` to see which models are cached locally:

```
Available models and cache status:

Speaker Diarization: ✅ Cached
  pyannote/speaker-diarization-3.1

Whisper Transcription Models:
  tiny     - Fastest, least accurate (~39 MB)    ✅ Cached
  small    - Balanced speed/accuracy (~484 MB)   ✅ Cached
  medium   - Default, good quality (~1.5 GB)     ✅ Cached
  large-v3 - Best accuracy, high memory (~2.9 GB) ✅ Cached
  turbo    - Fast & accurate, v3 optimized (~809 MB) ✅ Cached
```

### First Run (Setup Required)

Run `setup.py` before first use:

```
$ python setup.py
Voice Journey - Model Setup
========================================

Checking internet connectivity...
Connected.

Downloading diarization model: pyannote/speaker-diarization-3.1
Enter your Hugging Face token: <paste token here>
Diarization model cached successfully!
Downloading Whisper medium model...
Whisper medium cached successfully!

Setup complete! Process audio with:
  python audio.py <audio_file>
```

### Subsequent Runs (No Setup Needed)

Models load directly from cache at `~/.cache/huggingface/hub/`. `audio.py` runs completely offline.

If a required model is missing, `audio.py` will tell you exactly which `setup.py` command to run:
```
Error: Whisper large-v3 model not found in cache.
Run 'python setup.py --whisper-sizes large-v3' to download it.
```

### Output Format

The script produces timestamped, speaker-labeled transcripts:

```
============================================================
SPEAKER-LABELED TRANSCRIPT
============================================================

[0:00:00] SPEAKER_00:
[0:00:00 - 0:00:02] Well, stocks are mooning today, which is great.
[0:00:02 - 0:00:06] And part of it is because of the TikTok deal and Oracle.

[0:00:06] SPEAKER_01:
[0:00:06 - 0:00:10] But what do we need to know about that deal that could create...
```

## Project Structure

```
voice-journey/
├── models.py              # Shared model constants, cache paths, list_models()
├── setup.py               # Online: download models, HF token setup
├── audio.py               # Offline: speaker diarization + transcription
├── requirements.txt       # Python pip dependencies
├── conda_packages.txt     # Conda environment packages
├── CLAUDE.md              # AI assistant guidance
├── README.md              # Project documentation
├── sessions/              # Sample audio files for testing
└── .github/
    └── copilot-instructions.md  # GitHub Copilot guidance
```

## Troubleshooting

### Common Issues

1. **"Diarization model not found in cache"**:
   - Run `python setup.py` to download models first
   - Requires internet and a Hugging Face token

2. **"Whisper X model not found in cache"**:
   - Run `python setup.py --whisper-sizes <size>` to download the specific model

3. **Hugging Face Access Denied (401/403)**:
   - Verify you've accepted model licenses at hf.co/pyannote/speaker-diarization-3.1
   - Check token validity at hf.co/settings/tokens

4. **CUDA Not Available**:
   - Ensure GPU drivers are up to date
   - Check CUDA installation: `nvidia-smi`

5. **Memory Issues**:
   - Speaker diarization is memory-intensive
   - Try smaller Whisper models: `--model_size tiny`
   - Ensure sufficient RAM (62GB+) and VRAM (16GB+)

6. **Model Cache Issues**:
   - Check cache status: `python audio.py --list`
   - Clear cache if needed: `rm -rf ~/.cache/huggingface/hub/models--*`
   - Re-run `python setup.py` to re-download

### Network Requirements

- **Setup**: Internet required only for `python setup.py` (model downloads)
- **Processing**: `python audio.py` runs completely offline
- Models are cached at `~/.cache/huggingface/hub/`

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is for educational and research purposes. Please respect the licenses of the underlying libraries and models.
