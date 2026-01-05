# Voice Journey

A project for speaker diarization using PyTorch and the pyannote.audio library. This setup enables automatic speaker identification and segmentation in audio recordings.

## Features

- Speaker diarization with state-of-the-art pyannote.audio models
- Speech transcription using OpenAI Whisper (via faster-whisper)
- GPU acceleration support with CUDA
- Offline-first operation with cached models
- Intelligent network connectivity checking
- Flexible command-line interface with multiple options
- Automatic error handling and user-friendly messages

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

**Note**: On first run, the script will prompt you for your token to download the models. Once downloaded, models are cached locally and the token is never needed again.

### 4. Verify Installation

```bash
python audio.py --list
```

This will show all available models and their cache status. You should see output confirming PyTorch version, CUDA availability, and model cache status.

## Usage

### Command Line Options

```bash
python audio.py [OPTIONS] [AUDIO_FILE]

Options:
  AUDIO_FILE              Path to audio file to process (optional with --update)
  --model_size SIZE       Whisper model size: tiny, small, medium, large-v3 (default: medium)
  --update                Check for online model updates (requires internet)
  --list                  Show available models and cache status
  -h, --help             Show help message

Examples:
  # Basic usage (offline mode)
  python audio.py audio.wav
  
  # Specify model size
  python audio.py audio.wav --model_size large-v3
  
  # Check for updates and process
  python audio.py audio.wav --update
  
  # Update models only (no audio processing)
  python audio.py --update
  
  # List models and cache status
  python audio.py --list
  
  # Combine options
  python audio.py audio.wav --model_size small --update
```

### Model Sizes

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| `tiny` | ~39 MB | Fastest | Lowest | Testing, quick results |
| `small` | ~484 MB | Fast | Good | Balance of speed/quality |
| `medium` | ~1.5 GB | Medium | Better | Default, good quality |
| `large-v3` | ~2.9 GB | Slowest | Best | Maximum accuracy |

### Offline vs Online Operation

- **Default (Offline)**: Uses cached models, no internet required
- **With `--update`**: Checks for model updates, requires internet connection
- **Network Check**: Automatically detects connectivity before attempting updates

### Cache Status

Use `python audio.py --list` to see which models are cached locally:

```
Available models and cache status:

Speaker Diarization: ✅ Cached
  pyannote/speaker-diarization-3.1

Whisper Transcription Models:
  tiny     - Fastest, least accurate (~39 MB)    ✅ Cached
  small    - Balanced speed/accuracy (~484 MB)   ✅ Cached
  medium   - Default, good quality (~1.5 GB)     ✅ Cached
  large-v3 - Best accuracy, high memory (~2.9 GB) ✅ Cached
```

### First Run (Token Required)

On first run with uncached models, you'll see:

```
Loading pyannote speaker diarization pipeline...

============================================================
Model not cached. Hugging Face token required for first download.
Get your token at: https://hf.co/settings/tokens
Make sure you've accepted the model terms at:
  https://hf.co/pyannote/speaker-diarization-3.1
============================================================

Enter your Hugging Face token: <paste token here>
Downloading model (this only needs to happen once)...
Model cached successfully! Token won't be needed again.
```

### Subsequent Runs (No Token Needed)

Models load directly from cache at `~/.cache/huggingface/hub/`. The script runs completely offline by default.

### Offline Operation

The script is designed to work offline once models are cached:

- No internet required for normal operation
- Cached models persist between runs
- Use `--update` only when you want to check for newer model versions
- Network connectivity is automatically detected

### Error Handling

The script gracefully handles various scenarios:

- **No internet with `--update`**: Shows friendly message and uses cached models
- **Invalid token**: Prompts for re-entry
- **Missing audio file**: Clear error messages
- **Model loading failures**: Fallback to cached versions when possible

### Python API Example

For programmatic use, you can use the underlying libraries directly:

```python
import torch
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel

# Load diarization pipeline (handles token prompting automatically)
diarizer = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarizer.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Load Whisper model
whisper_model = WhisperModel("medium", device="cuda", compute_type="int8")

# Process audio
diarization = diarizer("audio.wav")
segments, info = whisper_model.transcribe("audio.wav", vad_filter=True)

# Combine results
for segment in segments:
    # Find which speaker was active during this segment
    speakers = diarization.speaker_diarization.crop(segment.start, segment.end)
    active_speaker = max(speakers.labels(), key=lambda s: speakers.label_duration(s))
    
    print(f"[{segment.start:.1f}s] SPEAKER_{active_speaker}: {segment.text}")
```

### Main Script Features

The `audio.py` script provides a complete command-line solution for speaker diarization and transcription:

- **Speaker Diarization**: Identifies who spoke when using pyannote.audio
- **Speech Transcription**: Converts speech to text using OpenAI Whisper
- **Flexible Model Selection**: Choose from 4 Whisper model sizes
- **Offline Operation**: Works without internet once models are cached
- **Smart Updates**: Checks connectivity before attempting downloads
- **Error Handling**: Graceful handling of network issues and missing files
- **GPU Acceleration**: Automatic CUDA detection and utilization

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
├── README.md              # Comprehensive documentation
├── audio.py               # Main diarization/transcription script
├── requirements.txt       # Python pip dependencies
├── conda_packages.txt     # Conda environment packages
└── test.py               # Legacy test script (deprecated)
```

## Troubleshooting

### Common Issues

1. **"No internet connection detected"**: 
   - Expected when offline - script will use cached models
   - Connect to internet and use `--update` to check for updates

2. **TorchCodec Warning**: 
   - Harmless warning, doesn't affect functionality
   - Can be ignored

3. **TensorFloat-32 (TF32) Warning**: 
   - Suppressed in code for better performance
   - TF32 provides faster computation with minimal precision loss

4. **CUDA Not Available**: 
   - Ensure GPU drivers are up to date
   - Check CUDA installation: `nvidia-smi`

5. **Hugging Face Access Denied**: 
   - Verify you've accepted model terms on Hugging Face
   - Check token validity
   - Models are cached after first download

6. **Memory Issues**: 
   - Speaker diarization is memory-intensive
   - Try smaller Whisper models: `--model_size tiny`
   - Ensure sufficient RAM (62GB+) and VRAM (16GB+)

7. **Model Cache Issues**:
   - Check cache status: `python audio.py --list`
   - Clear cache if needed: `rm -rf ~/.cache/huggingface/hub/models--*`
   - Re-run to re-download models

### Performance Tips

- **GPU Usage**: Ensure CUDA is available for significant speedups
- **Model Selection**: Use smaller models for faster processing
- **Batch Processing**: Process shorter audio segments for better memory management
- **Offline Operation**: Work offline once models are cached
- **Regular Updates**: Use `--update` periodically to get model improvements

### Network Requirements

- **First Run**: Internet required for model downloads
- **Normal Operation**: Completely offline capable
- **Updates**: Internet required only when using `--update`
- **Automatic Detection**: Script checks connectivity before attempting downloads

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is for educational and research purposes. Please respect the licenses of the underlying libraries and models.
