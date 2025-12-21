# Voice Journey

A project for speaker diarization using PyTorch and the pyannote.audio library. This setup enables automatic speaker identification and segmentation in audio recordings.

## Features

- Speaker diarization with state-of-the-art models
- GPU acceleration support
- Easy setup with conda environment
- Integration with Hugging Face models

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
conda install -c pytorch -c nvidia -c conda-forge pytorch torchaudio pytorch-cuda=12.4 ffmpeg
pip install -r requirements.txt
```

This installs all required Python packages (pyannote-audio, faster-whisper, huggingface-hub, transformers, soundfile, tqdm).

### 3. Set Up Hugging Face Access

The pyannote models are gated and require accepting the license terms:

1. Visit [Hugging Face](https://hf.co/settings/tokens) and create an access token
2. Request access to the following models (accept the license terms):
   - [pyannote/speaker-diarization-3.1](https://hf.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://hf.co/pyannote/segmentation-3.0)

**Note**: On first run, the script will prompt you for your token to download the models. Once downloaded, models are cached locally and the token is never needed again.

### 4. Verify Installation

```bash
python audio.py your_audio_file.wav
```

On first run:
- You'll be prompted for your Hugging Face token
- Models will be downloaded and cached (~2-3GB)
- Subsequent runs load instantly from cache

You should see output confirming PyTorch version, CUDA availability, and successful model loading.

## Usage

### Command Line

```bash
# Process an audio file with default medium model
python audio.py path/to/audio.wav

# Specify model size (tiny, small, medium, large-v3)
python audio.py path/to/audio.wav large-v3
```

Model sizes:
- `tiny`: Fastest, least accurate (good for testing)
- `small`: Balanced speed/accuracy
- `medium`: Default, good quality
- `large-v3`: Best accuracy, requires more VRAM/CPU

### First Run (Token Required)

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

Models load directly from cache at `~/.cache/huggingface/hub/`.

### Python API Example

```python
import torch
from pyannote.audio import Pipeline

# Load the pipeline (prompts for token if not cached)
try:
    diarizer = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
except Exception as e:
    # Handle first-time download
    hf_token = input("Enter Hugging Face token: ")
    diarizer = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token
    )

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
diarizer.to(device)

# Process audio file
diarization = diarizer("path/to/audio.wav")

# Print results
for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
```

### Main Script

The `audio.py` file is a complete speaker diarization and transcription tool:
- Accepts audio file as command-line argument
- Performs speaker diarization (who spoke when)
- Transcribes speech using Whisper
- Outputs speaker-labeled transcript with timestamps

## Project Structure

```
voice-journey/
├── README.md              # This file
├── audio.py               # Main diarization/transcription script
└── requirements.txt       # Pip requirements for dependencies
```

## Troubleshooting

### Common Issues

1. **TorchCodec Warning**: This is usually harmless and doesn't affect diarization functionality.

2. **TensorFloat-32 (TF32) Warning**: This has been suppressed in the code for better performance. TF32 provides faster computation with minimal precision loss.

3. **CUDA Not Available**: Ensure your GPU drivers are up to date and CUDA is properly installed.

4. **Hugging Face Access Denied**: Make sure you have requested access to all required models and your token is valid.

5. **Memory Issues**: Speaker diarization can be memory-intensive; ensure you have sufficient RAM and VRAM.

### Performance Tips

- Use GPU acceleration for faster processing
- Process shorter audio segments for better memory management
- Consider model quantization for reduced memory usage

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is for educational and research purposes. Please respect the licenses of the underlying libraries and models.
