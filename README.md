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
# Install conda packages
conda install --file conda_packages.txt

# Optional: Install additional pip packages if needed
pip install -r pip_requirements.txt
```

### 3. Set Up Hugging Face Access (Optional for Cached Models)

If the models are not cached locally, you'll need:

1. Visit [Hugging Face](https://hf.co/settings/tokens) and create an access token
2. Request access to the following models:
   - [pyannote/speaker-diarization-3.1](https://hf.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://hf.co/pyannote/segmentation-3.0)
   - [pyannote/speaker-diarization-community-1](https://hf.co/pyannote/speaker-diarization-community-1)
3. Update the `HF_TOKEN` in `audio.py` with your token

**Note**: Once models are downloaded, they can be loaded from cache without requiring the token again.

### 4. Verify Installation

```bash
python audio.py
```

You should see output confirming PyTorch version, CUDA availability, and successful model loading.

## Usage

### Basic Speaker Diarization

```python
import torch
from pyannote.audio import Pipeline

# Load the pipeline (token only needed if not cached)
HF_TOKEN = "your_huggingface_token_here"  # Optional if models are cached

try:
    diarizer = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
except Exception as e:
    if "401" in str(e) or "403" in str(e):
        diarizer = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=HF_TOKEN
        )

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
diarizer.to(device)

# Process audio file
diarization = diarizer("path/to/audio.wav")

# Print results
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
```

### Test Script

The `audio.py` file demonstrates basic setup and can be used as a starting point for your applications.

## Project Structure

```
voice-journey/
├── README.md              # This file
├── requirements.txt       # Detailed requirements documentation
├── audio.py               # Basic test script
├── conda_packages.txt    # Conda environment export
└── pip_packages.txt      # Pip packages list
```

## Troubleshooting

### Common Issues

1. **TorchCodec Warning**: This is usually harmless and doesn't affect diarization functionality.

2. **TensorFloat-32 (TF32) Warning**: This has been suppressed in the code for better performance. TF32 provides faster computation with minimal precision loss.

3. **CUDA Not Available**: Ensure your GPU drivers are up to date and CUDA is properly installed.

3. **Hugging Face Access Denied**: Make sure you have requested access to all required models and your token is valid.

4. **Memory Issues**: Speaker diarization can be memory-intensive; ensure you have sufficient RAM and VRAM.

### Performance Tips

- Use GPU acceleration for faster processing
- Process shorter audio segments for better memory management
- Consider model quantization for reduced memory usage

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is for educational and research purposes. Please respect the licenses of the underlying libraries and models.
