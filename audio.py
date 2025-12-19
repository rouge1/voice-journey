import torch
from pyannote.audio import Pipeline

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

# ←←← PUT YOUR REAL HF TOKEN HERE ←←←
HF_TOKEN = "hf_your_real_token_here"

print("Loading pyannote speaker diarization pipeline...")
try:
    # Try to load from cache first (no token needed if cached)
    diarizer = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    print("Loaded from cache successfully!")
except Exception as e:
    if "401" in str(e) or "403" in str(e) or "Unauthorized" in str(e) or "gated" in str(e).lower():
        print("Model not cached or requires authentication. Loading with token...")
        diarizer = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=HF_TOKEN
        )
    else:
        raise e

diarizer.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
print("Pyannote diarization pipeline loaded successfully on GPU! 🚀")