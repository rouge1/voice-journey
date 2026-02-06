import os

CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")

DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"

WHISPER_SIZES = ["tiny", "small", "medium", "large-v3", "turbo"]

WHISPER_SIZE_INFO = {
    "tiny": "Fastest, least accurate (~39 MB)",
    "small": "Balanced speed/accuracy (~484 MB)",
    "medium": "Default, good quality (~1.5 GB)",
    "large-v3": "Best accuracy, high memory (~2.9 GB)",
    "turbo": "Fast & accurate, v3 optimized (~809 MB)",
}


def _diarization_cache_path():
    return os.path.join(CACHE_DIR, "models--pyannote--speaker-diarization-3.1")


def _whisper_cache_path(size):
    if size == "turbo":
        return os.path.join(CACHE_DIR, "models--mobiuslabsgmbh--faster-whisper-large-v3-turbo")
    return os.path.join(CACHE_DIR, f"models--Systran--faster-whisper-{size}")


def is_diarization_cached():
    return os.path.exists(_diarization_cache_path())


def is_whisper_cached(size):
    return os.path.exists(_whisper_cache_path(size))


def list_models():
    status = "✅ Cached" if is_diarization_cached() else "❌ Not cached"
    print("Available models and cache status:")
    print()
    print(f"Speaker Diarization: {status}")
    print(f"  {DIARIZATION_MODEL}")
    print()
    print("Whisper Transcription Models:")
    for size in WHISPER_SIZES:
        cached = "✅ Cached" if is_whisper_cached(size) else "❌ Not cached"
        print(f"  {size:<8} - {WHISPER_SIZE_INFO[size]:<35} {cached}")
    print()
