#!/usr/bin/env python3
"""Online model setup for Voice Journey.

Downloads pyannote diarization and faster-whisper models,
guiding the user through Hugging Face token authentication.
"""

import argparse
import os
import socket
import sys

from models import (
    DIARIZATION_MODEL,
    WHISPER_SIZES,
    is_diarization_cached,
    is_whisper_cached,
    list_models,
)


def check_network_connectivity():
    """Quick check if we can reach the internet."""
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False


def get_hf_token(args_token):
    """Resolve HF token from flag, env var, or interactive prompt."""
    token = args_token or os.environ.get("HF_TOKEN")
    if token:
        return token

    print("\n" + "=" * 60)
    print("Hugging Face token required for pyannote diarization model.")
    print()
    print("Steps to get a token:")
    print("  1. Create account: https://huggingface.co/join")
    print("  2. Create token:   https://hf.co/settings/tokens")
    print("  3. Accept model licenses:")
    print("     https://hf.co/pyannote/speaker-diarization-3.1")
    print("     https://hf.co/pyannote/segmentation-3.0")
    print("=" * 60)
    token = input("\nEnter your Hugging Face token: ").strip()
    if not token:
        print("Error: Token is required.")
        sys.exit(1)
    return token


def download_diarization(token):
    """Download pyannote diarization model."""
    if is_diarization_cached():
        print(f"Diarization model already cached, skipping.")
        return True

    print(f"\nDownloading diarization model: {DIARIZATION_MODEL}")
    try:
        from pyannote.audio import Pipeline

        Pipeline.from_pretrained(DIARIZATION_MODEL, token=token)
        print("Diarization model cached successfully!")
        return True
    except Exception as e:
        if "401" in str(e) or "403" in str(e) or "gated" in str(e).lower():
            print(f"\nAuth error: {e}")
            print("Check that your token is valid and you accepted the model licenses:")
            print("  https://hf.co/pyannote/speaker-diarization-3.1")
            print("  https://hf.co/pyannote/segmentation-3.0")
        else:
            print(f"\nFailed to download diarization model: {e}")
        return False


def download_whisper(size):
    """Download a single faster-whisper model."""
    if is_whisper_cached(size):
        print(f"Whisper {size} already cached, skipping.")
        return True

    print(f"Downloading Whisper {size} model...")
    try:
        from faster_whisper import WhisperModel

        WhisperModel(size, device="cpu", compute_type="int8")
        print(f"Whisper {size} cached successfully!")
        return True
    except Exception as e:
        print(f"Failed to download Whisper {size}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download and set up models for Voice Journey."
    )
    parser.add_argument(
        "--list", action="store_true", help="Show model cache status and exit"
    )
    parser.add_argument("--token", help="Hugging Face token (or set HF_TOKEN env var)")
    parser.add_argument(
        "--whisper-sizes",
        nargs="+",
        default=["medium"],
        metavar="SIZE",
        help='Whisper sizes to download (default: medium, use "all" for all sizes)',
    )
    args = parser.parse_args()

    if args.list:
        list_models()
        sys.exit(0)

    # Resolve whisper sizes
    if "all" in args.whisper_sizes:
        whisper_sizes = WHISPER_SIZES
    else:
        for s in args.whisper_sizes:
            if s not in WHISPER_SIZES:
                print(f"Error: Unknown whisper size '{s}'. Choose from: {', '.join(WHISPER_SIZES)}")
                sys.exit(1)
        whisper_sizes = args.whisper_sizes

    print("Voice Journey - Model Setup")
    print("=" * 40)
    print()
    list_models()

    # Check connectivity
    print("Checking internet connectivity...")
    if not check_network_connectivity():
        print("No internet connection detected. Connect to the internet and try again.")
        sys.exit(1)
    print("Connected.\n")

    # Determine what needs downloading
    need_diarization = not is_diarization_cached()
    need_whisper = [s for s in whisper_sizes if not is_whisper_cached(s)]

    if not need_diarization and not need_whisper:
        print(f"Requested models (whisper: {', '.join(whisper_sizes)}) are already set up.")
        sys.exit(0)

    # Download diarization
    diarization_ok = True
    if need_diarization:
        token = get_hf_token(args.token)
        diarization_ok = download_diarization(token)
    else:
        print("Diarization model already cached.")

    # Download whisper models
    whisper_results = {}
    for size in whisper_sizes:
        whisper_results[size] = download_whisper(size)

    # Summary
    print("\n" + "=" * 40)
    print("Setup Summary")
    print("=" * 40)
    d_status = "✅" if is_diarization_cached() else "❌"
    print(f"  Diarization: {d_status}")
    for size in whisper_sizes:
        w_status = "✅" if is_whisper_cached(size) else "❌"
        print(f"  Whisper {size}: {w_status}")

    if diarization_ok and all(whisper_results.values()):
        print("\nSetup complete! Process audio with:")
        print("  python audio.py <audio_file>")
    else:
        print("\nSome downloads failed. Re-run setup to retry.")
        sys.exit(1)


if __name__ == "__main__":
    main()
