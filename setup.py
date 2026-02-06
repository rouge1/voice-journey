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
    remove_diarization_cache,
    remove_whisper_cache,
    _get_whisper_model_name,
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

        Pipeline.from_pretrained(DIARIZATION_MODEL, use_auth_token=token)
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

        actual_model = _get_whisper_model_name(size)
        WhisperModel(actual_model, device="cpu", compute_type="int8")
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
    parser.add_argument(
        "--remove",
        nargs="+",
        metavar="MODEL",
        help='Remove cached models (use "diarization" or whisper sizes like "medium", or "all")',
    )
    parser.add_argument(
        "--update",
        nargs="+",
        metavar="MODEL",
        help='Download/update models (diarization, whisper sizes like "medium", or "all". Downloads if missing, updates if SHA mismatch detected)',
    )
    parser.add_argument("--token", help="Hugging Face token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    # If no action specified, show list and exit
    if not args.list and not args.remove and not args.update:
        list_models()
        print("\nUsage:")
        print("  python setup.py --list              # Show model status")
        print("  python setup.py --update medium     # Download/update models")
        print("  python setup.py --remove medium     # Remove cached models")
        sys.exit(0)

    if args.list:
        list_models()
        sys.exit(0)

    if args.remove:
        print("Voice Journey - Remove Models")
        print("=" * 40)
        print()
        list_models()
        
        removed_any = False
        for model in args.remove:
            if model == "diarization":
                if remove_diarization_cache():
                    print(f"✅ Removed diarization model")
                    removed_any = True
                else:
                    print(f"❌ Diarization model not found")
            elif model == "all":
                # Remove all models
                if remove_diarization_cache():
                    print(f"✅ Removed diarization model")
                    removed_any = True
                for size in WHISPER_SIZES:
                    if remove_whisper_cache(size):
                        print(f"✅ Removed Whisper {size} model")
                        removed_any = True
            elif model in WHISPER_SIZES:
                if remove_whisper_cache(model):
                    print(f"✅ Removed Whisper {model} model")
                    removed_any = True
                else:
                    print(f"❌ Whisper {model} model not found")
            else:
                print(f"❌ Unknown model '{model}'. Use 'diarization', whisper sizes, or 'all'")
        
        if removed_any:
            print("\nModels removed successfully!")
        else:
            print("\nNo models were removed.")
        sys.exit(0)

    # Resolve models to update/download
    models_to_process = []
    if "all" in args.update:
        models_to_process = ["diarization"] + WHISPER_SIZES
    else:
        for model in args.update:
            if model == "diarization":
                models_to_process.append("diarization")
            elif model in WHISPER_SIZES:
                models_to_process.append(model)
            else:
                print(f"Error: Unknown model '{model}'. Choose from: diarization, {', '.join(WHISPER_SIZES)}, or 'all'")
                sys.exit(1)

    print("Voice Journey - Model Setup/Update")
    print("=" * 40)
    print()
    list_models()

    # Check connectivity
    print("Checking internet connectivity...")
    if not check_network_connectivity():
        print("No internet connection detected. Connect to the internet and try again.")
        sys.exit(1)
    print("Connected.\n")

    # Check which models actually need updates (SHA mismatch)
    from models import check_model_updates, get_local_sha, _diarization_cache_path, _whisper_cache_path
    
    print("Checking for updates...")
    updates_info = check_model_updates()
    
    need_diarization = "diarization" in models_to_process and not is_diarization_cached()
    need_whisper = [m for m in models_to_process if m in WHISPER_SIZES and not is_whisper_cached(m)]
    
    # Only update if SHA mismatch detected
    update_diarization = False
    update_whisper = []
    
    if "diarization" in models_to_process and is_diarization_cached():
        if updates_info and 'diarization' in updates_info and updates_info['diarization'].get('has_update'):
            update_diarization = True
            print("⚠️  Diarization model has updates available")
        else:
            print("✅ Diarization model is already up to date")
    
    for size in [m for m in models_to_process if m in WHISPER_SIZES and is_whisper_cached(m)]:
        if updates_info and f'whisper_{size}' in updates_info and updates_info[f'whisper_{size}'].get('has_update'):
            update_whisper.append(size)
            print(f"⚠️  Whisper {size} model has updates available")
        else:
            print(f"✅ Whisper {size} model is already up to date")

    if not need_diarization and not need_whisper and not update_diarization and not update_whisper:
        print(f"\nAll requested models are already up to date.")
        sys.exit(0)

    # Handle updates (remove old versions first)
    if update_diarization:
        print("\nUpdating diarization model...")
        remove_diarization_cache()
        
    for size in update_whisper:
        print(f"\nUpdating Whisper {size} model...")
        remove_whisper_cache(size)

    # Download diarization
    diarization_ok = True
    if need_diarization or update_diarization:
        token = get_hf_token(args.token)
        diarization_ok = download_diarization(token)
    else:
        print("Diarization model already cached.")

    # Download whisper models
    whisper_results = {}
    whisper_sizes_to_download = need_whisper + update_whisper
    for size in set(whisper_sizes_to_download):  # Remove duplicates
        whisper_results[size] = download_whisper(size)

    # Summary
    print("\n" + "=" * 40)
    print("Setup Summary")
    print("=" * 40)
    if "diarization" in models_to_process:
        d_status = "✅" if is_diarization_cached() else "❌"
        print(f"  Diarization: {d_status}")
    
    for size in WHISPER_SIZES:
        if size in models_to_process:
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
