import os

CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")

DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"

WHISPER_SIZES = ["tiny", "small", "medium", "large", "turbo"]

WHISPER_SIZE_INFO = {
    "tiny": "Fastest, least accurate (~39 MB)",
    "small": "Balanced speed/accuracy (~484 MB)",
    "medium": "Default, good quality (~1.5 GB)",
    "large": "Best accuracy, high memory (~2.9 GB)",
    "turbo": "Fast & accurate, v3 optimized (~809 MB)",
}


def _get_whisper_model_name(size):
    """Map user-facing size names to actual model names."""
    if size == "large":
        return "large-v3"
    return size


def _diarization_cache_path():
    return os.path.join(CACHE_DIR, "models--pyannote--speaker-diarization-3.1")


def _whisper_cache_path(size):
    model_name = _get_whisper_model_name(size)
    if model_name == "turbo":
        return os.path.join(CACHE_DIR, "models--mobiuslabsgmbh--faster-whisper-large-v3-turbo")
    return os.path.join(CACHE_DIR, f"models--Systran--faster-whisper-{model_name}")


def is_diarization_cached():
    return os.path.exists(_diarization_cache_path())


def is_whisper_cached(size):
    return os.path.exists(_whisper_cache_path(size))


def remove_diarization_cache():
    """Remove the cached diarization model."""
    path = _diarization_cache_path()
    if os.path.exists(path):
        import shutil
        shutil.rmtree(path)
        return True
    return False


def remove_whisper_cache(size):
    """Remove the cached whisper model for the given size."""
    path = _whisper_cache_path(size)
    if os.path.exists(path):
        import shutil
        shutil.rmtree(path)
        return True
    return False


def get_local_sha(cache_path):
    """Get the SHA of the locally cached model."""
    refs_file = os.path.join(cache_path, 'refs', 'main')
    if os.path.exists(refs_file):
        with open(refs_file, 'r') as f:
            return f.read().strip()
    return None


def check_model_updates():
    """Check for available model updates and return status info."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        updates_info = {}
        
        # Check diarization model
        try:
            remote_info = api.model_info(DIARIZATION_MODEL)
            local_sha = get_local_sha(_diarization_cache_path())
            updates_info['diarization'] = {
                'cached': local_sha is not None,
                'local_sha': local_sha,
                'remote_sha': remote_info.sha,
                'has_update': local_sha != remote_info.sha if local_sha else False,
                'last_modified': remote_info.last_modified
            }
        except Exception as e:
            updates_info['diarization'] = {'error': str(e)}
        
        # Check Whisper models
        for size in WHISPER_SIZES:
            try:
                actual_model_name = _get_whisper_model_name(size)
                model_name = f'Systran/faster-whisper-{actual_model_name}' if actual_model_name != 'turbo' else 'mobiuslabsgmbh/faster-whisper-large-v3-turbo'
                remote_info = api.model_info(model_name)
                local_sha = get_local_sha(_whisper_cache_path(size))
                updates_info[f'whisper_{size}'] = {
                    'cached': local_sha is not None,
                    'local_sha': local_sha,
                    'remote_sha': remote_info.sha,
                    'has_update': local_sha != remote_info.sha if local_sha else False,
                    'last_modified': remote_info.last_modified
                }
            except Exception as e:
                updates_info[f'whisper_{size}'] = {'error': str(e)}
                
        return updates_info
    except ImportError:
        # If huggingface_hub not available, just return cache status
        return None


def list_models_with_updates():
    """Enhanced list function that shows update status."""
    updates_info = check_model_updates()
    
    print("Voice Journey - Model Status & Updates")
    print("=" * 50)
    print()
    
    # Diarization
    if updates_info and 'diarization' in updates_info:
        info = updates_info['diarization']
        if 'error' in info:
            status = f"❌ Error: {info['error']}"
        else:
            cached_status = "✅ Cached" if info['cached'] else "❌ Not cached"
            update_status = " (⚠️ Update available)" if info['has_update'] else " (✅ Up to date)" if info['cached'] else ""
            status = f"{cached_status}{update_status}"
    else:
        status = "✅ Cached" if is_diarization_cached() else "❌ Not cached"
    
    print(f"Speaker Diarization: {status}")
    print(f"  {DIARIZATION_MODEL}")
    print()
    
    # Whisper models
    print("Whisper Transcription Models:")
    for size in WHISPER_SIZES:
        if updates_info and f'whisper_{size}' in updates_info:
            info = updates_info[f'whisper_{size}']
            if 'error' in info:
                status = f"❌ Error: {info['error']}"
            else:
                cached_status = "✅ Cached" if info['cached'] else "❌ Not cached"
                update_status = " (⚠️ Update available)" if info['has_update'] else " (✅ Up to date)" if info['cached'] else ""
                status = f"{cached_status}{update_status}"
        else:
            status = "✅ Cached" if is_whisper_cached(size) else "❌ Not cached"
        
        print(f"  {size:<8} - {WHISPER_SIZE_INFO[size]:<35} {status}")
    print()


def list_models():
    """Legacy function for backward compatibility."""
    list_models_with_updates()
