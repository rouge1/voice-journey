#!/usr/bin/env python3
"""Recursively scan a directory for WAV files and print all metadata tags."""

import os
import struct
import sys
import argparse
from datetime import timedelta

try:
    import mutagen
    from mutagen import File as MutagenFile
    from mutagen.wave import WAVE
except ImportError:
    print("Error: mutagen is required. Install with: pip install mutagen")
    sys.exit(1)

# Human-readable names for common RIFF INFO chunk tags
RIFF_INFO_NAMES = {
    "INAM": "Title",
    "IART": "Artist",
    "ICRD": "Date",
    "ICMT": "Comment",
    "IGNR": "Genre",
    "IPRD": "Album",
    "ITRK": "Track",
    "ISBJ": "Subject",
    "ISFT": "Software",
    "IENG": "Engineer",
    "ICOP": "Copyright",
    "IMED": "Medium",
    "ISRC": "Source",
    "ITCH": "Technician",
    "IKEY": "Keywords",
    "ILNG": "Language",
}


def get_wav_real_duration(path):
    """Compute WAV duration from actual file size, ignoring header's stated data size.

    SDR/intercept tools often write 0x00000000 or 0xFFFFFFFF as the data chunk
    size (streaming mode), causing mutagen to report 0:00 or thousands of hours.
    We find the real data start offset from the RIFF structure and divide actual
    bytes by bytes-per-second.
    """
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"RIFF":
                return None
            f.read(4)  # overall file size field (skip — often wrong too)
            if f.read(4) != b"WAVE":
                return None

            sample_rate = channels = bits_per_sample = None
            data_offset = None

            while True:
                chunk_id = f.read(4)
                if len(chunk_id) < 4:
                    break
                raw_size = f.read(4)
                if len(raw_size) < 4:
                    break
                chunk_size = struct.unpack("<I", raw_size)[0]

                if chunk_id == b"fmt ":
                    fmt = f.read(min(chunk_size, 16))
                    if len(fmt) >= 16:
                        _, channels, sample_rate, _, _, bits_per_sample = struct.unpack("<HHIIHH", fmt)
                    # skip any remaining fmt bytes
                    remaining = chunk_size - len(fmt)
                    if remaining > 0:
                        f.seek(remaining, 1)
                    if chunk_size % 2:
                        f.seek(1, 1)
                elif chunk_id == b"data":
                    data_offset = f.tell()
                    break
                else:
                    f.seek(chunk_size, 1)
                    if chunk_size % 2:
                        f.seek(1, 1)

        if None in (data_offset, sample_rate, channels, bits_per_sample):
            return None
        bytes_per_second = sample_rate * channels * (bits_per_sample // 8)
        if bytes_per_second == 0:
            return None
        actual_data_bytes = os.path.getsize(path) - data_offset
        return max(0, actual_data_bytes) / bytes_per_second
    except Exception:
        return None


def format_duration(seconds):
    if seconds is None:
        return "unknown"
    td = timedelta(seconds=int(seconds))
    total_seconds = int(td.total_seconds())
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def print_wav_info(path):
    print(path)
    try:
        audio = MutagenFile(path)
    except Exception as e:
        print(f"  Error reading file: {e}")
        print("  ---")
        return

    if audio is None:
        print("  (unrecognized format)")
        print("  ---")
        return

    # Audio properties
    info = getattr(audio, "info", None)
    if info:
        real_secs = get_wav_real_duration(path)
        duration = format_duration(real_secs if real_secs is not None else getattr(info, "length", None))
        sample_rate = getattr(info, "sample_rate", None)
        channels = getattr(info, "channels", None)
        print(f"  Duration:    {duration}")
        if sample_rate:
            print(f"  Sample rate: {sample_rate} Hz")
        if channels:
            print(f"  Channels:    {channels}")

    # Tags
    tags = audio.tags
    if not tags:
        print("  Tags:        (none)")
    else:
        print("  Tags:")
        for key, value in tags.items():
            # RIFF INFO values are lists; ID3 values have .text attribute
            if hasattr(value, "text"):
                display = ", ".join(str(v) for v in value.text)
            elif isinstance(value, list):
                display = ", ".join(str(v) for v in value)
            else:
                display = str(value)
            label = RIFF_INFO_NAMES.get(key, "")
            if label:
                print(f"    {key} ({label}):".ljust(24) + f" {display}")
            else:
                print(f"    {key}:".ljust(24) + f" {display}")

    print("  ---")


def scan_directory(directory):
    wav_files = []
    for root, _, files in os.walk(directory):
        for fname in files:
            if fname.lower().endswith(".wav"):
                wav_files.append(os.path.join(root, fname))

    wav_files.sort()

    if not wav_files:
        print(f"No WAV files found in: {directory}")
        return

    print(f"Found {len(wav_files)} WAV file(s) in {directory}\n")
    for path in wav_files:
        print_wav_info(path)


def main():
    parser = argparse.ArgumentParser(
        description="Print metadata tags for a WAV file or recursively scan a directory."
    )
    parser.add_argument("path", help="WAV file or directory to scan")
    args = parser.parse_args()

    if os.path.isfile(args.path):
        print_wav_info(args.path)
    elif os.path.isdir(args.path):
        scan_directory(args.path)
    else:
        print(f"Error: not a file or directory: {args.path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
