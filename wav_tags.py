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


def read_riff_info(path):
    """Parse RIFF LIST/INFO chunk directly. Returns dict of tag→value string."""
    tags = {}
    try:
        with open(path, "rb") as f:
            header = f.read(12)
            if len(header) < 12:
                return tags
            riff_id, _, wave_id = struct.unpack_from("<4sI4s", header)
            if riff_id != b"RIFF" or wave_id != b"WAVE":
                return tags
            while True:
                chunk_hdr = f.read(8)
                if len(chunk_hdr) < 8:
                    break
                chunk_id, chunk_size = struct.unpack_from("<4sI", chunk_hdr)
                chunk_data = f.read(chunk_size)
                if chunk_size % 2:
                    f.read(1)
                if chunk_id == b"LIST" and len(chunk_data) >= 4 and chunk_data[:4] == b"INFO":
                    offset = 4
                    while offset + 8 <= len(chunk_data):
                        sub_id = chunk_data[offset:offset+4].decode("latin-1")
                        sub_size = struct.unpack_from("<I", chunk_data, offset + 4)[0]
                        offset += 8
                        value = chunk_data[offset:offset+sub_size]
                        offset += sub_size
                        if sub_size % 2:
                            offset += 1
                        value = value.rstrip(b"\x00").decode("latin-1", errors="replace").strip()
                        if value:
                            tags[sub_id] = value
    except (OSError, struct.error):
        pass
    return tags


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
    if seconds < 1.0:
        return f"0:00 ({seconds:.2f}s)"
    td = timedelta(seconds=round(seconds))
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

    # Tags — mutagen (ID3) + raw RIFF INFO fallback
    tags = audio.tags
    riff_tags = read_riff_info(path)

    if not tags and not riff_tags:
        print("  Tags:        (none)")
    else:
        print("  Tags:")
        if tags:
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
        for key, value in riff_tags.items():
            if tags and key in tags:
                continue  # already shown via mutagen
            label = RIFF_INFO_NAMES.get(key, "")
            if label:
                print(f"    {key} ({label}):".ljust(24) + f" {value}")
            else:
                print(f"    {key}:".ljust(24) + f" {value}")

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
