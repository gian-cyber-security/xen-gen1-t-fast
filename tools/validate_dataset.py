#!/usr/bin/env python3
"""Read-only validator for the repository's real dataset manifest."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".webm", ".mov", ".avi", ".mkv"}

def err(errors, line, msg): errors.append(f"line {line}: {msg}")

def validate_text(path, errors, warnings):
    seen = set(); count = 0
    with path.open(encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            if not raw.strip(): continue
            count += 1
            try: row = json.loads(raw)
            except json.JSONDecodeError as e:
                err(errors, line_no, f"invalid JSON: {e.msg}"); continue
            if not isinstance(row, dict): err(errors, line_no, "record must be an object"); continue
            ins, resp = row.get("instruction"), row.get("response")
            if not isinstance(ins, str) or not ins.strip(): err(errors, line_no, "empty or missing instruction")
            if not isinstance(resp, str) or not resp.strip(): err(errors, line_no, "empty or missing response")
            key = json.dumps(row, sort_keys=True, ensure_ascii=False)
            if key in seen: warnings.append(f"line {line_no}: duplicate record")
            seen.add(key)
    return count

def validate_media(path, kind, errors, warnings):
    seen = set(); count = 0; allowed = IMAGE_EXTS if kind == "image" else VIDEO_EXTS
    try:
        from PIL import Image
    except Exception: Image = None
    try:
        import imageio.v3 as iio
    except Exception: iio = None
    with path.open(encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            if not raw.strip(): continue
            count += 1
            try: row = json.loads(raw)
            except json.JSONDecodeError as e:
                err(errors, line_no, f"invalid JSON: {e.msg}"); continue
            if not isinstance(row, dict): err(errors, line_no, "record must be an object"); continue
            field = "image" if kind == "image" else "video"
            ref, caption = row.get(field), row.get("caption")
            if not isinstance(ref, str) or not ref.strip(): err(errors, line_no, f"missing {field} path")
            if not isinstance(caption, str) or not caption.strip(): err(errors, line_no, "empty or missing caption")
            if not isinstance(ref, str): continue
            media = (path.parent / ref).resolve()
            if not media.exists(): err(errors, line_no, f"missing file: {ref}"); continue
            if media.suffix.lower() not in allowed: err(errors, line_no, f"unsupported format: {media.suffix}")
            if str(media) in seen: warnings.append(f"line {line_no}: duplicate media reference")
            seen.add(str(media))
            try:
                if kind == "image":
                    if Image is None: warnings.append(f"line {line_no}: Pillow unavailable; image integrity not checked")
                    else:
                        with Image.open(media) as im:
                            im.verify()
                        with Image.open(media) as im:
                            if im.width < 2 or im.height < 2: err(errors, line_no, "image dimensions are too small")
                else:
                    if iio is None: warnings.append(f"line {line_no}: imageio unavailable; video integrity not checked")
                    else:
                        meta = iio.immeta(media, plugin="ffmpeg")
                        if not meta: warnings.append(f"line {line_no}: video metadata unavailable")
            except Exception as e: err(errors, line_no, f"corrupt or unreadable media: {e}")
    return count

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["text", "image", "video"], required=True)
    ap.add_argument("--data", required=True)
    args = ap.parse_args(); path = Path(args.data)
    errors, warnings = [], []
    if not path.exists():
        print(f"DATASET VALIDATION: BLOCKED\nmissing manifest: {path}")
        return 2
    count = validate_text(path, errors, warnings) if args.kind == "text" else validate_media(path, args.kind, errors, warnings)
    print(f"records={count}")
    print(f"errors={len(errors)} warnings={len(warnings)}")
    for x in errors: print("ERROR: " + x)
    for x in warnings: print("WARNING: " + x)
    print("DATASET VALIDATION: " + ("PASS" if not errors and count else "FAIL"))
    return 0 if not errors and count else 1
if __name__ == "__main__": sys.exit(main())
