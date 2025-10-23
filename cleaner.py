#!/usr/bin/env python3
# cleaner.py — Converts TSV to clean .pipe files (text|label), removing stray pipes in text

import sys
from pathlib import Path
import re

def convert_tsv_to_pipe(input_path, output_path=None):
    """
    Converts a TSV file to a clean .pipe format:
      - Removes all extra '|' inside the message text.
      - Ensures the final separator before OFF/NOT is a single '|'.
    Example:
      "Hello | world\tOFF" -> "Hello  world|OFF"
      "Text with\ttabs\tNOT" -> "Text with tabs|NOT"
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_suffix(".pipe")

    pattern = re.compile(r"^(.*?)(?:\t|\|)\s*(OFF|NOT)\s*$", re.IGNORECASE)

    cleaned, skipped = 0, 0
    with open(input_path, "r", encoding="utf-8", errors="ignore") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            match = pattern.match(line)
            if match:
                text, label = match.groups()
                # Remove any pipes or extra whitespace in text
                text = text.replace("|", " ").strip()
                fout.write(f"{text}|{label.upper()}\n")
                cleaned += 1
            else:
                # If line doesn't match, try to recover tab-separated lines
                if "\t" in line:
                    parts = line.rsplit("\t", 1)
                    text, label = parts[0].replace("|", " ").strip(), parts[1].strip().upper()
                    if label in {"OFF", "NOT"}:
                        fout.write(f"{text}|{label}\n")
                        cleaned += 1
                    else:
                        skipped += 1
                else:
                    skipped += 1

    print(f"✅ Cleaned: {input_path} → {output_path}")
    print(f"   Lines processed: {cleaned}")
    if skipped:
        print(f"   ⚠️ Skipped {skipped} malformed lines.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cleaner.py <input_file> [output_file]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    convert_tsv_to_pipe(input_file, output_file)
