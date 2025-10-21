#!/usr/bin/env python3
# cleaner.py — Converts TSV with tab before OFF/NOT to pipe-separated format (|)

import sys
from pathlib import Path

def convert_tsv_to_pipe(input_path, output_path=None):
    """
    Reads a TSV file and replaces the tab before 'OFF' or 'NOT' with a pipe '|'.
    Creates a new .pipe file or a user-specified output file.
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_suffix(".pipe")

    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            # Replace the last TAB before OFF or NOT with a pipe
            new_line = line.rstrip("\n").replace("\tOFF", "|OFF").replace("\tNOT", "|NOT")
            fout.write(new_line + "\n")

    print(f"✅ Cleaned: {input_path} → {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cleaner.py <input_file> [output_file]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    convert_tsv_to_pipe(input_file, output_file)
