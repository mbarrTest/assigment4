#!/usr/bin/env python3
# validate_pipe.py — find malformed lines in .pipe files (expected: message|OFF or message|NOT)

import sys
import re
from pathlib import Path

LABELS = {"OFF", "NOT"}
# Match anything, last '|' then OFF/NOT (case-insensitive), optional trailing spaces
PAT = re.compile(r"^(?P<text>.*)\|(?P<label>OFF|NOT)\s*$", re.IGNORECASE)

def check_file(path):
    bad = []
    empty = 0
    total = 0

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln, raw in enumerate(f, start=1):
            total += 1
            line = raw.rstrip("\n\r")
            if not line.strip():
                empty += 1
                continue

            m = PAT.match(line)
            if m:
                # Normalize label to uppercase; also check empty text (optional)
                text = m.group("text").strip()
                label = m.group("label").upper()
                if label not in LABELS:
                    bad.append((ln, "Label not in {OFF,NOT}", line))
                elif text == "":
                    # not fatal for parsing, but often unintended → mark as warning
                    bad.append((ln, "Empty message text before '|'", line))
            else:
                # Figure out why it failed for clearer diagnostics
                reason = []
                if "|" not in line:
                    reason.append("No '|' separator found")
                else:
                    # Has a pipe but not ending with OFF/NOT
                    rhs = line.rsplit("|", 1)[-1].strip().upper()
                    if rhs not in LABELS:
                        reason.append(f"Last token after '|' is not OFF/NOT (got '{rhs or '<empty>'}')")
                    else:
                        reason.append("Regex mismatch (unexpected format)")

                bad.append((ln, "; ".join(reason), line))

    return total, empty, bad

def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_pipe.py <file1.pipe> [file2.pipe ...]")
        sys.exit(1)

    exit_code = 0
    for p in sys.argv[1:]:
        path = Path(p)
        if not path.exists():
            print(f"❌ File not found: {path}")
            exit_code = 2
            continue

        total, empty, bad = check_file(path)
        print(f"\n📄 {path} — lines: {total} (empty: {empty})")
        if not bad:
            print("✅ All non-empty lines look valid: message|OFF or message|NOT")
            continue

        print(f"⚠️ Found {len(bad)} malformed line(s):")
        for ln, reason, line in bad[:50]:
            print(f"  - Line {ln}: {reason}\n    {line}")

        if len(bad) > 50:
            print(f"  ... and {len(bad) - 50} more. (Showing first 50)")

        # Non-zero exit if any bad lines
        exit_code = 3

    sys.exit(exit_code)

if __name__ == "__main__":
    main()
