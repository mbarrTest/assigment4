import re
from pathlib import Path

# Define file paths
input_files = {
    "train": "data/train.pipe",
    "dev": "data/dev.pipe",
    "test": "data/test.pipe"
}

output_files = {
    "train": "data/ftrain.pipe",
    "dev": "data/fdev.pipe",
    "test": "data/ftest.pipe"
}

# Regex patterns
user_pattern = re.compile(r"@\w+")
emoji_pattern = re.compile(
    "["                             # Start of character class
    "\U0001F600-\U0001F64F"         # Emoticons
    "\U0001F300-\U0001F5FF"         # Symbols & pictographs
    "\U0001F680-\U0001F6FF"         # Transport & map symbols
    "\U0001F1E0-\U0001F1FF"         # Flags
    "\U00002700-\U000027BF"         # Dingbats
    "\U0001F900-\U0001F9FF"         # Supplemental symbols & pictographs
    "\U00002600-\U000026FF"         # Misc symbols
    "]+", 
    flags=re.UNICODE
)

def clean_text(text):
    """Remove @USER mentions and emojis from text."""
    text = user_pattern.sub("", text)
    text = emoji_pattern.sub("", text)
    # Optionally remove extra spaces left behind
    return re.sub(r"\s+", " ", text).strip()

def process_file(in_path, out_path):
    """Clean each line in a file and save the result."""
    with open(in_path, "r", encoding="utf-8") as infile, open(out_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            cleaned_line = clean_text(line)
            outfile.write(cleaned_line + "\n")

def main():
    for split in ["train", "dev", "test"]:
        in_path = Path(input_files[split])
        out_path = Path(output_files[split])
        process_file(in_path, out_path)
        print(f" Cleaned {in_path} → {out_path}")

if __name__ == "__main__":
    main()
