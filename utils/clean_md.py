import re
from pathlib import Path

def clean_markdown(input_path, output_path=None):
    """
    Cleans a Markdown file by:
    - Removing the 'References' section (any heading level)
    - Trimming excessive blank lines and trailing spaces

    Parameters:
        input_path (str | Path): Path to the input .md file
        output_path (str | Path | None): Optional output path.
                                         Defaults to '<input>_clean.md'

    Returns:
        Path: Path to the cleaned output file
    """
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else input_path.with_name(f"{input_path.stem}_clean.md")

    if not input_path.exists():
        print(f"Warning: Input file {input_path} not found. Skipping cleaning.")
        return None

    text = input_path.read_text(encoding="utf-8")

    # Remove references (case insensitive)
    text = re.split(r"^#{1,6}\s*References\b.*", text, flags=re.IGNORECASE | re.MULTILINE)[0]
    
    # Trim trailing spaces on lines
    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
    
    # Reduce multiple newlines to max 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    text = text.strip() + "\n"

    # Write output
    output_path.write_text(text, encoding="utf-8")

    print(f"Cleaned file saved to: {output_path}")
    return output_path
