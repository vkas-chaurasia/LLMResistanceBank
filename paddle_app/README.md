# PaddleOCR PDF Extraction App

This application uses [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) (specifically the `PaddleOCR-VL` model) to extract high-quality structured content from PDF documents. It is designed to be part of a larger multi-model pipeline.

## Features

*   **PDF to Markdown**: Converts PDF pages into formatted Markdown.
*   **Custom Filtering**: Generates a "custom" Markdown version that:
    *   Removes "References" sections automatically.
    *   Filters out redundant images (headers/footers) while keeping text headers/footers.
    *   Maintains original document reading order.
*   **Structured Output**: Produces JSON layout data, extracted images, and per-page Markdown files.
*   **Combined Output**: Merges all pages into single `combined.md` and `combined_custom.md` files.

## Project Structure

```
paddle_app/
├── extract_tables_opensource.py  # Core PaddleOCRGenerator class
├── requirements.txt              # Dependencies
├── paddlepaddle/                 # Virtual environment (ignored in git)
├── tests/                        # App-specific tests
│   └── outputs/                  # Generated artifacts
└── README.md                     # This file
```

## Setup & usage

### Prerequisites

```bash
module load uv python/3.12.4 cuda/12.6.3
```

### Installation

1.  Create and activate the environment:
    ```bash
    uv venv paddlepaddle --python 3.12.4
    source paddlepaddle/bin/activate
    ```

2.  Install dependencies:
    ```bash
    uv pip install -r requirements.txt
    ```

### Running Tests

To verify the pipeline on sample data (located in `../data/pdf`):

```bash
# From the project root
pytest paddle_app/tests/test_paddle.py
```

### Python API

```python
from paddle_app.extract_tables_opensource import PaddleOCRGenerator

generator = PaddleOCRGenerator(use_gpu=True)
summary = generator.generate(
    input_file="../data/pdf/sample.pdf",
    output_dir="../outputs/sample_result"
)
```
