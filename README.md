# LLMResistanceBank

A multi-model AI pipeline for analyzing antibiotic resistance literature.

This repository hosts a collection of tools and models designed to extract, process, and analyze scientific documents related to antibiotic resistance. The project is structured to support multiple components, starting with a robust OCR extraction pipeline.

## Project Structure

```
LLMResistanceBank/
├── paddle_app/            # PaddleOCR-based extraction pipeline
│   ├── README.md          # Documentation for the OCR app
│   ├── extract_tables...  # Core extraction logic
│   └── tests/             # OCR-specific tests
├── utils/                 # Shared utilities (PDF processing, markdown cleaning)
├── data/                  # Data directory (PDFs, etc.)
└── README.md              # This file
```

## Components

### 1. Paddle App (`paddle_app/`)
A high-performance OCR engine using `PaddleOCR-VL` to convert PDF documents into structured Markdown, JSON, and images. It features advanced layout analysis and custom content filtering (e.g., reference removal).

[View Paddle App Documentation](./paddle_app/README.md)

## Getting Started

### Prerequisites

*   **OS**: Linux
*   **Python**: 3.12.4
*   **CUDA**: 12.6.3 (for GPU acceleration)

### Environment Setup

This project uses `uv` for fast package management.

1.  Load required modules:
    ```bash
    module load uv python/3.12.4 cuda/12.6.3
    ```

2.  Navigate to the specific application directory (e.g., `paddle_app`) to set up its environment as described in its README.

## Usage

Please refer to the detailed READMEs within each application subdirectory for specific usage instructions.

*   [Paddle App Guide](./paddle_app/README.md)

## Shared Utilities

The `utils/` directory contains helper scripts shared across apps:
*   `pdf_utils.py`: PDF page-to-image conversion.
*   `clean_md.py`: Metrics and cleaning tools for Markdown content.
