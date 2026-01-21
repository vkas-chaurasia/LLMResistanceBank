# LLMResistanceBank

A multi-model AI pipeline for analyzing antibiotic resistance literature.

This repository hosts a collection of tools and models designed to extract, process, and analyze scientific documents related to antibiotic resistance. The project is structured to support multiple components, from robust OCR extraction to LLM-based table classification.

## Project Structure

```
LLMResistanceBank/
├── paddle_app/            # OCR extraction using PaddleOCR-VL
├── pymupdf_app/           # PDF table extraction using PyMuPDF
├── llm_classification/    # LLM-based table complexity classification
├── gemini_app/            # Outputs and tests related to Gemini models
├── utils/                 # Shared utilities (PDF processing, markdown cleaning)
├── shared/                # Shared code across modules
├── data/                  # Source documents (PDFs, etc.)
└── README.md              # This file
```

## Components

### 1. Paddle App (`paddle_app/`)
A high-performance OCR engine using `PaddleOCR-VL` to convert PDF documents into structured Markdown, JSON, and images. It features advanced layout analysis and custom content filtering (e.g., reference removal).

[View Paddle App Documentation](./paddle_app/README.md)

### 2. PyMuPDF App (`pymupdf_app/`)
A lightweight table extraction tool using `PyMuPDF` (Fitz) to extract tables from PDF files directly from the document metadata and drawing commands.

### 3. LLM Classification (`llm_classification/`)
A module dedicated to classifying the complexity of extracted tables using Large Language Models (LLMs) such as OpenAI and Gemma. It includes scripts for gathering tables, analyzing their structural/semantic complexity, and generating reports.

### 4. Gemini App (`gemini_app/`)
Contains test outputs and processing results from Google's Gemini models.

## Getting Started

### Prerequisites

*   **OS**: Linux
*   **Python**: 3.12.4
*   **CUDA**: 12.6.3 (for GPU acceleration in OCR)

### Environment Setup

This project uses `uv` for fast package management.

1.  Load required modules:
    ```bash
    module load uv python/3.12.4 cuda/12.6.3
    ```

2.  Navigate to the specific application directory (e.g., `paddle_app`, `pymupdf_app`, or `llm_classification`) to set up its environment as described in its respective directory.

## Usage

Please refer to the detailed READMEs within each application subdirectory for specific usage instructions.

*   [Paddle App Guide](./paddle_app/README.md)

## Shared Resources

*   **`shared/`**: Contains core modules for embedding generation (`embedding_model.py`) and vector database management (`vector_db.py`).
*   **`utils/`**: Helper scripts for PDF and Markdown processing.
    *   `pdf_utils.py`: PDF page-to-image conversion.
    *   `clean_md.py`: Metrics and cleaning tools for Markdown content.
    *   `convert_pdfs.py`: Script for batch PDF conversion.
