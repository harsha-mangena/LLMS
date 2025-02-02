# PDF Question Answering System

A simple command-line tool for querying PDF documents using embeddings and semantic search.

## Installation

Create a virtual environment and install dependencies:

```bash
# Create virtual environment
python -m venv env

# Activate environment
# On macOS/Linux:
source env/bin/activate
# On Windows:
env\Scripts\activate

# Install dependencies
pip install numpy faiss-cpu tiktoken PyPDF2 sentence-transformers tqdm
```

## Usage

### Basic Commands

Process a PDF:
```bash
python vector.py -i document.pdf
```

Query the document:
```bash
python vector.py -q "What is this document about?"
```

Clear existing data and process new document:
```bash
python vector.py --clear -i document.pdf
```

### Command Options

```bash
-i, --input       PDF file to process
-q, --query       Query to search
--verbose         Enable detailed logging
--clear          Clear existing vector store
```

## Examples

Process a document with verbose output:
```bash
python vector.py --clear -i research.pdf --verbose
```

Search the document:
```bash
python vector.py -q "What were the main findings?"
```

## Requirements

- Python 3.7 or higher
- 4GB+ RAM recommended
- Required libraries:
  - faiss-cpu
  - numpy
  - tiktoken 
  - PyPDF2
  - sentence-transformers
  - tqdm

## Features

- PDF text extraction
- Smart text chunking
- Vector similarity search
- Semantic embeddings
- Command line interface

## License

MIT License
