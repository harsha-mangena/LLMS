#!/usr/bin/env python3
import sys
import os
import logging
import warnings
import argparse
import pickle
import gc
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import numpy as np
import faiss
import tiktoken
from tqdm import tqdm
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import fickling

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")


@dataclass
class DocumentChunk:
    """
    A chunk of text from a document along with its metadata and an optional embedding.
    """
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None


class VectorStore:
    """
    Stores document embeddings in a FAISS index and manages saving/loading of vector data.
    """
    def __init__(self, dimension: int = 384, index_path: str = "vectors") -> None:
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.chunks: List[DocumentChunk] = []
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)

    def add_chunks(self, chunks: List[DocumentChunk], encoder: SentenceTransformer) -> int:
        """
        Encodes and adds document chunks to the vector store in batches.

        Args:
            chunks: List of document chunks.
            encoder: Sentence transformer model for encoding texts.

        Returns:
            Number of chunks added.
        """
        if not chunks:
            return 0

        batch_size = 8
        new_chunks = []
        new_embeddings = []

        for i in tqdm(range(0, len(chunks), batch_size), desc="Processing chunks"):
            batch = chunks[i : i + batch_size]
            texts = [chunk.text for chunk in batch]
            embeddings = encoder.encode(texts, convert_to_numpy=True)
            new_chunks.extend(batch)
            new_embeddings.extend(embeddings)
            gc.collect()

        if new_embeddings:
            embeddings_array = np.array(new_embeddings, dtype="float32")
            self.index.add(embeddings_array)
            self.chunks.extend(new_chunks)
            self._save_vectors()

        return len(new_chunks)

    def _save_vectors(self) -> None:
        """
        Saves the FAISS index and chunks to disk.
        """
        try:
            index_file = self.index_path / "faiss.index"
            chunks_file = self.index_path / "chunks.pkl"
            faiss.write_index(self.index, str(index_file))
            with open(chunks_file, "wb") as f:
                pickle.dump(self.chunks, f)
            logger.info(f"Saved {len(self.chunks)} chunks to '{self.index_path}'.")
        except Exception as e:
            logger.error(f"Error saving vectors: {e}")
            raise

    def load_vectors(self) -> None:
        """
        Loads the FAISS index and associated chunks from disk.
        """
        try:
            index_file = self.index_path / "faiss.index"
            chunks_file = self.index_path / "chunks.pkl"
            if index_file.exists() and chunks_file.exists():
                self.index = faiss.read_index(str(index_file))
                with open(chunks_file, "rb") as f:
                    self.chunks = fickling.load(f)
                logger.info(f"Loaded {len(self.chunks)} chunks from '{self.index_path}'.")
        except Exception as e:
            logger.error(f"Error loading vectors: {e}")
            # Reinitialize if loading fails
            self.index = faiss.IndexFlatL2(self.dimension)
            self.chunks = []

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[DocumentChunk]:
        """
        Searches for the top_k most similar document chunks based on the query embedding.

        Args:
            query_embedding: The query embedding vector.
            top_k: Number of top results to return.

        Returns:
            List of matching DocumentChunk objects.
        """
        if not self.chunks:
            return []
        query_embedding = query_embedding.astype("float32")
        distances, indices = self.index.search(query_embedding.reshape(1, -1), min(top_k, len(self.chunks)))
        return [self.chunks[i] for i in indices[0] if i != -1]


class DocumentProcessor:
    """
    Processes PDF files by extracting text and splitting it into manageable chunks.
    """
    def __init__(self, chunk_size: int = 512) -> None:
        self.chunk_size = chunk_size
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def process_file(self, file_path: str) -> List[DocumentChunk]:
        """
        Processes a PDF file to extract text and create chunks.

        Args:
            file_path: Path to the PDF file.

        Returns:
            List of DocumentChunk objects.
        """
        try:
            text = self._extract_text(file_path)
            if not text.strip():
                logger.warning(f"No text content found in '{file_path}'.")
                return []
            return self._create_chunks(text, {"source": file_path})
        except Exception as e:
            logger.error(f"Error processing file '{file_path}': {e}")
            raise

    def _extract_text(self, file_path: str) -> str:
        """
        Extracts text from each page of the PDF.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Extracted text as a single string.
        """
        try:
            text_chunks = []
            with open(file_path, "rb") as file:
                pdf = PdfReader(file, strict=False)
                logger.info(f"Processing PDF '{file_path}' with {len(pdf.pages)} pages.")
                for i, page in enumerate(pdf.pages, start=1):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_chunks.append(page_text)
                        logger.debug(f"Processed page {i}.")
                    except Exception as page_err:
                        logger.warning(f"Error on page {i}: {page_err}")
                        continue
            return "\n".join(text_chunks)
        except Exception as e:
            logger.error(f"Error extracting text from '{file_path}': {e}")
            raise

    def _create_chunks(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Splits the text into chunks based on the configured token size.

        Args:
            text: The text to be split.
            metadata: Associated metadata for the text.

        Returns:
            A list of DocumentChunk objects.
        """
        tokens = self.tokenizer.encode(text)
        return [
            DocumentChunk(
                text=self.tokenizer.decode(tokens[i : i + self.chunk_size]),
                metadata=metadata.copy()
            )
            for i in range(0, len(tokens), self.chunk_size)
            if self.tokenizer.decode(tokens[i : i + self.chunk_size]).strip()
        ]


class RAGSystem:
    """
    Retrieval-Augmented Generation (RAG) system that processes documents,
    stores their vector embeddings, and supports similarity-based querying.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", vector_dir: str = "vectors") -> None:
        self.document_processor = DocumentProcessor()
        self.encoder = SentenceTransformer(model_name)
        dimension = self.encoder.get_sentence_embedding_dimension()
        self.vector_store = VectorStore(dimension, vector_dir)
        self.vector_store.load_vectors()

    def add_document(self, file_path: str) -> int:
        """
        Processes a document and adds its chunks to the vector store.

        Args:
            file_path: Path to the document (PDF) file.

        Returns:
            Number of chunks added.
        """
        chunks = self.document_processor.process_file(file_path)
        num_added = self.vector_store.add_chunks(chunks, self.encoder)
        return num_added

    def query(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Queries the vector store for documents similar to the given query.

        Args:
            query: Query string.
            top_k: Number of top results to return.

        Returns:
            A list of dictionaries with chunk text and metadata.
        """
        query_embedding = self.encoder.encode(query, convert_to_numpy=True)
        relevant_chunks = self.vector_store.search(query_embedding, top_k)
        return [
            {
                "text": chunk.text,
                "metadata": chunk.metadata,
                "source": chunk.metadata.get("source", "Unknown")
            }
            for chunk in relevant_chunks
        ]


def main() -> int:
    """
    Main entry point for the RAG system command-line interface.
    """
    parser = argparse.ArgumentParser(description="Retrieval-Augmented Generation (RAG) System")
    parser.add_argument("--input", "-i", type=str, help="Path to the PDF file to process")
    parser.add_argument("--query", "-q", type=str, help="Query string for search")
    parser.add_argument(
        "--vector_dir", "-v", type=str, default="vectors", help="Directory for storing vector index"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--clear", action="store_true", help="Clear existing vector store")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    try:
        vector_dir_path = Path(args.vector_dir)
        if args.clear and vector_dir_path.exists():
            shutil.rmtree(vector_dir_path)
            logger.info("Cleared existing vector store.")

        rag = RAGSystem(vector_dir=str(vector_dir_path))

        if args.input:
            file_path = Path(args.input).resolve()
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return 1
            logger.info(f"Processing file: {file_path}")
            num_chunks = rag.add_document(str(file_path))
            logger.info(f"Successfully processed file and added {num_chunks} chunks.")

        if args.query:
            results = rag.query(args.query)
            header = (
                f"\n{'=' * 80}\n"
                f"Results for query: {args.query}\n"
                f"{'=' * 80}"
            )
            print(header)
            if not results:
                print("No results found.")
            else:
                for idx, result in enumerate(results, start=1):
                    separator = f"\n[{idx}] {'=' * 40}"
                    print(separator)
                    print(f"Source: {result['source']}")
                    print(f"Text:\n{result['text']}\n")

    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
