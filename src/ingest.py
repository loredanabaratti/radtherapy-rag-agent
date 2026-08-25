import argparse
import hashlib
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from src.config import (
    CHROMA_COLLECTION,
    CHROMA_DB_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    RAW_DATA_DIR,
)

def parse_document(file_path: Path) -> tuple[dict[str, str], str]:
    """ Reads a text file and returns its metadata and content."""

    lines = file_path.read_text(encoding="utf-8").splitlines()
    metadata = {
        "title": file_path.stem,
        "source": "Unknown",
        "url": "",
        "file_name": file_path.name,
    }

    content_starts_index = 0
    for i, line in enumerate(lines):
        if line.startswith("Title:"):
            metadata["title"] = line.removeprefix("Title:").strip()
        elif line.startswith("Source:"):
            metadata["source"] = line.removeprefix("Source:").strip()
        elif line.startswith("URL:"):
            metadata["url"] = line.removeprefix("URL:").strip()
        elif line.strip() == "":
            content_starts_index = i + 1
            break

    content = "\n".join(lines[content_starts_index:]).strip()

    if not content:
        raise ValueError(f"No content found in file: {file_path}")
    
    return metadata, content

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """ Splits the text into chunks of specified size with overlap."""

    if chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be smaller than chunk size.")
    
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()

        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)

        if end >= len(text):
            break  # Exit if we've reached the end of the text

        start = end - chunk_overlap

    return chunks

def create_chunk_id(file_name: str, chunk_index: int, chunk: str) -> str:
    """ Creates a unique ID for each chunk """

    content_hash = hashlib.md5(chunk.encode("utf-8")).hexdigest()[:10]  # Shorten the hash for brevity
    return f"{file_name}-chunk-{chunk_index}-{content_hash}"

def ingest_documents(reset: bool = False) -> int:
    """ Ingests documents from the RAW_DATA_DIR into ChromaDB. """

    text_files = sorted(RAW_DATA_DIR.glob("*.txt"))

    if not text_files:
        raise FileNotFoundError(
            f"No text files found in the directory: {RAW_DATA_DIR}. Please add some .txt files to ingest."
        )
    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

    if reset:
        try:
            client.delete_collection(name=CHROMA_COLLECTION)
            print(f"Collection '{CHROMA_COLLECTION}' deleted successfully.")
        except (ValueError, chromadb.errors.NotFoundError):
            print(f"Collection '{CHROMA_COLLECTION}' does not exist. No deletion performed.")

    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    documents = []
    metadatas = []
    ids = []

    for file_path in text_files:
        try:
            metadata, content = parse_document(file_path)
            chunks = chunk_text(content, CHUNK_SIZE, CHUNK_OVERLAP)

            print(f"Ingesting '{file_path.name}' with {len(chunks)} chunks.")
            for chunk_index, chunk in enumerate(chunks):
                chunk_metadata = {
                    **metadata,
                    "chunk_index": chunk_index,
                }
                documents.append(chunk)
                metadatas.append(chunk_metadata)
                ids.append(create_chunk_id(file_path.name, chunk_index, chunk))

        except Exception as e:
            print(f"Error processing file '{file_path.name}': {e}")

    print(f"Adding {len(documents)} chunks to the ChromaDB collection '{CHROMA_COLLECTION}'.")
    embeddings = embedding_model.encode(
        documents,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).tolist()

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print(f"Successfully ingested {len(documents)} chunks into the collection '{CHROMA_COLLECTION}'.")

    return collection.count()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest documents from the RAW_DATA_DIR into ChromaDB."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset the ChromaDB collection before ingestion.",
    )
    args = parser.parse_args()
    ingest_documents(reset=args.reset)