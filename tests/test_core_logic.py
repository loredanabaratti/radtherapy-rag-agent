"""Fast unit tests for ingestion and graph routing logic.

These tests deliberately avoid Ollama, embeddings and ChromaDB.
They validate small deterministic functions that should work without external services.
"""

import pytest

from src.graph import route_after_critic
from src.ingest import chunk_text, parse_document


def test_parse_document_extracts_metadata_and_content(tmp_path):
    """Metadata header lines and the document body are separated correctly."""

    source_file = tmp_path / "imrt.txt"
    source_file.write_text(
        "Title: IMRT and Inverse Treatment Planning\n"
        "Source: IAEA\n"
        "URL: https://example.org/imrt\n"
        "\n"
        "Inverse planning uses dose objectives for targets and organs at risk.\n",
        encoding="utf-8",
    )
    metadata, content = parse_document(source_file)

    assert metadata == {
        "title": "IMRT and Inverse Treatment Planning",
        "source": "IAEA",
        "url": "https://example.org/imrt",
        "file_name": "imrt.txt",
    }
    assert content == "Inverse planning uses dose objectives for targets and organs at risk."


def test_chunk_text_creates_expected_overlap():
    """Neighbouring character chunks share the configured overlap."""

    chunks = chunk_text("abcdefghij", chunk_size=6, chunk_overlap=2)

    assert chunks == ["abcdef", "efghij"]
    assert chunks[0][-2:] == chunks[1][:2]  # overlap is correct


def test_chunk_text_rejects_overlap_at_least_chunk_size():
    """Invalid parameters are rejected rather than causing an infinite loop."""

    with pytest.raises(ValueError, match="overlap must be smaller"):
        chunk_text("some text", chunk_size=5, chunk_overlap=5)


@pytest.mark.parametrize(
    ("critic_result", "retry_count", "expected_route"),
    [
        ({"is_grounded": True}, 0, "end"),
        ({"is_grounded": False}, 0, "retry"),
        ({"is_grounded": False}, 1, "end"),
    ],
)
def test_route_after_critic_limits_retries(critic_result, retry_count, expected_route):
    """An ungrounded answer may be revised once, but never retried indefinitely."""

    state = {
        "critic_result": critic_result,
        "retry_count": retry_count,
    }

    assert route_after_critic(state) == expected_route
