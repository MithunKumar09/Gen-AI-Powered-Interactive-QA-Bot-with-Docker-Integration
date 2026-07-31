from types import SimpleNamespace

import pytest

from rag_core.errors import ProviderError
from rag_core.store import _extract_listed_ids


def test_extract_listed_ids_from_string_page():
    assert _extract_listed_ids(
        ["doc#chunk-1", "doc#chunk-2"]
    ) == {
        "doc#chunk-1",
        "doc#chunk-2",
    }


def test_extract_listed_ids_from_sdk_vector_objects():
    page = [
        SimpleNamespace(id="doc#chunk-1"),
        SimpleNamespace(id="doc#chunk-2"),
    ]

    assert _extract_listed_ids(page) == {
        "doc#chunk-1",
        "doc#chunk-2",
    }


def test_extract_listed_ids_from_response_vectors():
    page = SimpleNamespace(
        vectors=[
            SimpleNamespace(id="doc#chunk-1"),
            SimpleNamespace(id="doc#chunk-2"),
        ]
    )

    assert _extract_listed_ids(page) == {
        "doc#chunk-1",
        "doc#chunk-2",
    }


def test_extract_listed_ids_from_mapping_response():
    page = {
        "vectors": [
            {"id": "doc#chunk-1"},
            {"id": "doc#chunk-2"},
        ]
    }

    assert _extract_listed_ids(page) == {
        "doc#chunk-1",
        "doc#chunk-2",
    }


def test_extract_listed_ids_rejects_unknown_page_shape():
    with pytest.raises(ProviderError) as exc_info:
        _extract_listed_ids(object())

    assert exc_info.value.detail is not None
    assert "Unsupported Pinecone list page shape" in exc_info.value.detail

    # Internal SDK details must not appear in the public exception message.
    assert str(exc_info.value) == "An upstream service failed. Please try again."


def test_extract_listed_ids_rejects_unknown_item_shape():
    with pytest.raises(ProviderError) as exc_info:
        _extract_listed_ids([object()])

    assert exc_info.value.detail is not None
    assert "Unsupported Pinecone list item shape" in exc_info.value.detail

    # Internal SDK details must not appear in the public exception message.
    assert str(exc_info.value) == "An upstream service failed. Please try again."