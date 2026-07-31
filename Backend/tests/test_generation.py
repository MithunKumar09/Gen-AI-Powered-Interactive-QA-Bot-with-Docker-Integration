from types import SimpleNamespace

from rag_core.generation import (
    Evidence,
    _extract_citations,
    _extract_text,
)


def test_evidence_uses_cohere_v2_document_shape():
    evidence = Evidence(
        chunk_index=7,
        page=3,
        text="Phase 1 quotation is ₹100,000.",
        score=0.95,
    )

    assert evidence.as_document() == {
        "id": "7",
        "data": {
            "text": "Phase 1 quotation is ₹100,000.",
            "page": "3",
        },
    }


def test_extract_text_from_sdk_response():
    response = SimpleNamespace(
        message=SimpleNamespace(
            content=[
                SimpleNamespace(text="Phase 1"),
                SimpleNamespace(text="costs ₹100,000."),
            ]
        )
    )

    assert _extract_text(response) == "Phase 1\ncosts ₹100,000."


def test_extract_citation_from_nested_document_object():
    evidence = [
        Evidence(
            chunk_index=7,
            page=3,
            text="Phase 1 quotation is ₹100,000.",
            score=0.95,
        )
    ]

    response = SimpleNamespace(
        message=SimpleNamespace(
            citations=[
                SimpleNamespace(
                    text="Phase 1 quotation is ₹100,000.",
                    sources=[
                        SimpleNamespace(
                            document=SimpleNamespace(id="7")
                        )
                    ],
                )
            ]
        )
    )

    assert _extract_citations(response, evidence) == [
        {
            "page": 3,
            "chunk_index": 7,
            "snippet": "Phase 1 quotation is ₹100,000.",
        }
    ]


def test_extract_citation_from_mapping_document():
    evidence = [
        Evidence(
            chunk_index=11,
            page=5,
            text="Implementation costs ₹250,000.",
            score=0.90,
        )
    ]

    response = {
        "message": {
            "citations": [
                {
                    "text": "Implementation costs ₹250,000.",
                    "sources": [
                        {
                            "document": {
                                "id": "11",
                                "data": {
                                    "text": "Implementation costs ₹250,000."
                                },
                            }
                        }
                    ],
                }
            ]
        }
    }

    assert _extract_citations(response, evidence) == [
        {
            "page": 5,
            "chunk_index": 11,
            "snippet": "Implementation costs ₹250,000.",
        }
    ]


def test_missing_model_citations_falls_back_to_best_evidence():
    evidence = [
        Evidence(
            chunk_index=1,
            page=1,
            text="Weak evidence.",
            score=0.30,
        ),
        Evidence(
            chunk_index=2,
            page=4,
            text="Strongest supporting evidence.",
            score=0.91,
        ),
    ]

    response = SimpleNamespace(
        message=SimpleNamespace(
            citations=[]
        )
    )

    assert _extract_citations(response, evidence) == [
        {
            "page": 4,
            "chunk_index": 2,
            "snippet": "Strongest supporting evidence.",
            "inferred": True,
        }
    ]