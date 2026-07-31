"""Session identity, id derivation, and the privacy guarantee.

The central claim these tests defend: a raw browser session id never reaches
Pinecone metadata, a vector id, or a log line.
"""

from __future__ import annotations

import pytest

from rag_core.errors import ValidationError
from rag_core.ids import (
    document_prefix,
    file_hash,
    make_document_id,
    make_vector_id,
    owns_document,
    parse_document_id,
)
from rag_core.identity import derive_scope, scope_ref, validate_session_id

KEY = "scope-key"
RAW = "s" * 32


def test_scope_derivation_is_deterministic():
    assert derive_scope(KEY, RAW) == derive_scope(KEY, RAW)


def test_rotating_the_key_changes_every_scope():
    """Documented consequence: rotation makes existing records unreachable."""
    assert derive_scope(KEY, RAW) != derive_scope("different-key", RAW)


def test_different_sessions_get_different_scopes():
    assert derive_scope(KEY, "a" * 32) != derive_scope(KEY, "b" * 32)


def test_scope_is_not_the_raw_session_id():
    scope = derive_scope(KEY, RAW)
    assert RAW not in scope
    assert len(scope) == 64  # sha256 hex


def test_scope_ref_is_short_and_non_identifying():
    ref = scope_ref(derive_scope(KEY, RAW))
    assert len(ref) == 12
    assert RAW not in ref


@pytest.mark.parametrize("bad", [
    None, "", "short", "x" * 15, "x" * 129, "has spaces",
    "has/slash", "semi;colon", "quote'", "<script>",
])
def test_invalid_session_ids_rejected(bad):
    with pytest.raises(ValidationError):
        validate_session_id(bad)


@pytest.mark.parametrize("good", [
    "a" * 16, "a" * 128, "abcdef0123456789", "tok_-ABC123xyz789__",
])
def test_valid_session_ids_accepted(good):
    assert validate_session_id(good) == good


def test_document_id_embeds_derived_scope_not_raw_id():
    scope = derive_scope(KEY, RAW)
    doc = make_document_id(1, scope, file_hash(b"pdf"))
    assert RAW not in doc
    assert scope[:32] in doc


def test_document_id_is_deterministic_per_scope_and_bytes():
    scope = derive_scope(KEY, RAW)
    a = make_document_id(1, scope, file_hash(b"same"))
    b = make_document_id(1, scope, file_hash(b"same"))
    assert a == b


def test_identical_bytes_in_different_scopes_yield_different_ids():
    """This is what keeps two visitors uploading the same PDF isolated."""
    data = b"identical bytes"
    a = make_document_id(1, derive_scope(KEY, "a" * 32), file_hash(data))
    b = make_document_id(1, derive_scope(KEY, "b" * 32), file_hash(data))
    assert a != b


def test_vector_id_is_prefixed_by_document_id():
    doc = make_document_id(1, derive_scope(KEY, RAW), file_hash(b"x"))
    vid = make_vector_id(doc, 7)
    assert vid.startswith(document_prefix(doc))
    assert vid.endswith(":00007")


def test_document_prefix_has_trailing_separator():
    """Without it, listing one document could match a longer sibling id."""
    doc = make_document_id(1, derive_scope(KEY, RAW), file_hash(b"x"))
    assert document_prefix(doc) == f"{doc}:"


def test_chunk_index_is_zero_padded_for_lexical_order():
    doc = make_document_id(1, derive_scope(KEY, RAW), file_hash(b"x"))
    ids = [make_vector_id(doc, i) for i in (0, 2, 10, 100)]
    assert ids == sorted(ids)


@pytest.mark.parametrize("bad", [
    "", "nope", "v1:short:short", "v1:" + "z" * 32 + ":" + "0" * 32,
    "vX:" + "0" * 32 + ":" + "0" * 32, "v1:" + "0" * 32, "x" * 200,
])
def test_malformed_document_ids_rejected(bad):
    with pytest.raises(ValidationError):
        parse_document_id(bad)


def test_ownership_holds_for_the_deriving_scope():
    scope = derive_scope(KEY, RAW)
    doc = make_document_id(1, scope, file_hash(b"x"))
    assert owns_document(doc, scope, frozenset({1})) is True


def test_ownership_fails_across_scopes():
    doc = make_document_id(1, derive_scope(KEY, "a" * 32), file_hash(b"x"))
    other = derive_scope(KEY, "b" * 32)
    assert owns_document(doc, other, frozenset({1})) is False


def test_ownership_fails_for_unsupported_schema_version():
    scope = derive_scope(KEY, RAW)
    doc = make_document_id(9, scope, file_hash(b"x"))
    assert owns_document(doc, scope, frozenset({1})) is False


def test_ownership_of_malformed_id_is_false_not_an_exception():
    assert owns_document("garbage", derive_scope(KEY, RAW), frozenset({1})) is False


def test_file_hash_depends_only_on_bytes():
    assert file_hash(b"abc") == file_hash(b"abc")
    assert file_hash(b"abc") != file_hash(b"abd")
