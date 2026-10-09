"""Tests for the synchronous ABCs under :mod:`chimera_rag.interfaces`.

Each ABC must:
1. be abstract (cannot be instantiated directly),
2. declare the documented abstract method(s),
3. allow instantiation once subclasses implement them.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_base_chunker_is_abstract_and_subclass_works():
    from chimera_rag.core.types import Chunk, Document
    from chimera_rag.interfaces.chunker import BaseChunker

    with pytest.raises(TypeError):
        BaseChunker()  # cannot instantiate

    class DummyChunker(BaseChunker):
        def chunk(self, document: Document) -> list[Chunk]:
            return [Chunk(chunk_id=f"{document.doc_id}::0", doc_id=document.doc_id, index=0, text=document.content)]

    out = DummyChunker().chunk(Document(doc_id="d", content="hello"))
    assert len(out) == 1
    assert out[0].text == "hello"


def test_base_triple_extractor_is_abstract_and_subclass_works():
    from chimera_rag.core.types import Chunk, Triple
    from chimera_rag.interfaces.extractor import BaseTripleExtractor

    with pytest.raises(TypeError):
        BaseTripleExtractor()

    class DummyExtractor(BaseTripleExtractor):
        def extract(self, chunk: Chunk) -> list[Triple]:
            return [Triple(subject="a", predicate="p", object="b", source_chunk_id=chunk.chunk_id)]

    out = DummyExtractor().extract(Chunk(chunk_id="c1", doc_id="d1", index=0, text="x"))
    assert out[0].subject == "a"


def test_base_pruner_is_abstract_and_subclass_works():
    from chimera_rag.core.types import Triple
    from chimera_rag.interfaces.pruner import BasePruner

    with pytest.raises(TypeError):
        BasePruner()

    class KeepAll(BasePruner):
        def prune(self, triples: list[Triple]) -> list[Triple]:
            return triples

    t = Triple(subject="a", predicate="p", object="b")
    assert KeepAll().prune([t]) == [t]


def test_base_intent_classifier_is_abstract_and_subclass_works():
    from chimera_rag.core.types import Intent, Query
    from chimera_rag.interfaces.intent_classifier import BaseIntentClassifier

    with pytest.raises(TypeError):
        BaseIntentClassifier()

    class AlwaysFactual(BaseIntentClassifier):
        def classify(self, query: Query) -> Intent:
            return Intent(label="factual", confidence=1.0)

    intent = AlwaysFactual().classify(Query(text="x"))
    assert intent.label == "factual"


def test_base_retriever_is_abstract_and_subclass_works():
    from chimera_rag.core.types import Query, RetrievalResult
    from chimera_rag.interfaces.retriever import BaseRetriever

    with pytest.raises(TypeError):
        BaseRetriever()

    class EmptyRetriever(BaseRetriever):
        def retrieve(self, query: Query, top_k: int = 5) -> RetrievalResult:
            return RetrievalResult()

    result = EmptyRetriever().retrieve(Query(text="x"))
    assert result.chunks == []



def test_base_answer_generator_is_abstract_and_subclass_works():
    from chimera_rag.core.types import Answer, Query, RetrievalResult
    from chimera_rag.interfaces.generator import BaseAnswerGenerator

    with pytest.raises(TypeError):
        BaseAnswerGenerator()

    class EchoGenerator(BaseAnswerGenerator):
        def generate(self, query: Query, retrieval: RetrievalResult) -> Answer:
            return Answer(text=query.text)

    ans = EchoGenerator().generate(Query(text="hi"), RetrievalResult())
    assert ans.text == "hi"


def test_base_graph_store_is_abstract_and_subclass_works():
    from chimera_rag.core.types import Triple
    from chimera_rag.interfaces.graph_store import BaseGraphStore

    with pytest.raises(TypeError):
        BaseGraphStore()

    class TinyStore(BaseGraphStore):
        def __init__(self) -> None:
            self._triples: list[Triple] = []

        def add_triple(self, triple: Triple) -> None:
            self._triples.append(triple)

        def get_neighbors(self, entity: str, hops: int = 1) -> list[Triple]:
            return [t for t in self._triples if t.subject == entity or t.object == entity]

        def all_triples(self) -> list[Triple]:
            return list(self._triples)

        def persist(self, path: str) -> None:
            return None

        def load(self, path: str) -> None:
            return None

    s = TinyStore()
    s.add_triple(Triple(subject="A", predicate="p", object="B"))
    assert len(s.all_triples()) == 1
    assert len(s.get_neighbors("A")) == 1


def test_base_vector_store_is_abstract_and_subclass_works():
    from chimera_rag.interfaces.vector_store import BaseVectorStore

    with pytest.raises(TypeError):
        BaseVectorStore()

    class TinyVecStore(BaseVectorStore):
        def __init__(self) -> None:
            self._vecs: list[tuple[str, np.ndarray]] = []

        def add(self, ids: list[str], vectors: np.ndarray) -> None:
            for i, v in zip(ids, vectors):
                self._vecs.append((i, v))

        def search(self, query_vector: np.ndarray, top_k: int = 5):
            # Return list[(id, score)] using dot product.
            scored = [(i, float(np.dot(v, query_vector))) for i, v in self._vecs]
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]

        def persist(self, path: str) -> None:
            return None

        def load(self, path: str) -> None:
            return None

    vs = TinyVecStore()
    vs.add(["a", "b"], np.array([[1.0, 0.0], [0.0, 1.0]]))
    results = vs.search(np.array([1.0, 0.0]), top_k=1)
    assert results[0][0] == "a"


def test_base_embedding_provider_is_abstract_and_subclass_works():
    from chimera_rag.interfaces.embedding_provider import BaseEmbeddingProvider

    with pytest.raises(TypeError):
        BaseEmbeddingProvider()

    class ConstEmbed(BaseEmbeddingProvider):
        @property
        def dim(self) -> int:
            return 4

        def encode(self, texts: list[str]) -> np.ndarray:
            return np.ones((len(texts), 4), dtype=np.float32)

    arr = ConstEmbed().encode(["a", "b"])
    assert arr.shape == (2, 4)
