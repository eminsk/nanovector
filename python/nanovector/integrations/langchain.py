"""
NanoVector LangChain Integration
Drop-in, zero-bloat vector store replacement for ChromaDB / FAISS.
Copyright (c) 2026 eminsk (M_N_Nik@yahoo.com)
MIT License
"""

from __future__ import annotations
import uuid
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)
import numpy as np

# Try importing langchain base classes; provide zero-dependency fallbacks if absent
try:
    from langchain_core.documents import Document
    from langchain_core.vectorstores import VectorStore
    from langchain_core.embeddings import Embeddings
except ImportError:
    try:
        from langchain.schema import Document
        from langchain.vectorstores.base import VectorStore
        from langchain.embeddings.base import Embeddings
    except ImportError:
        VectorStore = object  # Fallback base class
        Embeddings = Any

        class Document:  # type: ignore
            """Lightweight standalone Document representation when langchain is not installed."""
            def __init__(
                self,
                page_content: str,
                metadata: Optional[Dict[str, Any]] = None,
                id: Optional[str] = None,
            ):
                self.page_content = page_content
                self.metadata = metadata or {}
                self.id = id

            def __repr__(self) -> str:
                snippet = (self.page_content[:47] + "...") if len(self.page_content) > 50 else self.page_content
                return f"Document(id={self.id!r}, page_content={snippet!r}, metadata={self.metadata!r})"

            def __eq__(self, other: Any) -> bool:
                if not isinstance(other, Document):
                    return False
                return (
                    self.page_content == other.page_content
                    and self.metadata == other.metadata
                    and self.id == other.id
                )


from nanovector import Index, Match


class NanoVectorStore(VectorStore):
    """
    NanoVector VectorStore for LangChain.

    A bare-metal, ultra-fast embedded vector store with SIMD acceleration
    and zero heavy dependencies.

    Parameters
    ----------
    embedding : Embeddings or callable
        LangChain Embeddings model (or object with embed_documents and embed_query).
    dim : int, optional
        Vector dimension. If not provided, it will be automatically inferred from the first embedded text.
    metric : str, default 'cosine'
        Distance metric: 'cosine', 'dot', or 'l2'.
    index : Index, optional
        Existing pre-initialized NanoVector Index instance.
    normalize : bool, default False
        Whether to L2-normalize vectors.
    """

    def __init__(
        self,
        embedding: Any,
        dim: Optional[int] = None,
        metric: str = "cosine",
        index: Optional[Index] = None,
        normalize: bool = False,
    ):
        self.embedding = embedding
        self.metric = metric
        self.normalize = normalize
        self._index: Optional[Index] = index
        if index is not None and dim is None:
            dim = index.dim
        self.dim = dim

        if self._index is None and self.dim is not None:
            self._index = Index(dim=self.dim, metric=self.metric, normalize=self.normalize)

    @property
    def index(self) -> Index:
        """The underlying NanoVector native Index instance."""
        if self._index is None:
            raise ValueError(
                "NanoVectorStore index is not initialized yet. Add documents first or specify dim at initialization."
            )
        return self._index

    def _ensure_index(self, sample_vec: Any) -> Index:
        if self._index is None:
            dim = len(sample_vec)
            self.dim = dim
            self._index = Index(dim=dim, metric=self.metric, normalize=self.normalize)
        return self._index

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Run texts through the embeddings and add them to the vectorstore.

        Parameters
        ----------
        texts : iterable of str
            Texts to add to the vectorstore.
        metadatas : list of dict, optional
            List of metadata dicts corresponding to texts.
        ids : list of str, optional
            List of IDs corresponding to texts. Generated as UUID4 if not provided.

        Returns
        -------
        List[str]
            List of IDs of the added texts.
        """
        text_list = list(texts)
        if not text_list:
            return []

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in text_list]
        elif len(ids) != len(text_list):
            raise ValueError(f"Number of ids ({len(ids)}) must match number of texts ({len(text_list)})")

        if metadatas is None:
            metadatas = [{} for _ in text_list]
        elif len(metadatas) != len(text_list):
            raise ValueError(f"Number of metadatas ({len(metadatas)}) must match number of texts ({len(text_list)})")

        # Embed texts
        if hasattr(self.embedding, "embed_documents"):
            embeddings = self.embedding.embed_documents(text_list)
        elif callable(self.embedding):
            embeddings = self.embedding(text_list)
        else:
            raise ValueError("Embedding object must have embed_documents method or be callable.")

        vectors = np.asarray(embeddings, dtype=np.float32)
        idx = self._ensure_index(vectors[0])

        # Prepare metadata payload containing the text content
        augmented_metas = []
        for text, meta in zip(text_list, metadatas):
            doc_meta = dict(meta) if meta else {}
            doc_meta["_page_content"] = text
            augmented_metas.append(doc_meta)

        idx.add_batch(ids=ids, vectors=vectors, metadatas=augmented_metas)
        return ids

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add LangChain Document objects to the vectorstore."""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        if ids is None and documents and getattr(documents[0], "id", None) is not None:
            ids = [doc.id for doc in documents]
        return self.add_texts(texts=texts, metadatas=metadatas, ids=ids, **kwargs)

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Union[Dict[str, Any], Callable[[Any], bool]]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return LangChain documents most similar to embedding vector with scores."""
        if self._index is None or len(self._index) == 0:
            return []

        query = np.asarray(embedding, dtype=np.float32)
        matches = self._index.search(query=query, top_k=k, filter=filter)

        results: List[Tuple[Document, float]] = []
        for match in matches:
            meta = match.meta or {}
            if isinstance(meta, dict):
                meta_copy = dict(meta)
                page_content = meta_copy.pop("_page_content", meta_copy.get("text", ""))
            else:
                meta_copy = {"raw_metadata": meta}
                page_content = ""
            doc = Document(page_content=page_content, metadata=meta_copy, id=match.id)
            results.append((doc, match.score))

        return results

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Union[Dict[str, Any], Callable[[Any], bool]]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return documents most similar to embedding vector."""
        docs_and_scores = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter, **kwargs
        )
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Union[Dict[str, Any], Callable[[Any], bool]]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Run query through embeddings and return similar documents with scores."""
        if hasattr(self.embedding, "embed_query"):
            query_embedding = self.embedding.embed_query(query)
        elif callable(self.embedding):
            query_embedding = self.embedding([query])[0]
        else:
            raise ValueError("Embedding object must have embed_query method or be callable.")

        return self.similarity_search_with_score_by_vector(
            embedding=query_embedding, k=k, filter=filter, **kwargs
        )

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Union[Dict[str, Any], Callable[[Any], bool]]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Run query through embeddings and return similar documents."""
        docs_and_scores = self.similarity_search_with_score(
            query=query, k=k, filter=filter, **kwargs
        )
        return [doc for doc, _ in docs_and_scores]

    def as_retriever(self, **kwargs: Any) -> Any:
        """Return a VectorStoreRetriever initialized from this VectorStore."""
        try:
            if hasattr(super(), "as_retriever") and callable(getattr(super(), "as_retriever")):
                return super().as_retriever(**kwargs)
        except Exception:
            pass

        class _SimpleRetriever:
            def __init__(self, store: NanoVectorStore, search_kwargs: dict):
                self.store = store
                self.search_kwargs = search_kwargs

            def invoke(self, input: str) -> List[Document]:
                k = self.search_kwargs.get("k", 4)
                filter_spec = self.search_kwargs.get("filter")
                return self.store.similarity_search(input, k=k, filter=filter_spec)

            def get_relevant_documents(self, query: str) -> List[Document]:
                return self.invoke(query)

        search_kwargs = kwargs.get("search_kwargs", {})
        return _SimpleRetriever(self, search_kwargs)

    def save(self, filepath: str) -> None:
        """Save vectorstore index to a .nvec file."""
        self.index.save(filepath)

    @classmethod
    def load(
        cls,
        filepath: str,
        embedding: Any,
        metric: Optional[str] = None,
    ) -> NanoVectorStore:
        """Load a NanoVectorStore from a .nvec file."""
        idx = Index.load(filepath)
        return cls(
            embedding=embedding,
            dim=idx.dim,
            metric=metric or idx.metric,
            index=idx,
        )

    @classmethod
    def from_texts(
        cls: Type[NanoVectorStore],
        texts: List[str],
        embedding: Any,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        metric: str = "cosine",
        normalize: bool = False,
        **kwargs: Any,
    ) -> NanoVectorStore:
        """Create a NanoVectorStore from raw texts."""
        store = cls(embedding=embedding, metric=metric, normalize=normalize)
        store.add_texts(texts=texts, metadatas=metadatas, ids=ids, **kwargs)
        return store

    @classmethod
    def from_documents(
        cls: Type[NanoVectorStore],
        documents: List[Document],
        embedding: Any,
        ids: Optional[List[str]] = None,
        metric: str = "cosine",
        normalize: bool = False,
        **kwargs: Any,
    ) -> NanoVectorStore:
        """Create a NanoVectorStore from LangChain Document instances."""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return cls.from_texts(
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
            ids=ids,
            metric=metric,
            normalize=normalize,
            **kwargs,
        )


__all__ = ["NanoVectorStore", "Document"]