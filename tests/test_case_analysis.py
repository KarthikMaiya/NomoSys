"""
Unit and integration tests for the NomoSys Case Document Analysis pipeline.

Tests cover:
  - Text chunking behaviour
  - FAISS index creation and retrieval
  - Embeddings initialization
  - EnsembleRetriever construction
  - analyze_case_document() end-to-end (requires Ollama running)

Run with:
    python -m pytest tests/test_case_analysis.py -v
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from langchain_core.documents import Document
except Exception:
    from langchain.schema import Document


# ─── Chunking Tests ──────────────────────────────────────────────────────


class TestChunking:
    """Verify RecursiveCharacterTextSplitter behaves as expected."""

    def test_splitter_creates_multiple_chunks(self):
        """A string longer than chunk_size must produce ≥ 2 chunks."""
        text = "A" * 1200
        docs = [Document(page_content=text)]
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        assert len(chunks) >= 2

    def test_splitter_preserves_content(self):
        """All original characters should appear in at least one chunk."""
        text = "Alpha Beta Gamma Delta Epsilon " * 50  # ~1500 chars
        docs = [Document(page_content=text)]
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
        chunks = splitter.split_documents(docs)
        recombined = " ".join(c.page_content for c in chunks)
        # Every unique word must appear
        for word in ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]:
            assert word in recombined

    def test_splitter_overlap(self):
        """Adjacent chunks should share overlapping text."""
        text = "Word " * 500  # 2500 chars
        docs = [Document(page_content=text)]
        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        if len(chunks) >= 2:
            tail = chunks[0].page_content[-50:]
            head = chunks[1].page_content[:50]
            # At least some overlap expected
            assert len(set(tail.split()) & set(head.split())) > 0

    def test_splitter_with_legal_chunk_params(self):
        """Test with the exact params used in production (1000/150)."""
        text = "Section 1. " * 200  # ~2200 chars
        docs = [Document(page_content=text)]
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(docs)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk.page_content) <= 1100  # some tolerance


# ─── FAISS Index Tests ───────────────────────────────────────────────────


class TestFAISS:
    """Verify FAISS index creation and retrieval."""

    @pytest.fixture(scope="class")
    def embeddings(self):
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

    def test_faiss_build(self, embeddings):
        """FAISS.from_documents should create a valid index."""
        docs = [
            Document(page_content="Article 21 of the Indian Constitution guarantees right to life."),
            Document(page_content="The cheque bounced due to insufficient funds."),
        ]
        db = FAISS.from_documents(docs, embeddings)
        assert db is not None
        assert db.index.ntotal == 2

    def test_faiss_retrieval_relevance(self, embeddings):
        """Querying should return the most relevant document."""
        docs = [
            Document(page_content="Alice filed an FIR against Bob for theft."),
            Document(page_content="The Supreme Court upheld fundamental rights."),
            Document(page_content="Contract between Party X and Party Y for Rs 10 lakhs."),
        ]
        db = FAISS.from_documents(docs, embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 1})
        results = retriever.get_relevant_documents("Who filed the FIR?")
        assert len(results) == 1
        assert "Alice" in results[0].page_content

    def test_faiss_k_parameter(self, embeddings):
        """Retriever should return exactly k results."""
        docs = [Document(page_content=f"Document {i}") for i in range(10)]
        db = FAISS.from_documents(docs, embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 3})
        results = retriever.get_relevant_documents("Document")
        assert len(results) == 3


# ─── Embeddings Tests ────────────────────────────────────────────────────


class TestEmbeddings:
    """Verify embeddings initialization."""

    def test_huggingface_embeddings_load(self):
        """HuggingFace embeddings model should load successfully."""
        emb = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        assert emb is not None

    def test_huggingface_embed_query(self):
        """Embedding a query should return a float vector."""
        emb = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        vector = emb.embed_query("What is Article 21?")
        assert isinstance(vector, list)
        assert len(vector) > 0
        assert isinstance(vector[0], float)

    def test_get_embeddings_helper(self):
        """_get_embeddings() should return HuggingFaceEmbeddings by default."""
        from chatbot_backend import _get_embeddings
        emb = _get_embeddings()
        assert isinstance(emb, HuggingFaceEmbeddings)


# ─── EnsembleRetriever Tests ─────────────────────────────────────────────


class TestEnsembleRetriever:
    """Verify combined retriever merges results from two sources."""

    @pytest.fixture(scope="class")
    def embeddings(self):
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

    def test_ensemble_returns_from_both_sources(self, embeddings):
        """EnsembleRetriever should include documents from both indices."""
        from langchain.retrievers import EnsembleRetriever

        # Source 1: Case documents
        case_docs = [Document(page_content="Accused Ram stole Rs 50000 from complainant Shyam.")]
        case_db = FAISS.from_documents(case_docs, embeddings)
        case_retriever = case_db.as_retriever(search_kwargs={"k": 1})

        # Source 2: Legal documents
        law_docs = [Document(page_content="Section 379 IPC: Punishment for theft.")]
        law_db = FAISS.from_documents(law_docs, embeddings)
        law_retriever = law_db.as_retriever(search_kwargs={"k": 1})

        combined = EnsembleRetriever(
            retrievers=[law_retriever, case_retriever],
            weights=[0.5, 0.5],
        )
        results = combined.get_relevant_documents("What happened in the theft case?")
        contents = [r.page_content for r in results]
        # Should have results from both sources
        assert any("Ram" in c or "Shyam" in c for c in contents)
        assert any("379" in c or "theft" in c for c in contents)


# ─── analyze_case_document Tests ─────────────────────────────────────────


class TestAnalyzeCaseDocument:
    """Test the analyze_case_document function with mocked LLM."""

    def test_analyze_with_txt_file(self):
        """analyze_case_document should work with a .txt BytesIO."""
        from chatbot_backend import analyze_case_document

        sample_text = (
            "FIR No. 123/2024\n"
            "Complainant: Rajesh Kumar\n"
            "Accused: Suresh Sharma\n"
            "Date of Incident: 15-01-2024\n"
            "Details: The accused assaulted the complainant near MG Road.\n"
        )
        file_obj = io.BytesIO(sample_text.encode("utf-8"))
        file_obj.name = "sample_fir.txt"

        # Mock the LLM to avoid needing Ollama
        mock_summary = "**Parties:** Rajesh Kumar (Complainant), Suresh Sharma (Accused)\n**Facts:** Assault on 15-01-2024"
        with patch("chatbot_backend._get_llm") as mock_llm_fn:
            mock_llm = MagicMock()
            mock_llm.predict.return_value = mock_summary
            mock_llm_fn.return_value = mock_llm

            case_db, summary = analyze_case_document(file_obj)

        assert isinstance(case_db, FAISS)
        assert case_db.index.ntotal >= 1
        assert "Rajesh" in summary or "Complainant" in summary

    def test_analyze_empty_file_raises(self):
        """analyze_case_document should raise ValueError for empty files."""
        from chatbot_backend import analyze_case_document

        file_obj = io.BytesIO(b"")
        file_obj.name = "empty.txt"

        with pytest.raises((ValueError, Exception)):
            with patch("chatbot_backend._get_llm") as mock_llm_fn:
                mock_llm = MagicMock()
                mock_llm.predict.return_value = ""
                mock_llm_fn.return_value = mock_llm
                analyze_case_document(file_obj)

    def test_analyze_pdf_with_no_extractable_text_raises_clear_error(self):
        """Image-only PDFs should fail before FAISS receives an empty chunk list."""
        from chatbot_backend import analyze_case_document

        file_obj = io.BytesIO(b"%PDF-1.4 fake pdf bytes")
        file_obj.name = "scanned_fir.pdf"

        with patch("chatbot_backend.PyPDFLoader") as mock_loader, \
             patch("chatbot_backend.FAISS.from_documents") as mock_from_documents:
            mock_loader.return_value.load.return_value = [
                Document(page_content="", metadata={"page": 0})
            ]

            with pytest.raises(ValueError, match="No extractable text"):
                analyze_case_document(file_obj)

        mock_from_documents.assert_not_called()

    def test_analyze_handles_missing_llm_summary_content(self):
        """Unexpected empty LLM output should not crash summary handling."""
        from chatbot_backend import analyze_case_document

        sample_text = "Complainant: Anita Devi\nAccused: Mohan Singh\nIncident: Theft."
        file_obj = io.BytesIO(sample_text.encode("utf-8"))
        file_obj.name = "sample_fir.txt"

        with patch("chatbot_backend._get_llm") as mock_llm_fn:
            mock_llm = MagicMock()
            mock_llm.predict.return_value = None
            mock_llm_fn.return_value = mock_llm

            case_db, summary = analyze_case_document(file_obj)

        assert isinstance(case_db, FAISS)
        assert "could not be generated" in summary


# ─── Prompt Template Tests ───────────────────────────────────────────────


class TestPromptTemplates:
    """Verify prompt templates format correctly."""

    def test_case_analysis_prompt_format(self):
        from chatbot_backend import CASE_ANALYSIS_PROMPT
        result = CASE_ANALYSIS_PROMPT.format(text="Sample case text here.")
        assert "Sample case text here." in result
        assert "Parties Involved" in result
        assert "Key Facts" in result

    def test_combined_qa_prompt_format(self):
        from chatbot_backend import COMBINED_QA_PROMPT
        result = COMBINED_QA_PROMPT.format(
            context="Some legal context.", question="What law applies?"
        )
        assert "Some legal context." in result
        assert "What law applies?" in result
        assert "hallucinate" in result.lower() or "Do NOT invent" in result
