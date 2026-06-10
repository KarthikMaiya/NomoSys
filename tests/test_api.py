"""
FastAPI endpoint tests for the NomoSys API server.

Tests cover:
  - GET /healthz
  - POST /upload (case document upload)
  - GET /case (case status)
  - DELETE /case (clear case)
  - POST /chat (with and without case document)

Run with:
    python -m pytest tests/test_api.py -v
"""
from __future__ import annotations

import io
import base64
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def client():
    """Create a FastAPI TestClient for the NomoSys API."""
    from fastapi.testclient import TestClient
    from api_server import app
    with TestClient(app) as c:
        yield c


# ─── Health Check ────────────────────────────────────────────────────────


class TestHealthz:
    def test_healthz_returns_200(self, client):
        resp = client.get("/healthz")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_healthz_has_case_loaded_field(self, client):
        resp = client.get("/healthz")
        data = resp.json()
        assert "case_loaded" in data


# ─── Case Status ─────────────────────────────────────────────────────────


class TestCaseStatus:
    def test_no_case_initially(self, client):
        resp = client.get("/case")
        assert resp.status_code == 200
        data = resp.json()
        assert data["has_case"] is False
        assert data["summary"] is None


# ─── Upload Endpoint ─────────────────────────────────────────────────────


class TestUpload:
    def test_upload_empty_file_returns_400(self, client):
        """Uploading an empty file should return 400."""
        resp = client.post(
            "/upload",
            files={"file": ("empty.txt", b"", "text/plain")},
        )
        assert resp.status_code == 400

    def test_upload_txt_file(self, client):
        """Uploading a valid TXT file should return 200 with a summary."""
        sample_text = (
            "FIR No. 456/2024\n"
            "Complainant: Anita Devi\n"
            "Accused: Mohan Singh\n"
            "Incident: Theft of gold jewelry on 10-03-2024\n"
        ).encode("utf-8")

        mock_summary = "Parties: Anita Devi (Complainant), Mohan Singh (Accused)"
        with patch("api_server.analyze_case_document") as mock_analyze:
            mock_db = MagicMock()
            mock_db.index.ntotal = 3
            mock_analyze.return_value = (mock_db, mock_summary)

            resp = client.post(
                "/upload",
                files={"file": ("test_fir.txt", sample_text, "text/plain")},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "Anita" in data["summary"] or mock_summary == data["summary"]
        assert data["chunks"] == 3


# ─── Clear Case ──────────────────────────────────────────────────────────


class TestClearCase:
    def test_delete_case(self, client):
        """DELETE /case should clear the case state."""
        resp = client.delete("/case")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "cleared"

        # Verify case is cleared
        resp = client.get("/case")
        data = resp.json()
        assert data["has_case"] is False


# ─── Chat Endpoint ───────────────────────────────────────────────────────


class TestChat:
    def test_chat_without_case(self, client):
        """POST /chat without a case document should use the legal chain."""
        with patch("api_server.build_legal_chain") as mock_chain_fn:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = {"answer": "Article 21 guarantees right to life."}
            mock_chain_fn.return_value = mock_chain

            resp = client.post(
                "/chat",
                json={"question": "What is Article 21?", "history": [], "translate": False},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert data["target_lang"] == "en"

    def test_chat_with_inline_file(self, client):
        """POST /chat with file_content should process the file and use combined chain."""
        sample_text = b"Accused stole property from complainant."
        encoded = base64.b64encode(sample_text).decode()

        mock_summary = "Theft case summary"
        mock_db = MagicMock()

        with patch("api_server.analyze_case_document") as mock_analyze, \
             patch("api_server.build_combined_chain") as mock_combined, \
             patch("api_server.build_legal_chain") as mock_legal:
            mock_analyze.return_value = (mock_db, mock_summary)
            mock_combined_chain = MagicMock()
            mock_combined_chain.invoke.return_value = {"answer": "Theft under BNS Section 303."}
            mock_combined.return_value = mock_combined_chain
            mock_legal.return_value = MagicMock()

            resp = client.post(
                "/chat",
                json={
                    "question": "What offence was committed?",
                    "history": [],
                    "translate": False,
                    "file_name": "case.txt",
                    "file_content": encoded,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data

    def test_chat_validation_empty_question(self, client):
        """POST /chat with empty question should return 422."""
        resp = client.post(
            "/chat",
            json={"question": "", "history": []},
        )
        assert resp.status_code == 422
