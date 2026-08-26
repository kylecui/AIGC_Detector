"""Tests for POST /api/v1/detect/file (file-upload detection route).

Asserts the CURRENT behavior implemented in src/aigc_detector/api/routes.py:
    - extension whitelist {.pdf, .txt, .md} → 415 otherwise
    - >20 MB body → 413
    - empty file → 422
    - PDF with no extractable text → 422 (extraction-diagnosis branch)
    - extracted text <50 chars → 422 (min-length branch)
    - happy paths return the standard DetectionResponse schema (which has
      no file_name/chars fields — detection payload only)

The response schema (DetectionResponse) carries no file metadata; we assert
the detection fields that exist today plus caveat/segments semantics.
"""

from __future__ import annotations

import fitz  # PyMuPDF
from fastapi.testclient import TestClient

# Casual-register English, no formal lexical markers, and deliberately
# <120 chars: the optional ML register gate (register.py Layer 2) only runs
# on len(text) >= 120, so caveat is deterministically None here regardless
# of whether the ML gate artifact is deployed. Comfortably above the
# 50-char extraction minimum.
TXT_CONTENT = (
    "Once upon a time a little dog chased balls around the garden every "
    "morning, barking at the mailman happily."
)

# Long enough to produce at least one segment (MIN_SEGMENT_CHARS=80).
MD_CONTENT = (
    "# A quiet afternoon\n\n"
    "The cat slept on the windowsill and dreamed about fish and warm milk. "
    "Outside, rain tapped gently on the glass while the kettle whistled "
    "softly in the kitchen, promising tea and biscuits for everyone.\n\n"
    "## Later\n\nThe dog napped too, snoring in the corner."
)


def _make_pdf(text: str | None) -> bytes:
    """Generate a minimal 1-page PDF (~1 KB) via PyMuPDF at test time."""
    doc = fitz.open()
    page = doc.new_page()
    if text is not None:
        page.insert_text((72, 72), text)
    payload = doc.tobytes()
    doc.close()
    return payload


class TestDetectFileHappyPath:
    def test_upload_txt(self, client: TestClient):
        resp = client.post(
            "/api/v1/detect/file?include_segments=false",
            files={"file": ("story.txt", TXT_CONTENT.encode("utf-8"), "text/plain")},
        )
        assert resp.status_code == 200
        data = resp.json()
        # Detection payload fields (stub pipeline verdict)
        assert data["predicted_label"] == "AI-generated"
        assert data["confidence"] == 0.87
        assert data["p_ai"] == 0.87
        assert data["detected_language"] == "en"
        assert "statistical" in data["stages_used"]
        assert data["processing_time_ms"] == 450.0
        # include_segments=false → no segments / no highlights
        assert data["segments"] == []
        assert data["segment_highlights"] is None
        # Casual register → no caveat, no calibration, no decision rule
        assert data["caveat"] is None
        assert data["calibration"] is None
        assert data["decision_rule"] is None

    def test_upload_md_default_segments(self, client: TestClient):
        """File endpoint defaults include_segments=true → segments populated."""
        resp = client.post(
            "/api/v1/detect/file",
            files={"file": ("notes.md", MD_CONTENT.encode("utf-8"), "text/markdown")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["predicted_label"] == "AI-generated"
        assert isinstance(data["segments"], list)
        assert len(data["segments"]) >= 1
        segment = data["segments"][0]
        assert {"index", "text", "char_start", "char_end", "predicted_label", "p_ai"} <= set(segment)
        # Stub pipeline scores every segment p_ai=0.87 → highlights present
        assert data["segment_highlights"] is not None
        assert data["segment_highlights"]["max_p_ai"] == 0.87
        # NOTE: no caveat assertion here — this text is >=120 chars, so the
        # optional ML register gate may legitimately fire or not depending
        # on the deployed calibration artifact (verdict stays 200 either way).


class TestDetectFileRejection:
    def test_reject_exe_extension(self, client: TestClient):
        resp = client.post(
            "/api/v1/detect/file",
            files={"file": ("payload.exe", b"MZfake", "application/octet-stream")},
        )
        assert resp.status_code == 415
        assert "Unsupported file type" in resp.json()["detail"]

    def test_reject_oversized_file(self, client: TestClient):
        big = b"a" * (21 * 1024 * 1024)  # 21 MB > 20 MB limit
        resp = client.post(
            "/api/v1/detect/file",
            files={"file": ("big.txt", big, "text/plain")},
        )
        assert resp.status_code == 413
        assert "too large" in resp.json()["detail"].lower()

    def test_reject_empty_file(self, client: TestClient):
        resp = client.post(
            "/api/v1/detect/file",
            files={"file": ("empty.txt", b"", "text/plain")},
        )
        assert resp.status_code == 422
        assert "empty" in resp.json()["detail"].lower()

    def test_reject_blank_pdf_no_text(self, client: TestClient):
        """Valid 1-page PDF with zero extractable text → 422 no-text branch."""
        pdf_bytes = _make_pdf(text=None)
        resp = client.post(
            "/api/v1/detect/file",
            files={"file": ("blank.pdf", pdf_bytes, "application/pdf")},
        )
        assert resp.status_code == 422
        assert "未找到任何文本" in resp.json()["detail"]

    def test_reject_pdf_short_text(self, client: TestClient):
        """Valid PDF whose extraction succeeds but yields <50 chars → 422 min-length."""
        pdf_bytes = _make_pdf(text="Hello world, short PDF.")
        resp = client.post(
            "/api/v1/detect/file",
            files={"file": ("short.pdf", pdf_bytes, "application/pdf")},
        )
        assert resp.status_code == 422
        assert "too short" in resp.json()["detail"]

    def test_reject_short_txt_content(self, client: TestClient):
        """Decodable .txt whose content is <50 chars → 422 min-length."""
        resp = client.post(
            "/api/v1/detect/file",
            files={"file": ("tiny.txt", b"This file is way too short.", "text/plain")},
        )
        assert resp.status_code == 422
        assert "too short" in resp.json()["detail"]
