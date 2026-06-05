"""
ui.utils.api_client
===================
Synchronous HTTP client for interacting with the ARIA FastAPI backend.

All methods catch exceptions and return empty/default values on failure
to prevent the Streamlit UI from crashing.
"""
from __future__ import annotations

import requests


class ARIAClient:
    """Wraps all HTTP calls to the FastAPI backend."""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.timeout = 120  # seconds

    def chat(self, message: str, session_id: str, user_id: str, debug_mode: bool = False, primary_model: str | None = None) -> dict:
        """Synchronous chat request. Returns ChatResponse dict or error dict."""
        try:
            response = requests.post(
                f"{self.base_url}/chat",
                json={
                    "message": message,
                    "session_id": session_id,
                    "user_id": user_id,
                    "debug_mode": debug_mode,
                    "primary_model": primary_model,
                },
                headers=self.headers,
                timeout=self.timeout,
            )
            if response.ok:
                return response.json()
            else:
                return {"error": f"API error {response.status_code}: {response.text}"}
        except Exception as e:
            return {"error": f"Connection error: {str(e)}"}

    def get_model_status(self) -> dict:
        """Fetch model health status. Public endpoint, no auth needed."""
        try:
            r = requests.get(f"{self.base_url}/model-status", timeout=5)
            return r.json() if r.ok else {}
        except Exception:
            return {}

    def get_history(self, session_id: str, limit: int = 50) -> dict:
        """Fetch conversation turns for a session."""
        try:
            r = requests.get(
                f"{self.base_url}/chat/{session_id}/history",
                headers=self.headers,
                timeout=10,
                params={"limit": limit},
            )
            return r.json() if r.ok else {"messages": []}
        except Exception:
            return {"messages": []}

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and its history."""
        try:
            r = requests.delete(
                f"{self.base_url}/chat/{session_id}",
                headers=self.headers,
                timeout=10,
            )
            return r.ok
        except Exception:
            return False

    def upload_document(self, file_bytes: bytes, filename: str, user_id: str) -> dict:
        """Upload a document to the Knowledge Base."""
        try:
            files = {"file": (filename, file_bytes)}
            r = requests.post(
                f"{self.base_url}/documents/upload",
                files=files,
                data={"user_id": user_id},
                headers={"Authorization": self.headers["Authorization"]},
                timeout=120,
            )
            return r.json() if r.ok else {"error": r.text}
        except Exception as e:
            return {"error": str(e)}

    def get_documents(self, user_id: str) -> list:
        """List documents ingested by the user."""
        try:
            r = requests.get(
                f"{self.base_url}/documents",
                headers=self.headers,
                timeout=10,
                params={"user_id": user_id},
            )
            return r.json().get("documents", []) if r.ok else []
        except Exception:
            return []

    def delete_document(self, doc_id: str, user_id: str) -> bool:
        """Delete a document from the Knowledge Base."""
        try:
            r = requests.delete(
                f"{self.base_url}/documents/{doc_id}",
                headers=self.headers,
                timeout=10,
                params={"user_id": user_id},
            )
            return r.ok
        except Exception:
            return False

    def get_memories(self, user_id: str, min_importance: float = 0.0) -> list:
        """List episodic memories for the user."""
        try:
            r = requests.get(
                f"{self.base_url}/memory",
                headers=self.headers,
                timeout=10,
                params={"user_id": user_id, "min_importance": min_importance},
            )
            return r.json().get("memories", []) if r.ok else []
        except Exception:
            return []

    def delete_memory(self, memory_id: str, user_id: str) -> bool:
        """Delete a specific episodic memory."""
        try:
            r = requests.delete(
                f"{self.base_url}/memory/{memory_id}",
                headers=self.headers,
                timeout=10,
                params={"user_id": user_id},
            )
            return r.ok
        except Exception:
            return False

    def get_analytics(self, days: int = 7) -> dict:
        """Fetch aggregated usage analytics."""
        try:
            r = requests.get(
                f"{self.base_url}/analytics",
                headers=self.headers,
                timeout=10,
                params={"days": days},
            )
            return r.json() if r.ok else {}
        except Exception:
            return {}
