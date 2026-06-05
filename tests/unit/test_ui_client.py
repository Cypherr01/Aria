"""
tests.unit.test_ui_client
=========================
Unit tests for the ARIA Streamlit API Client.
"""
from __future__ import annotations

import pytest
import requests
from unittest.mock import MagicMock

from ui.utils.api_client import ARIAClient


@pytest.fixture
def api_client():
    return ARIAClient("http://mock-api.local", "test-key")


def test_api_client_chat_success(api_client, mocker):
    """Test successful chat response."""
    mock_response = MagicMock()
    mock_response.ok = True
    mock_response.json.return_value = {"response": "test", "session_id": "s1"}
    mocker.patch("requests.post", return_value=mock_response)

    result = api_client.chat("hello", "s1", "user1")
    assert "response" in result
    assert result["response"] == "test"


def test_api_client_chat_failure(api_client, mocker):
    """Test chat handles timeout exceptions safely."""
    mocker.patch("requests.post", side_effect=requests.Timeout("Timeout"))

    result = api_client.chat("hello", "s1", "user1")
    assert "error" in result
    assert "Connection error" in result["error"]


def test_api_client_model_status_empty_on_failure(api_client, mocker):
    """Test model status returns empty dict on connection error."""
    mocker.patch("requests.get", side_effect=requests.ConnectionError("Failed"))

    result = api_client.get_model_status()
    assert result == {}


def test_api_client_upload_document(api_client, mocker):
    """Test document upload success."""
    mock_response = MagicMock()
    mock_response.ok = True
    mock_response.json.return_value = {"doc_id": "d1", "chunk_count": 5}
    mocker.patch("requests.post", return_value=mock_response)

    result = api_client.upload_document(b"content", "test.txt", "user1")
    assert result.get("doc_id") == "d1"
    assert result.get("chunk_count") == 5
