# type: ignore[reportUnknownMemberType]
from unittest.mock import MagicMock

import pytest

from openai_vstore_toolkit.rag_services import StoreService


@pytest.fixture()
def mock_client() -> MagicMock:
    client = MagicMock()
    client.vector_stores = MagicMock()
    return client


@pytest.fixture()
def store_service(mock_client) -> StoreService:
    return StoreService(client=mock_client)


def test_get_or_create_reuses_existing_store(monkeypatch, store_service) -> None:
    """Ensure get_or_create returns an existing store ID and does not create a new store."""
    existing_id = "store-123"
    monkeypatch.setattr(
        store_service, "find_id_by_name", MagicMock(return_value=existing_id)
    )
    create_mock = MagicMock(return_value="created-id")
    monkeypatch.setattr(store_service, "create", create_mock)

    result = store_service.get_or_create(" Existing Store ")

    assert result == existing_id
    create_mock.assert_not_called()


def test_get_or_create_creates_store_when_missing(monkeypatch, store_service) -> None:
    """Ensure get_or_create delegates to create when the store name is not found."""
    monkeypatch.setattr(store_service, "find_id_by_name", MagicMock(return_value=None))
    create_mock = MagicMock(return_value="store-created")
    monkeypatch.setattr(store_service, "create", create_mock)

    result = store_service.get_or_create("New Store")

    assert result == "store-created"
    create_mock.assert_called_once_with("New Store")


def test_find_id_by_name_is_case_insensitive(monkeypatch, store_service) -> None:
    """Confirm find_id_by_name matches store names regardless of casing or whitespace."""
    stores = [
        {"id": "id-1", "name": "Primary"},
        {"id": "target-id", "name": "My Store"},
    ]
    monkeypatch.setattr(store_service, "list_store", MagicMock(return_value=stores))

    assert store_service.find_id_by_name(" my store ") == "target-id"


def test_find_id_by_name_returns_none_when_missing(monkeypatch, store_service) -> None:
    """Return None when find_id_by_name cannot locate the requested store name."""
    stores = [{"id": "id-1", "name": "Primary"}]
    monkeypatch.setattr(store_service, "list_store", MagicMock(return_value=stores))

    assert store_service.find_id_by_name("Missing") is None


def test_create_calls_openai_vector_store(mock_client, store_service) -> None:
    """Verify create forwards the request to the OpenAI client and returns the new ID."""
    mock_client.vector_stores.create.return_value = MagicMock(id="vs_123")

    result = store_service.create("Test Store")

    assert result == "vs_123"
    mock_client.vector_stores.create.assert_called_once_with(name="Test Store")


def test_get_returns_model_dump(mock_client, store_service) -> None:
    """Ensure get returns the dictionary produced by the model_dump() call."""
    retrieve_response = MagicMock()
    retrieve_response.model_dump.return_value = {"id": "vs_123", "name": "Stored"}
    mock_client.vector_stores.retrieve.return_value = retrieve_response

    result = store_service.get("vs_123")

    assert result == {"id": "vs_123", "name": "Stored"}
    mock_client.vector_stores.retrieve.assert_called_once_with(vector_store_id="vs_123")


def test_list_store_combines_paginated_results(mock_client, store_service) -> None:
    """Aggregate paginated list responses and propagate the correct cursor between calls."""

    class FakeVectorStore:
        def __init__(self, store_id, name):
            self.id = store_id
            self._payload = {"id": store_id, "name": name}

        def model_dump(self):
            return self._payload

    class FakeListResponse:
        def __init__(self, data, has_more, last_id=None):
            self.data = data
            self.has_more = has_more
            self.last_id = last_id

    first_page = FakeListResponse(
        data=[FakeVectorStore("vs_1", "Store One")],
        has_more=True,
        last_id="vs_1",
    )
    second_page = FakeListResponse(
        data=[FakeVectorStore("vs_2", "Store Two")],
        has_more=False,
        last_id="vs_2",
    )
    mock_client.vector_stores.list.side_effect = [first_page, second_page]

    result = store_service.list_store()

    assert result == [
        {"id": "vs_1", "name": "Store One"},
        {"id": "vs_2", "name": "Store Two"},
    ]
    first_call_kwargs = mock_client.vector_stores.list.call_args_list[0].kwargs
    second_call_kwargs = mock_client.vector_stores.list.call_args_list[1].kwargs
    assert first_call_kwargs == {"limit": 100, "after": None}
    assert second_call_kwargs == {"limit": 100, "after": "vs_1"}


def test_list_store_handles_missing_last_id(mock_client, store_service) -> None:
    """Ensure pagination falls back to the final item ID when last_id is absent."""

    class FakeVectorStore:
        def __init__(self, store_id, payload):
            self.id = store_id
            self._payload = payload

        def model_dump(self):
            return self._payload

    class FakeListResponse:
        def __init__(self, data, has_more, last_id=None):
            self.data = data
            self.has_more = has_more
            self.last_id = last_id

    payload = {"id": "vs_3", "name": "Store Three"}
    first_response = FakeListResponse(
        data=[FakeVectorStore("vs_3", payload)],
        has_more=True,
        last_id=None,
    )
    second_response = FakeListResponse(data=[], has_more=False, last_id=None)
    mock_client.vector_stores.list.side_effect = [first_response, second_response]

    result = store_service.list_store()

    assert result == [payload]
    second_call_kwargs = mock_client.vector_stores.list.call_args_list[1].kwargs
    assert second_call_kwargs == {"limit": 100, "after": "vs_3"}


def test_list_store_handles_empty_page_without_progress(mock_client, store_service) -> None:
    """Stop paginating when the API reports more data but provides no cursor."""

    class FakeListResponse:
        def __init__(self, data, has_more, last_id=None):
            self.data = data
            self.has_more = has_more
            self.last_id = last_id

    empty_page = FakeListResponse(data=[], has_more=True, last_id=None)
    mock_client.vector_stores.list.return_value = empty_page

    result = store_service.list_store()

    assert result == []
    mock_client.vector_stores.list.assert_called_once_with(limit=100, after=None)


def test_list_store_id_returns_ids(monkeypatch, store_service) -> None:
    """Extract vector store IDs including None for stores missing the field."""
    stores = [
        {"id": "vs_1"},
        {"id": "vs_2"},
        {"name": "missing"},
    ]
    monkeypatch.setattr(store_service, "list_store", MagicMock(return_value=stores))

    assert store_service._list_store_id() == ["vs_1", "vs_2", None]


def test_list_store_id_returns_empty_when_no_stores(monkeypatch, store_service) -> None:
    """Return an empty list when no vector stores are available."""
    monkeypatch.setattr(store_service, "list_store", MagicMock(return_value=[]))

    assert store_service._list_store_id() == []


def test_delete_returns_true_on_success(mock_client, store_service) -> None:
    """Return True when the delete call reports a successful deletion."""
    mock_client.vector_stores.delete.return_value = MagicMock(deleted=True)

    assert store_service.delete("vs_123") is True
    mock_client.vector_stores.delete.assert_called_once_with(vector_store_id="vs_123")


def test_delete_returns_false_when_not_deleted(mock_client, store_service) -> None:
    """Return False when the delete call indicates the store was not deleted."""
    mock_client.vector_stores.delete.return_value = MagicMock(deleted=False)

    assert store_service.delete("vs_123") is False
    mock_client.vector_stores.delete.assert_called_once_with(vector_store_id="vs_123")
