"""MongoDB persistence for registration metadata.

The registration pipeline remains file-based.  This module stores only the
run document and references to files produced under ``runs/<run_id>``.
"""

from __future__ import annotations

import os
from datetime import date, datetime
from typing import Any

from bson import ObjectId
from pymongo import ASCENDING, DESCENDING, MongoClient


MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DATABASE = os.getenv("MONGO_DATABASE", "lunar")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "lunarimage")
MONGO_SERVER_SELECTION_TIMEOUT_MS = int(os.getenv("MONGO_SERVER_SELECTION_TIMEOUT_MS", "3000"))

_client: MongoClient | None = None
_collection: Any | None = None
_indexes_ready = False


def _get_collection() -> Any:
    global _client, _collection, _indexes_ready
    if _collection is None:
        _client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=MONGO_SERVER_SELECTION_TIMEOUT_MS,
            connectTimeoutMS=MONGO_SERVER_SELECTION_TIMEOUT_MS,
        )
        _collection = _client[MONGO_DATABASE][MONGO_COLLECTION]
    if not _indexes_ready:
        _collection.create_index([("run_id", ASCENDING)], unique=True, name="run_id_unique")
        _collection.create_index([("created_at", DESCENDING)], name="created_at_desc")
        _collection.create_index([("status", ASCENDING)], name="status_idx")
        _indexes_ready = True
    return _collection


def ping() -> dict[str, Any]:
    client = MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=MONGO_SERVER_SELECTION_TIMEOUT_MS,
        connectTimeoutMS=MONGO_SERVER_SELECTION_TIMEOUT_MS,
    )
    result = client.admin.command("ping")
    return {
        "status": "connected",
        "database": MONGO_DATABASE,
        "collection": MONGO_COLLECTION,
        "server_version": client.server_info().get("version"),
        "ping": result.get("ok") == 1.0,
    }


def insert(document: dict[str, Any]) -> dict[str, Any]:
    result = _get_collection().insert_one(document)
    return {
        "status": "stored",
        "database": MONGO_DATABASE,
        "collection": MONGO_COLLECTION,
        "document_id": str(result.inserted_id),
    }


def get(run_id: str) -> dict[str, Any] | None:
    document = _get_collection().find_one({"run_id": run_id}, {"_id": 0})
    return _json_safe(document) if document is not None else None


def list_recent(limit: int = 20, skip: int = 0) -> list[dict[str, Any]]:
    bounded_limit = max(1, min(int(limit), 100))
    bounded_skip = max(0, int(skip))
    documents = _get_collection().find({}, {"_id": 0}).sort("created_at", DESCENDING).skip(bounded_skip).limit(bounded_limit)
    return [_json_safe(document) for document in documents]


def config_summary() -> dict[str, str]:
    return {
        "uri": MONGO_URI,
        "database": MONGO_DATABASE,
        "collection": MONGO_COLLECTION,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
