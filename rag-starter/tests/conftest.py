"""Shared fixtures.

The heavy fixtures (loading and embedding the four demo documents) are
session-scoped: they are built once and reused. They use the offline
HashingEmbedder, so the whole test-suite needs no network and no API key.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.config import Config
from rag.embedders import HashingEmbedder
from rag.loaders import load_documents
from rag.pipeline import RagPipeline, build_store

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "data" / "docs"


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def docs_dir() -> Path:
    return DOCS_DIR


@pytest.fixture(scope="session")
def documents():
    return load_documents(DOCS_DIR)


@pytest.fixture(scope="session")
def config() -> Config:
    return Config()


@pytest.fixture(scope="session")
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture(scope="session")
def store(documents, embedder, config):
    return build_store(documents, embedder, config)


@pytest.fixture(scope="session")
def pipeline(store, embedder, config) -> RagPipeline:
    return RagPipeline(store, embedder, config)
