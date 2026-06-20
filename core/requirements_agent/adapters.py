from __future__ import annotations

import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def provenance(source: str, url_or_api: str = "", confidence: str = "low", license_note: str = "") -> Dict[str, Any]:
    return {
        "source": source,
        "url_or_api": url_or_api,
        "retrieval_date": datetime.now(timezone.utc).date().isoformat(),
        "license_or_usage_note": license_note or "Respect source Terms of Service, API limits, caching policy, and robots.txt where applicable.",
        "confidence": confidence,
    }


class ComponentDataAdapter(ABC):
    @abstractmethod
    def search_components(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        raise NotImplementedError


class DigikeyAdapter(ComponentDataAdapter):
    """Official-API placeholder.

    v1 does not crawl DigiKey. Credentials are read from the environment and
    missing fields stay unknown/null until an authorized API integration is added.
    """

    def __init__(self) -> None:
        self.client_id = os.getenv("DIGIKEY_CLIENT_ID")
        self.client_secret = os.getenv("DIGIKEY_CLIENT_SECRET")

    def search_components(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        return [{
            "query": query,
            "filters": filters or {},
            "status": "not_implemented_offline_placeholder",
            "credentials_configured": bool(self.client_id and self.client_secret),
            "provenance": provenance("DigiKey official integration placeholder", "", "low"),
        }]


class NexarOctopartAdapter(ComponentDataAdapter):
    def __init__(self) -> None:
        self.client_id = os.getenv("NEXAR_CLIENT_ID")
        self.client_secret = os.getenv("NEXAR_CLIENT_SECRET")

    def search_components(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        return [{
            "query": query,
            "filters": filters or {},
            "status": "not_implemented_offline_placeholder",
            "credentials_configured": bool(self.client_id and self.client_secret),
            "provenance": provenance("Nexar/Octopart integration placeholder", "", "low"),
        }]


class LiteratureSearchAdapter(ABC):
    @abstractmethod
    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        raise NotImplementedError


class OfflineLiteratureAdapter(LiteratureSearchAdapter):
    source_name = "offline literature placeholder"
    api_url = ""

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        return [{
            "query": query,
            "title": "Offline fixture only",
            "status": "not_implemented_offline_placeholder",
            "provenance": provenance(self.source_name, self.api_url, "low"),
        }][:limit]


class ArxivAdapter(OfflineLiteratureAdapter):
    source_name = "arXiv integration placeholder"
    api_url = ""


class SemanticScholarAdapter(OfflineLiteratureAdapter):
    source_name = "Semantic Scholar integration placeholder"
    api_url = ""


class OpenAlexAdapter(OfflineLiteratureAdapter):
    source_name = "OpenAlex integration placeholder"
    api_url = ""


class CrossrefAdapter(OfflineLiteratureAdapter):
    source_name = "Crossref integration placeholder"
    api_url = ""


class DatasheetIngestionService:
    def ingest(self, url: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            "url": url,
            "metadata": metadata or {},
            "status": "not_implemented_offline_placeholder",
            "provenance": provenance("datasheet ingestion placeholder", url, "low"),
        }


class RagIndexService:
    def index_items(self, items: List[Dict[str, Any]], namespace: str = "requirements") -> Dict[str, Any]:
        return {
            "namespace": namespace,
            "count": len(items),
            "status": "not_implemented_offline_placeholder",
            "provenance": provenance("RAG index placeholder", "local/offline", "low"),
        }
