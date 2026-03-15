from __future__ import annotations
import os
from dataclasses import dataclass

@dataclass
class ServiceSettings:
    host: str = os.getenv("RAG_API_HOST", "127.0.0.1")
    port: int = int(os.getenv("RAG_API_PORT", "8008"))
    service: str = "medical-rag-api"
    version: str = "0.1.0"

settings = ServiceSettings()
