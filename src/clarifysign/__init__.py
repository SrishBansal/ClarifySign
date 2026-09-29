"""ClarifySign production interfaces.

This package is deliberately provider-agnostic: local development uses mocks and
production adapters are selected through configuration.
"""

from clarifysign.api.service import ClarifySignService
from clarifysign.config.settings import AppSettings

__all__ = ["AppSettings", "ClarifySignService"]
