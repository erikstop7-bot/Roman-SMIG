"""
smig/catalogs — Phase 3 catalog ingestion and projection package.

Public re-exports for convenient access:

    from smig.catalogs import (
        CatalogProvider,
        StarRecord,
        MissingColumnError,
        BesanconProvider,
        RomanBulgeProvider,
        SyntheticCatalogProvider,
    )
"""
from smig.catalogs.base import CatalogProvider, MissingColumnError, StarRecord
from smig.catalogs.besancon import BesanconProvider
from smig.catalogs.roman_bulge import RomanBulgeProvider
from smig.catalogs.synthetic import SyntheticCatalogProvider

__all__ = [
    "CatalogProvider",
    "MissingColumnError",
    "StarRecord",
    "BesanconProvider",
    "RomanBulgeProvider",
    "SyntheticCatalogProvider",
]
