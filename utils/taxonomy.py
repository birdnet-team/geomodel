"""
Taxonomy and label management for BirdNET Geomodel.

Provides a unified lookup system for species metadata using the master
taxonomy.csv generated from species-data/ metadata.
"""

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class TaxonomyManager:
    """Manages species taxonomy and label mappings from a localized taxonomy.csv."""

    def __init__(self, taxonomy_path: Union[str, Path]):
        """Initialize the taxonomy manager.

        Args:
            taxonomy_path: Path to the master taxonomy.csv file.
        """
        self.taxonomy_path = Path(taxonomy_path)
        self.sci_to_meta: Dict[str, Dict[str, Any]] = {}
        self.code_to_meta: Dict[str, Dict[str, Any]] = {}
        
        if self.taxonomy_path.exists():
            self._load_taxonomy()
        else:
            logging.warning(f"Taxonomy file not found at {taxonomy_path}")

    def _load_taxonomy(self):
        """Load taxonomy from CSV and build lookup tables."""
        with open(self.taxonomy_path, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sci_name = row.get('sci_name', '').strip()
                code = row.get('species_code', '').strip()
                
                meta = {
                    'idx': row.get('idx', ''),
                    'sci_name': sci_name,
                    'com_name': row.get('com_name', sci_name),
                    'species_code': code,
                    'class_name': row.get('class_name', ''),
                    # Keep raw row for locale access if needed
                    'locales': {k: v for k, v in row.items() if k.startswith('common_name_')}
                }

                if sci_name:
                    self.sci_to_meta[sci_name.lower()] = meta
                if code:
                    self.code_to_meta[code.lower()] = meta

    def get_metadata_by_name(self, sci_name: str) -> Optional[Dict[str, Any]]:
        """Lookup metadata using scientific name."""
        return self.sci_to_meta.get(sci_name.lower())

    def get_primary_id(self, sci_name: str, fallback_gbif_key: Optional[int] = None) -> str:
        """Get the species code (eBird code or iNat ID or GBIF taxonKey)."""
        meta = self.get_metadata_by_name(sci_name)
        if meta and meta.get('species_code'):
            return str(meta['species_code'])
        return str(fallback_gbif_key) if fallback_gbif_key is not None else sci_name

    def get_label_line(self, sci_name: str, fallback_gbif_key: Optional[int] = None) -> str:
        """Generate a standardized labels.txt line: Code \t SciName \t ComName."""
        meta = self.get_metadata_by_name(sci_name.strip())
        if not meta:
            pid = str(fallback_gbif_key) if fallback_gbif_key is not None else sci_name
            return f"{pid}\t{sci_name}\t{sci_name}"
            
        return f"{meta['species_code']}\t{meta['sci_name']}\t{meta['com_name']}"
