"""
Taxonomy and label management for BirdNET Geomodel.

Provides a unified lookup system for species metadata using the master
taxonomy CSV (e.g. ``taxonomy_v0.2-Jun2026.csv``) generated from
species-data/ metadata by ``species-data/generate_taxonomy.py``.
"""

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


# Directories searched by :func:`find_taxonomy_csv`, in order.
_TAXONOMY_SEARCH_DIRS = (Path('.'), Path('data'))


def find_taxonomy_csv(
    search_dirs=_TAXONOMY_SEARCH_DIRS,
) -> Optional[Path]:
    """Locate the master taxonomy CSV without hardcoding a release name.

    Master taxonomies are versioned by filename (``taxonomy_v0.2-Jun2026.csv``),
    so callers must not pin one: a release bump would otherwise silently break
    every default. Within each directory an unsuffixed ``taxonomy.csv`` wins,
    otherwise the highest-sorting ``taxonomy_*.csv`` is used.

    Returns:
        Path to the taxonomy CSV, or ``None`` when none is found.
    """
    for directory in search_dirs:
        directory = Path(directory)
        exact = directory / 'taxonomy.csv'
        if exact.is_file():
            return exact
        versioned = sorted(p for p in directory.glob('taxonomy_*.csv')
                           if p.is_file())
        if versioned:
            return versioned[-1]
    return None


class TaxonomyManager:
    """Manages species taxonomy and label mappings from a localized taxonomy CSV."""

    # Default scientific-name remap file, relative to the repository root.
    DEFAULT_REMAP_PATH = Path(__file__).resolve().parent.parent / 'species-data' / 'species_remap.csv'

    def __init__(
        self,
        taxonomy_path: Union[str, Path],
        remap_path: Optional[Union[str, Path]] = None,
    ):
        """Initialize the taxonomy manager.

        Args:
            taxonomy_path: Path to the master taxonomy CSV file.
            remap_path: Optional path to a scientific-name remap CSV. When
                omitted, :data:`DEFAULT_REMAP_PATH` is used if it exists. Pass
                an empty string to disable remapping entirely.
        """
        self.taxonomy_path = Path(taxonomy_path)
        self.sci_to_meta: Dict[str, Dict[str, Any]] = {}
        self.code_to_meta: Dict[str, Dict[str, Any]] = {}
        # Direct species-code -> species-code redirects, used to remap an
        # already-combined parquet at train time (see remap_species_lists).
        self.code_remap: Dict[str, str] = {}

        if remap_path is None:
            self.remap_path: Optional[Path] = self.DEFAULT_REMAP_PATH
        elif remap_path == '':
            self.remap_path = None
        else:
            self.remap_path = Path(remap_path)

        if self.taxonomy_path.is_file():
            self._load_taxonomy()
        elif str(taxonomy_path):
            logging.warning(f"Taxonomy file not found at {taxonomy_path}")

        self._apply_remap()

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

    def _apply_remap(self):
        """Redirect ambiguous scientific names to canonical species codes.

        Handles cases where a taxonomic split leaves observation databases
        (GBIF, iNaturalist) still recording occurrences under a pre-split
        lumped scientific name. Each remap row maps a ``from_sci_name`` to a
        ``to_species_code``; the source name's metadata lookup is repointed to
        the target code's metadata, so every observation carrying that verbatim
        name is attributed to the intended post-split species.
        """
        if self.remap_path is None or not self.remap_path.exists():
            return

        applied = 0
        with open(self.remap_path, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                src = (row.get('from_sci_name') or '').strip()
                dst_code = (row.get('to_species_code') or '').strip()
                from_code = (row.get('from_species_code') or '').strip()
                if not dst_code:
                    continue

                # Species-code redirect (drives the train-time remap of an
                # already-combined parquet). Built directly from the file so it
                # does not depend on the loaded taxonomy containing either code.
                if not from_code and src:
                    orig = self.sci_to_meta.get(src.lower())
                    if orig:
                        from_code = str(orig.get('species_code') or '').strip()
                if from_code and from_code != dst_code:
                    self.code_remap[from_code] = dst_code

                # Scientific-name override (drives combine-time attribution).
                # A code-only row has nothing to override, so it does not
                # count toward the scientific-name remap tally below.
                if not src:
                    continue
                target = self.code_to_meta.get(dst_code.lower())
                if target is None:
                    # Only meaningful when a taxonomy was actually loaded; at
                    # train time (code redirect only) this is expected silence.
                    if self.code_to_meta:
                        logging.warning(
                            f"Species remap target code '{dst_code}' not found "
                            f"in taxonomy; scientific-name remap of '{src}' "
                            f"skipped (code redirect still recorded)"
                        )
                    continue
                self.sci_to_meta[src.lower()] = target
                applied += 1

        if applied:
            logging.info(
                f"Applied {applied} scientific-name remap(s) from "
                f"{self.remap_path.name}"
            )

    def remap_species_lists(self, species_lists: List[List[str]]) -> int:
        """Rewrite resolved species codes in place using :attr:`code_remap`.

        Lets an already-combined parquet (whose week columns store resolved
        species codes) pick up remap corrections at train time, avoiding a full
        re-run of ``combine.py``. Codes merged onto an existing target are
        de-duplicated per sample.

        Args:
            species_lists: Per-sample lists of species codes, mutated in place.

        Returns:
            Number of samples whose code list changed.
        """
        if not self.code_remap:
            return 0

        cm = self.code_remap
        changed = 0
        for i, sl in enumerate(species_lists):
            if not any(code in cm for code in sl):
                continue
            seen: set = set()
            new_list: List[str] = []
            for code in sl:
                mapped = cm.get(code, code)
                if mapped not in seen:
                    seen.add(mapped)
                    new_list.append(mapped)
            species_lists[i] = new_list
            changed += 1
        return changed

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
