"""Stage 2: the model's species list, from the BirdNET taxonomy, with GBIF keys to match records.

    python -m pipeline.taxonomy BirdNET+_Taxonomy_v0.5-Nov2026.csv data/species.csv [--cache data/gbif_keys.json]

The BirdNET taxonomy has one row per class with ``birdnet_id``,
``scientific_name``, ``taxon_group``, ``gbif_id``, ``ebird_code``,
``scientific_name_aliases`` and more. It is published as a CSV file in the BirdNET
model converter repository (https://github.com/birdnet-team/model-converter); the
JSON export of the same taxonomy (``{"rows": [...]}``) works as well. Every model
output is a taxonomy species with a ``birdnet_id``; sound classes (record type
other than ``species``) have no geography and are skipped.

GBIF files every occurrence under the *accepted* species key, so each species is
resolved to that key:

1. with a ``gbif_id``: the accepted species key of that name usage (synonyms move
   to their accepted species);
2. without one: an exact species-rank match of the scientific name, then of each
   alias.

A key is only given to one species. When two taxonomy species end up on the same
key, GBIF lumps them (typically a recent split); the species whose own ``gbif_id``
is that key keeps it, and the other one gets no key. Borrowing the parent's records
would give a split species the whole range of its parent. Species without a key
have no training data and are left out of the model.

GBIF lookups are cached in ``--cache``, so a re-run only asks for new names.

Output columns: ``birdnet_id, scientific_name, common_name, taxon_group,
ebird_code, gbif_key, how`` (how the key was found, or why there is none) and
``aliases`` (former scientific names, ``|``-separated; stage 3 matches reported names
against them too).
Cartilaginous fish join the bony-fish group (``Actinopterygii``): there are few of
them and they share observers.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import time
import urllib.parse
import urllib.request
from collections import defaultdict

GBIF_API = 'https://api.gbif.org/v1'
USER_AGENT = 'birdnet-geomodel/4 (+https://github.com/birdnet-team/geomodel)'
GROUP_ALIASES = {'Chondrichthyes': 'Actinopterygii'}
FIELDS = ['birdnet_id', 'scientific_name', 'common_name', 'taxon_group', 'ebird_code', 'gbif_key', 'how', 'aliases']


def load_taxonomy(path: str | pathlib.Path, species_only: bool = True) -> list[dict]:
    """Rows of the BirdNET taxonomy (CSV or JSON export); by default only ``record_type == 'species'``."""
    if str(path).lower().endswith('.csv'):
        csv.field_size_limit(10 ** 9)
        with open(path, newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
    else:
        with open(path, encoding='utf-8') as f:
            rows = json.load(f)['rows']
    return [r for r in rows if r.get('record_type') == 'species'] if species_only else rows


def aliases(row: dict) -> list[str]:
    """Scientific-name aliases of a taxonomy row (list or ``|``/``;``-separated string)."""
    value = row.get('scientific_name_aliases') or []
    if isinstance(value, str):
        value = value.replace(';', '|').split('|')
    return [a.strip() for a in value if a and a.strip()]


class GbifKeys:
    """Cached GBIF species lookups."""

    def __init__(self, cache_path: str | None = None, pause: float = 0.1):
        self.path = pathlib.Path(cache_path) if cache_path else None
        self.cache = json.loads(self.path.read_text()) if self.path and self.path.exists() else {}
        self.pause = pause

    def _get(self, url: str) -> dict:
        if url in self.cache:
            return self.cache[url]
        request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    data = json.load(response)
                break
            except Exception as error:  # network hiccups: retry with back-off, then give up on this name
                if attempt == 3:
                    print(f'GBIF lookup failed: {url} ({error})', flush=True)
                    return {}
                time.sleep(2 ** attempt)
        keep = {k: data.get(k) for k in ('speciesKey', 'matchType', 'rank')}
        self.cache[url] = keep
        time.sleep(self.pause)
        return keep

    def accepted_key(self, gbif_id: int) -> int | None:
        """Accepted species key of a GBIF name usage."""
        key = self._get(f'{GBIF_API}/species/{gbif_id}').get('speciesKey')
        return int(key) if key else None

    def match(self, name: str) -> int | None:
        """Species key of an exact, species-rank (or subspecies) name match."""
        query = urllib.parse.urlencode({'name': name, 'strict': 'true'})
        data = self._get(f'{GBIF_API}/species/match?{query}')
        if data.get('matchType') == 'EXACT' and data.get('rank') in ('SPECIES', 'SUBSPECIES') and data.get('speciesKey'):
            return int(data['speciesKey'])
        return None

    def save(self) -> None:
        if self.path:
            self.path.write_text(json.dumps(self.cache))


def resolve(rows: list[dict], gbif: GbifKeys) -> list[dict]:
    """One output row per taxonomy species with its GBIF key (or the reason it has none)."""
    found: dict[str, tuple[int | None, str]] = {}
    for i, row in enumerate(rows):
        key, how = None, 'not found'
        if row.get('gbif_id'):
            key = gbif.accepted_key(int(row['gbif_id']))
            how = 'gbif_id' if key == int(row['gbif_id']) else 'gbif_id, accepted key'
        if key is None:
            for name in [row['scientific_name'], *aliases(row)]:
                key = gbif.match(name)
                if key is not None:
                    how = 'name' if name == row['scientific_name'] else 'alias'
                    break
            else:
                how = 'not found'
        found[row['birdnet_id']] = (key, how)
        if i % 1000 == 0:
            print(f'{i} / {len(rows)} species', flush=True)
            gbif.save()
    gbif.save()

    owners = {int(r['gbif_id']): r['birdnet_id'] for r in rows if r.get('gbif_id')}
    claims = defaultdict(set)
    for birdnet_id, (key, _) in found.items():
        if key is not None:
            claims[key].add(birdnet_id)

    out = []
    for row in rows:
        key, how = found[row['birdnet_id']]
        if key is not None and len(claims[key]) > 1 and owners.get(key) != row['birdnet_id']:
            others = sorted(claims[key] - {row['birdnet_id']})
            key, how = None, 'key shared with ' + ' '.join(others)
        out.append(dict(
            birdnet_id=row['birdnet_id'], scientific_name=row['scientific_name'],
            common_name=row.get('common_name') or '',
            taxon_group=GROUP_ALIASES.get(row['taxon_group'], row['taxon_group']),
            ebird_code=row.get('ebird_code') or '', gbif_key=key or '', how=how,
            aliases='|'.join(aliases(row))))
    return out


def read_species(path: str | pathlib.Path) -> list[dict]:
    """Read the stage-2 species table."""
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('taxonomy', help='BirdNET taxonomy (CSV or JSON export)')
    parser.add_argument('out', help='species table to write (CSV)')
    parser.add_argument('--cache', help='JSON file that caches GBIF lookups between runs')
    args = parser.parse_args(argv)

    rows = load_taxonomy(args.taxonomy)
    table = resolve(rows, GbifKeys(args.cache))
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(table)

    by_group: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in table:
        by_group[r['taxon_group']][0] += 1
        by_group[r['taxon_group']][1] += bool(r['gbif_key'])
    for group, (n, keyed) in sorted(by_group.items(), key=lambda x: -x[1][0]):
        print(f'{group:16s} {n:6d} species, {keyed:6d} with a GBIF key')


if __name__ == '__main__':
    main()
