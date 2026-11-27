"""Helpers shared by the evaluation scripts: which model output is which species,
and predictions for a few outputs at many points.

Checkpoints name their outputs in one of two ways:

* V4 teachers, students and releases use BirdNET ids (``BN04673``). Names and
  eBird codes come from the BirdNET taxonomy (CSV or JSON); students also carry their
  eBird codes in the checkpoint (``codes``).
* V3 releases use the first column of their label file: the eBird code for
  birds, a number for other taxa. The label file has the scientific name in the
  second column and the common name in the third.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

Predict = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def read_labels(path: str) -> list[list[str]]:
    """Rows of a label file (tab-separated: id, scientific name, common name)."""
    with open(path, encoding='utf-8') as f:
        return [line.rstrip('\n').split('\t') for line in f if line.strip()]


def is_birdnet_id(name: str) -> bool:
    return name.startswith('BN') and name[2:].isdigit()


def uses_birdnet_ids(predictor) -> bool:
    return all(is_birdnet_id(s) for s in predictor.species)


def taxonomy_by_id(rows: list[dict]) -> dict[str, dict]:
    """Taxonomy rows keyed by BirdNET id."""
    return {r['birdnet_id']: r for r in rows}


def ebird_codes(predictor, labels: str | None = None,
                code_by_id: dict[str, str] | None = None) -> list[str]:
    """eBird code of every output ('' where there is none).

    Taken from the checkpoint (students), else the first column of a V3 label
    file, else ``code_by_id`` (BirdNET id -> eBird code), else from the output
    names themselves when they are V3 ids.
    """
    if predictor.extra.get('codes') is not None:
        return [str(c) if c and str(c) not in ('None', 'nan') else '' for c in predictor.extra['codes']]
    if labels:
        first = [row[0] for row in read_labels(labels)]
        if len(first) != len(predictor.species):
            raise ValueError(f'{labels} has {len(first)} rows, the model {len(predictor.species)} outputs')
        if not all(is_birdnet_id(c) for c in first):
            return [c if not c.isdigit() else '' for c in first]
    if uses_birdnet_ids(predictor):
        if code_by_id is None:
            raise ValueError('the checkpoint uses BirdNET ids; pass the taxonomy to find eBird codes')
        return [code_by_id.get(s) or '' for s in predictor.species]
    return [s if not s.isdigit() else '' for s in predictor.species]


def columns_by_name(predictor, labels: str | None = None,
                    taxonomy: list[dict] | None = None) -> dict[str, int]:
    """Scientific name -> output column.

    Outputs named by BirdNET id are matched through the taxonomy: its scientific
    names first, then its aliases (an alias never takes a name that is some
    species' own). V3 outputs are matched by the exact name in the label file.
    """
    if uses_birdnet_ids(predictor) and taxonomy is not None:
        from pipeline.taxonomy import aliases

        column = {b: i for i, b in enumerate(predictor.species)}
        rows = [r for r in taxonomy if r['birdnet_id'] in column]
        out: dict[str, int] = {}
        for r in rows:
            out.setdefault(r['scientific_name'], column[r['birdnet_id']])
        for r in rows:
            for name in aliases(r):
                out.setdefault(name, column[r['birdnet_id']])
        return out
    if labels is None:
        raise ValueError('pass the label file (or, for BirdNET ids, the taxonomy) to match species by name')
    out = {}
    for i, row in enumerate(read_labels(labels)):
        if len(row) > 1:
            out.setdefault(row[1], i)
    return out


def display_names(predictor, labels: str | None = None,
                  taxonomy: list[dict] | None = None) -> list[str]:
    """Common name per output where known, else the scientific name, else the id."""
    if uses_birdnet_ids(predictor) and taxonomy is not None:
        by_id = taxonomy_by_id(taxonomy)
        return [(by_id.get(s) or {}).get('common_name') or (by_id.get(s) or {}).get('scientific_name') or s
                for s in predictor.species]
    if labels:
        return [row[2] if len(row) > 2 and row[2] else row[1] if len(row) > 1 else row[0]
                for row in read_labels(labels)]
    return list(predictor.species)


def sigmoid_of_logits(predictor) -> Predict:
    """``sigmoid(logits)``: the encounter rate for V4 models, the score for V3 releases.

    No score curve, floor or cap, so per species the order of the predictions is
    that of the network output.
    """
    def predict(lat, lng, week):
        z = predictor.logits(lat, lng, week)
        with np.errstate(over='ignore'):  # exp(-z) = inf for very negative logits gives 0, as it should
            return 1.0 / (1.0 + np.exp(-z))
    return predict


def predict_columns(predict: Predict, lat: Sequence[float], lng: Sequence[float], week: Sequence[float],
                    columns: Sequence[int], batch: int = 2048) -> np.ndarray:
    """``predict`` at each point, keeping only ``columns``: [n_points, len(columns)].

    The network always computes every output, so points are sent in batches to
    keep memory bounded (a batch of 2,048 points over 17,000 outputs is ~140 MB).
    """
    lat, lng, week = (np.asarray(a, np.float32) for a in (lat, lng, week))
    columns = np.asarray(columns, np.int64)
    out = np.empty((len(lat), len(columns)), np.float32)
    for i in range(0, len(lat), batch):
        j = slice(i, i + batch)
        out[j] = predict(lat[j], lng[j], week[j])[:, columns]
    return out
