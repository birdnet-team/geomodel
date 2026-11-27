"""The geomodel as a filter for audio detections: re-score the 2025 app study.

    python -m evaluation.filter_study CHECKPOINT PAIRS.parquet [--taxonomy TAXONOMY.csv] [--labels LABELS.txt]
                                      [--lines 0.05 0.1 0.15] [--out result.json]

In the apps the geomodel removes classifier detections of species it scores
below a line at the recording's place and week. This script asks what a given
geomodel would remove from real recordings.

``PAIRS.parquet`` holds app recordings from 2025 with the classifier's top three
bird predictions each, one row per prediction: ``rank`` (1-3), ``sci``
(scientific name), ``score`` (classifier confidence), ``latitude``, ``longitude``,
``week`` (geomodel week 1-48) and ``app_scientific`` (the species the app showed).
There are no verified identifications, so two proxies stand in for them:

* likely true: rank-1 prediction with classifier score >= 0.9 that agrees with
  the species the app showed;
* likely false: rank-2 or rank-3 prediction with classifier score < 0.3.

Reported per line: ``false_reject``, the share of likely-true detections the
geomodel would remove (the costly error); ``removed``, the share of likely-false
ones it removes; and the share removed at each rank. Predictions whose species
the model does not have pass unfiltered; they are left out of all shares, and
the share of predictions that could be matched is printed.

The geomodel is read through ``model.io``: its app scores (score curve, floors
and caps of release checkpoints; the default curve for students). Species are
matched by scientific name: V3 releases through the second column of their
label file (``--labels``), checkpoints with BirdNET ids through the taxonomy
names and aliases (``--taxonomy``).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from evaluation.common import columns_by_name

LIKELY_TRUE_SCORE = 0.9
LIKELY_FALSE_SCORE = 0.3


def geomodel_scores(predictor, pairs: pd.DataFrame, column: dict[str, int], batch: int = 2048) -> np.ndarray:
    """Score of each prediction's species at its place and week; NaN where the species is not matched.

    The three predictions of a recording share place and week, so the model runs
    once per distinct (lat, lng, week).
    """
    col = pairs.sci.map(column)
    matched = col.notna().to_numpy()
    rows = np.flatnonzero(matched)
    cols = col.to_numpy()[rows].astype(np.int64)

    points = pairs.loc[matched, ['latitude', 'longitude', 'week']].to_numpy(np.float64)
    unique, point_of_row = np.unique(points, axis=0, return_inverse=True)
    point_of_row = point_of_row.ravel()

    out = np.full(len(pairs), np.nan)
    for first in range(0, len(unique), batch):
        chunk = unique[first:first + batch]
        scores = predictor.scores(chunk[:, 0], chunk[:, 1], chunk[:, 2])
        in_chunk = np.flatnonzero((point_of_row >= first) & (point_of_row < first + batch))
        out[rows[in_chunk]] = scores[point_of_row[in_chunk] - first, cols[in_chunk]]
    return out


def evaluate(pairs: pd.DataFrame, geo: np.ndarray, lines: list[float]) -> dict:
    matched = pd.Series(~np.isnan(geo), index=pairs.index)
    likely_true = (pairs['rank'] == 1) & (pairs.score >= LIKELY_TRUE_SCORE) & (pairs.sci == pairs.app_scientific)
    likely_false = (pairs['rank'] >= 2) & (pairs.score < LIKELY_FALSE_SCORE)
    result = dict(matched=float(matched.mean()), n_likely_true=int((likely_true & matched).sum()),
                  n_likely_false=int((likely_false & matched).sum()), lines={})
    for line in lines:
        rejected = matched & (geo < line)
        result['lines'][str(line)] = dict(
            false_reject=float(rejected[likely_true & matched].mean()),
            removed=float(rejected[likely_false & matched].mean()),
            **{f'rank{k}': float(rejected[(pairs['rank'] == k) & matched].mean()) for k in (1, 2, 3)})
    return result


def print_result(result: dict, title: str) -> None:
    print(f'{title}: species matched for {result["matched"] * 100:.1f} % of predictions; '
          f'likely true {result["n_likely_true"]}, likely false {result["n_likely_false"]}')
    print(' line | false rejections (likely true) | removed (likely false) | rejected rank 1 / 2 / 3')
    for line, r in result['lines'].items():
        print(f' {float(line):4.2f} | {r["false_reject"] * 100:29.1f} % | {r["removed"] * 100:21.1f} % | '
              f'{r["rank1"] * 100:5.1f} / {r["rank2"] * 100:4.1f} / {r["rank3"] * 100:4.1f} %')


def main(argv: list[str] | None = None) -> None:
    from model.io import load
    from pipeline.taxonomy import load_taxonomy

    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('checkpoint', help='student or release checkpoint')
    parser.add_argument('pairs', help='parquet of recordings x top-3 classifier predictions')
    parser.add_argument('--taxonomy', help='BirdNET taxonomy (CSV or JSON), for checkpoints that use BirdNET ids')
    parser.add_argument('--labels', help='V3 label file (scientific name in the second column)')
    parser.add_argument('--lines', nargs='+', type=float, default=[0.05, 0.1, 0.15], help='geomodel score lines')
    parser.add_argument('--out', help='write the result as JSON')
    args = parser.parse_args(argv)

    predictor = load(args.checkpoint)
    if predictor.kind == 'teacher':
        raise SystemExit('teachers need habitat features; use a student or release checkpoint')
    taxonomy = load_taxonomy(args.taxonomy) if args.taxonomy else None
    column = columns_by_name(predictor, labels=args.labels, taxonomy=taxonomy)

    pairs = pd.read_parquet(args.pairs, columns=['rank', 'sci', 'score', 'latitude', 'longitude', 'week',
                                                  'app_scientific'])
    geo = geomodel_scores(predictor, pairs, column)
    result = evaluate(pairs, geo, args.lines)
    print_result(result, args.checkpoint)
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(dict(checkpoint=args.checkpoint, **result), f, indent=1)


if __name__ == '__main__':
    main()
