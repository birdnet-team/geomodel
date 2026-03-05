#!/usr/bin/env python3
"""Collect ablation experiment results and render them as Markdown tables.

Scans ``checkpoints/ablation/*/training_history.json`` for completed runs,
extracts best-epoch metrics, and writes a summary table compatible with
the ablation report (``report/hypotheses.md``).

Usage:
    python scripts/collect_ablation_results.py
    python scripts/collect_ablation_results.py --dir checkpoints/ablation
    python scripts/collect_ablation_results.py --format csv --output results.csv
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


# Mapping from directory name to run metadata.
# Keys: (ID, Run Name, Loss, Scale, Env Wt, Freq Wt)
RUN_META = {
    'baseline':   ('baseline', 'ASL s1.0', 'asl', '1.0', '0.1', 'off'),
    'A1a_bce':    ('A1a', 'BCE',       'bce',   '1.0', '0.1', 'off'),
    'A1b_asl':    ('A1b', 'ASL',       'asl',   '1.0', '0.1', 'off'),
    'A1c_focal':  ('A1c', 'Focal',     'focal', '1.0', '0.1', 'off'),
    'A1d_an':     ('A1d', 'AN',        'an',    '1.0', '0.1', 'off'),
    'A2a_env0':   ('A2a', 'env=0',     'asl',   '1.0', '0.0', 'off'),
    'A2b_env001': ('A2b', 'env=0.01',  'asl',   '1.0', '0.01','off'),
    'A2c_env01':  ('A2c', 'env=0.1',   'asl',   '1.0', '0.1', 'off'),
    'A2d_env05':  ('A2d', 'env=0.5',   'asl',   '1.0', '0.5', 'off'),
    'A3a_s05':    ('A3a', 's=0.5',     'asl',   '0.5', '0.1', 'off'),
    'A3b_s10':    ('A3b', 's=1.0',     'asl',   '1.0', '0.1', 'off'),
    'A3c_s20':    ('A3c', 's=2.0',     'asl',   '2.0', '0.1', 'off'),
    'A4a_s05_noenv': ('A4a', 's0.5 no-env', 'asl', '0.5', '0.0', 'off'),
    'A4b_s05_env':   ('A4b', 's0.5 env',   'asl', '0.5', '0.1', 'off'),
    'A4c_s10_noenv': ('A4c', 's1.0 no-env', 'asl', '1.0', '0.0', 'off'),
    'A4d_s10_env':   ('A4d', 's1.0 env',   'asl', '1.0', '0.1', 'off'),
    'A4e_s20_noenv': ('A4e', 's2.0 no-env', 'asl', '2.0', '0.0', 'off'),
    'A4f_s20_env':   ('A4f', 's2.0 env',   'asl', '2.0', '0.1', 'off'),
    'A5a_nowt':   ('A5a', 'no-wt',     'asl',   '1.0', '0.1', 'off'),
    'A5b_wt01':   ('A5b', 'wt min=0.1','asl',   '1.0', '0.1', '0.1'),
    'A5c_wt001':  ('A5c', 'wt min=0.01','asl',  '1.0', '0.1', '0.01'),
    'A6a_ch4':    ('A6a', 'h=4',       'asl',   '1.0', '0.1', 'off'),
    'A6b_ch8':    ('A6b', 'h=8',       'asl',   '1.0', '0.1', 'off'),
    'A6c_ch16':   ('A6c', 'h=16',      'asl',   '1.0', '0.1', 'off'),
    'A7a_wh2':    ('A7a', 'wh=2',      'asl',   '1.0', '0.1', 'off'),
    'A7b_wh4':    ('A7b', 'wh=4',      'asl',   '1.0', '0.1', 'off'),
    'A7c_wh8':    ('A7c', 'wh=8',      'asl',   '1.0', '0.1', 'off'),
    'A8a_nojitter': ('A8a', 'no jitter','asl',   '1.0', '0.1', 'off'),
    'A8b_jitter':   ('A8b', 'jitter',  'asl',   '1.0', '0.1', 'off'),
    'A9a_yearly':   ('A9a', 'with yearly','asl', '1.0', '0.1', 'off'),
    'A9b_noyearly': ('A9b', 'no yearly','asl',   '1.0', '0.1', 'off'),
    'A10a_gn2':   ('A10a', 'γ-=2',     'asl',   '1.0', '0.1', 'off'),
    'A10b_gn4':   ('A10b', 'γ-=4',     'asl',   '1.0', '0.1', 'off'),
    'A10c_gn6':   ('A10c', 'γ-=6',     'asl',   '1.0', '0.1', 'off'),
}

# Watchlist taxonKeys for per-species AP (must match train.py WATCHLIST_SPECIES)
WATCHLIST_TAXONKEYS = {
    # Hawaiian
    5232445: 'Hawaiian Goose',
    2480528: 'Hawaiian Hawk',
    2486699: 'Hawaii Elepaio',
    2494524: 'Apapane',
    8070758: 'Iiwi',
    8346110: 'Hawaii Amakihi',
    # NZ
    2479593: 'Kea',
    2495144: 'North Island Brown Kiwi',
    5228153: 'South Island Takahe',
    5229954: 'Rifleman',
    2487029: 'Tui',
    5817136: 'North Island Kokako',
    # Galápagos
    2480569: 'Galápagos Hawk',
    2474597: 'Galápagos Rail',
    2481460: 'Galápagos Petrel',
    # Other
    2474354: 'Kagu',
    2481920: 'California Condor',
    2474941: 'Whooping Crane',
}


def _fmt(val: Any, precision: int = 4) -> str:
    """Format a metric value for display."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return ''
    if isinstance(val, float):
        return f'{val:.{precision}f}'
    return str(val)


def load_best_metrics(history_path: Path) -> Optional[Dict[str, Any]]:
    """Load training_history.json and return metrics at the best mAP epoch."""
    try:
        with open(history_path) as f:
            hist = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return None

    val_map = hist.get('val_map', [])
    if not val_map:
        return None

    best_epoch = int(max(range(len(val_map)), key=lambda i: val_map[i]))
    n_epochs = len(val_map)

    result: Dict[str, Any] = {'best_epoch': best_epoch + 1, 'total_epochs': n_epochs}

    # Extract scalar metrics at best epoch
    for key in ('val_map', 'val_top10_recall', 'val_top30_recall',
                'val_f1_5', 'val_f1_10', 'val_f1_25',
                'val_list_ratio_5', 'val_list_ratio_10', 'val_list_ratio_25',
                'val_watchlist_mean_ap'):
        vals = hist.get(key, [])
        result[key] = vals[best_epoch] if best_epoch < len(vals) else None

    # Per-species AP at best epoch
    for tk in WATCHLIST_TAXONKEYS:
        key = f'val_ap_{tk}'
        vals = hist.get(key, [])
        result[key] = vals[best_epoch] if best_epoch < len(vals) else None

    return result


def collect_results(base_dir: Path) -> List[Dict[str, Any]]:
    """Scan all subdirectories and collect best-epoch results."""
    results = []
    for run_dir in sorted(base_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        history_path = run_dir / 'training_history.json'
        if not history_path.exists():
            continue

        dir_name = run_dir.name
        metrics = load_best_metrics(history_path)
        if metrics is None:
            continue

        meta = RUN_META.get(dir_name, (dir_name, dir_name, '?', '?', '?', '?'))
        results.append({
            'dir': dir_name,
            'id': meta[0],
            'run_name': meta[1],
            'loss': meta[2],
            'scale': meta[3],
            'env_wt': meta[4],
            'freq_wt': meta[5],
            **metrics,
        })

    return results


def render_summary_table(results: List[Dict[str, Any]]) -> str:
    """Render the main summary markdown table."""
    header = ('| ID | Run Name | Loss | Scale | Env Wt | Freq Wt '
              '| mAP | Top-10 | Top-30 '
              '| F1@5% | F1@10% | F1@25% '
              '| LR@5% | LR@10% | LR@25% '
              '| Epochs |')
    sep = '|' + '|'.join(['----'] * 16) + '|'

    lines = [header, sep]
    for r in results:
        row = (f"| {r['id']} | {r['run_name']} | {r['loss']} | {r['scale']} "
               f"| {r['env_wt']} | {r['freq_wt']} "
               f"| {_fmt(r.get('val_map'))} "
               f"| {_fmt(r.get('val_top10_recall'))} "
               f"| {_fmt(r.get('val_top30_recall'))} "
               f"| {_fmt(r.get('val_f1_5'))} "
               f"| {_fmt(r.get('val_f1_10'))} "
               f"| {_fmt(r.get('val_f1_25'))} "
               f"| {_fmt(r.get('val_list_ratio_5'), 2)} "
               f"| {_fmt(r.get('val_list_ratio_10'), 2)} "
               f"| {_fmt(r.get('val_list_ratio_25'), 2)} "
               f"| {r.get('total_epochs', '')} |")
        lines.append(row)
    return '\n'.join(lines)


def render_endemic_table(results: List[Dict[str, Any]],
                         species_group: Dict[int, str],
                         group_name: str) -> str:
    """Render per-species AP table for a group of endemic species.

    Only includes loss-comparison runs (A1a–A1d or baseline).
    """
    loss_runs = [r for r in results if r['id'] in ('A1a', 'A1b', 'A1c', 'A1d', 'baseline')]
    if not loss_runs:
        return f'*No loss-comparison runs found for {group_name}.*\n'

    # Build columns: BCE, ASL, Focal, AN
    loss_map = {}
    for r in loss_runs:
        loss_map[r['loss']] = r

    header = f'| Species | TaxonKey | BCE | ASL | Focal | AN |'
    sep = '|---------|----------|-----|-----|-------|----|'
    lines = [f'**{group_name}:**\n', header, sep]

    for tk, name in species_group.items():
        row_parts = [f'| {name} | {tk}']
        for loss in ('bce', 'asl', 'focal', 'an'):
            r = loss_map.get(loss)
            val = r.get(f'val_ap_{tk}') if r else None
            row_parts.append(f' {_fmt(val)}')
        lines.append(' |'.join(row_parts) + ' |')

    return '\n'.join(lines)


def render_csv(results: List[Dict[str, Any]]) -> str:
    """Render results as CSV."""
    import csv
    import io

    cols = ['id', 'run_name', 'loss', 'scale', 'env_wt', 'freq_wt',
            'val_map', 'val_top10_recall', 'val_top30_recall',
            'val_f1_5', 'val_f1_10', 'val_f1_25',
            'val_list_ratio_5', 'val_list_ratio_10', 'val_list_ratio_25',
            'total_epochs', 'val_watchlist_mean_ap']

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=cols, extrasaction='ignore')
    writer.writeheader()
    for r in results:
        writer.writerow(r)
    return buf.getvalue()


def main():
    parser = argparse.ArgumentParser(
        description='Collect ablation experiment results into summary tables')
    parser.add_argument('--dir', type=str, default='checkpoints/ablation',
                        help='Base directory containing experiment subdirectories')
    parser.add_argument('--format', choices=['markdown', 'csv'], default='markdown',
                        help='Output format (default: markdown)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (default: stdout)')
    args = parser.parse_args()

    base_dir = Path(args.dir)
    if not base_dir.exists():
        print(f'No results directory found: {base_dir}', file=sys.stderr)
        print(f'Run ablation experiments first, then re-run this script.', file=sys.stderr)
        sys.exit(1)

    results = collect_results(base_dir)
    if not results:
        print(f'No completed experiments found in {base_dir}', file=sys.stderr)
        sys.exit(1)

    print(f'Found {len(results)} completed experiments in {base_dir}\n',
          file=sys.stderr)

    if args.format == 'csv':
        output = render_csv(results)
    else:
        sections = []
        sections.append('## Ablation Results (auto-collected)\n')
        sections.append('### Summary Table\n')
        sections.append(render_summary_table(results))
        sections.append('')

        # Endemic species tables
        hawaiian = {tk: n for tk, n in WATCHLIST_TAXONKEYS.items()
                    if n.startswith('Hawa') or n in ('Apapane', 'Iiwi')}
        nz = {tk: n for tk, n in WATCHLIST_TAXONKEYS.items()
              if any(kw in n for kw in ('Kea', 'Kiwi', 'Takahe', 'Rifleman', 'Tui', 'Kokako'))}
        galapagos = {tk: n for tk, n in WATCHLIST_TAXONKEYS.items()
                     if 'Gal' in n}
        other = {tk: n for tk, n in WATCHLIST_TAXONKEYS.items()
                 if tk not in hawaiian and tk not in nz and tk not in galapagos}

        sections.append('\n### Endemic Species AP (Loss Comparison)\n')
        for group, name in [(hawaiian, 'Hawaiian endemics'),
                            (nz, 'New Zealand endemics'),
                            (galapagos, 'Galápagos endemics'),
                            (other, 'Other restricted-range')]:
            sections.append(render_endemic_table(results, group, name))
            sections.append('')

        # Watchlist mean AP
        wl_rows = [(r['id'], r['run_name'], r.get('val_watchlist_mean_ap'))
                    for r in results if r.get('val_watchlist_mean_ap') is not None]
        if wl_rows:
            sections.append('\n### Watchlist Mean AP\n')
            sections.append('| ID | Run | Mean AP (18 spp.) |')
            sections.append('|----|-----|-------------------|')
            for rid, rname, ap in wl_rows:
                sections.append(f'| {rid} | {rname} | {_fmt(ap)} |')
            sections.append('')

        output = '\n'.join(sections)

    if args.output:
        Path(args.output).write_text(output, encoding='utf-8')
        print(f'Results written to {args.output}', file=sys.stderr)
    else:
        print(output)


if __name__ == '__main__':
    main()
