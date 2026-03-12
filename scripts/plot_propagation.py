"""
Compare species lists before and after environmental label propagation.

Produces a visual report showing:
  1. Per-week bar chart of species counts (before vs after)
  2. Detailed species list diff for a chosen location and week
  3. Summary statistics (cells modified, species added, etc.)

Usage:
    # Overview: per-week species count comparison for a location
    python scripts/plot_propagation.py --data_path outputs/combined.parquet --lat 50.83 --lon 12.92

    # Detailed diff for a specific week
    python scripts/plot_propagation.py --data_path outputs/combined.parquet --lat 50.83 --lon 12.92 --week 1

    # Adjust propagation parameters
    python scripts/plot_propagation.py --data_path outputs/combined.parquet --lat 50.83 --lon 12.92 \
        --propagate_k 5 --propagate_max_radius 2000 --propagate_min_obs 3
"""

import argparse
import copy
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.data import H3DataLoader, H3DataPreprocessor


def load_taxonomy(taxonomy_path: Optional[str] = None) -> Dict[str, Tuple[str, str]]:
    """Load taxonomy CSV mapping speciesCode → (scientificName, commonName).

    Auto-detects taxonomy.csv or data/taxonomy.csv if *taxonomy_path* is not
    provided, falling back to checkpoints/labels.txt (tab-separated
    code/sciName/comName format).
    """
    taxonomy: Dict[str, Tuple[str, str]] = {}

    if taxonomy_path is None:
        for candidate in ['taxonomy.csv', 'data/taxonomy.csv']:
            if Path(candidate).exists():
                taxonomy_path = candidate
                break

    if taxonomy_path is None or not Path(taxonomy_path).exists():
        # Fallback: labels.txt (code\tsciName\tcomName)
        labels_path = Path('checkpoints/labels.txt')
        if labels_path.exists():
            with open(labels_path, encoding='utf-8') as f:
                for line in f:
                    parts = line.rstrip('\n').split('\t')
                    if len(parts) >= 3:
                        taxonomy[parts[0]] = (parts[1], parts[2])
            return taxonomy
        return taxonomy

    with open(taxonomy_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = (row.get('species_code', '') or '').strip()
            sci = (row.get('sci_name', '') or row.get('scientificName', '') or '').strip()
            com = (row.get('com_name', '') or row.get('commonName', '') or sci).strip()
            if code and sci:
                taxonomy[code] = (sci, com)

    return taxonomy


def find_nearest_cell(
    target_lat: float,
    target_lon: float,
    lats: np.ndarray,
    lons: np.ndarray,
) -> int:
    """Find the index of the first sample at the nearest H3 cell centre."""
    # H3 cells repeat across weeks, so find the unique location nearest to target
    d = (lats - target_lat) ** 2 + (lons - target_lon) ** 2
    # Get unique locations
    coords = np.column_stack([lats, lons])
    _, unique_idx = np.unique(coords, axis=0, return_index=True)
    nearest_unique = unique_idx[np.argmin(d[unique_idx])]
    return int(nearest_unique)


def get_cell_samples(
    cell_lat: float,
    cell_lon: float,
    lats: np.ndarray,
    lons: np.ndarray,
) -> np.ndarray:
    """Return indices of all samples at this cell location."""
    mask = (lats == cell_lat) & (lons == cell_lon)
    return np.where(mask)[0]


def species_name(species_id: str, taxonomy: Dict[str, Tuple[str, str]]) -> str:
    """Format a species name string."""
    if species_id in taxonomy:
        sci, com = taxonomy[species_id]
        return f"{com} ({sci})" if com != sci else sci
    return str(species_id)


def plot_weekly_comparison(
    weeks: np.ndarray,
    before: List[List[int]],
    after: List[List[int]],
    cell_indices: np.ndarray,
    cell_lat: float,
    cell_lon: float,
    outdir: str,
):
    """Bar chart: species count per week, before vs after propagation."""
    # Collect per-week counts
    week_nums = []
    counts_before = []
    counts_after = []

    for idx in cell_indices:
        wk = int(weeks[idx])
        if wk == 0:
            continue  # skip yearly
        week_nums.append(wk)
        counts_before.append(len(before[idx]))
        counts_after.append(len(after[idx]))

    # Sort by week
    order = np.argsort(week_nums)
    week_nums = [week_nums[i] for i in order]
    counts_before = [counts_before[i] for i in order]
    counts_after = [counts_after[i] for i in order]

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(week_nums))
    width = 0.38

    bars_b = ax.bar(x - width / 2, counts_before, width, label='Before propagation',
                    color='#4A90D9', alpha=0.85, edgecolor='white', linewidth=0.5)
    bars_a = ax.bar(x + width / 2, counts_after, width, label='After propagation',
                    color='#E8913A', alpha=0.85, edgecolor='white', linewidth=0.5)

    # Highlight weeks where propagation added species
    for i, (b, a) in enumerate(zip(counts_before, counts_after)):
        if a > b:
            ax.annotate(f'+{a - b}', xy=(x[i] + width / 2, a),
                        ha='center', va='bottom', fontsize=7, color='#C05E10',
                        fontweight='bold')

    ax.set_xlabel('Week')
    ax.set_ylabel('Species count')
    ax.set_title(f'Label Propagation Effect — ({cell_lat:.2f}°, {cell_lon:.2f}°)')
    ax.set_xticks(x)
    ax.set_xticklabels(week_nums, fontsize=7)
    ax.legend(loc='upper right')
    ax.set_xlim(-0.8, len(week_nums) - 0.2)

    total_added = sum(a - b for a, b in zip(counts_after, counts_before))
    weeks_modified = sum(1 for a, b in zip(counts_after, counts_before) if a > b)
    ax.text(0.02, 0.95, f'Total species added: {total_added}\n'
            f'Weeks modified: {weeks_modified}/48',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.7))

    plt.tight_layout()
    out_path = Path(outdir) / 'propagation_weekly.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close(fig)


def print_species_diff(
    week: int,
    idx: int,
    before: List[List[int]],
    after: List[List[int]],
    taxonomy: Dict[int, Tuple[str, str]],
    cell_lat: float,
    cell_lon: float,
):
    """Print detailed species list diff for one sample."""
    before_set = set(before[idx])
    after_set = set(after[idx])
    added = after_set - before_set
    kept = before_set & after_set

    print(f"\n{'=' * 72}")
    print(f"Species diff — ({cell_lat:.2f}°, {cell_lon:.2f}°), week {week}")
    print(f"{'=' * 72}")
    print(f"  Before: {len(before_set)} species")
    print(f"  After:  {len(after_set)} species (+{len(added)} propagated)")
    print()

    if kept:
        print(f"  Original species ({len(kept)}):")
        for tk in sorted(kept):
            print(f"    • {species_name(tk, taxonomy)}")

    if added:
        print(f"\n  Propagated species ({len(added)}):")
        for tk in sorted(added):
            print(f"    + {species_name(tk, taxonomy)}")
    else:
        print("  No species were propagated for this week.")

    print()


def plot_global_summary(
    weeks: np.ndarray,
    before: List[List[int]],
    after: List[List[int]],
    outdir: str,
):
    """Histogram of species added per sample across the entire dataset."""
    added = np.array([len(set(a) - set(b)) for a, b in zip(after, before)])

    modified = added > 0
    n_modified = modified.sum()
    n_total = len(added)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: histogram of species added (only modified samples)
    ax = axes[0]
    if n_modified > 0:
        vals = added[modified]
        bins = min(50, int(vals.max()))
        ax.hist(vals, bins=max(bins, 1), color='#E8913A', alpha=0.85,
                edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Species added per sample')
    ax.set_ylabel('Number of samples')
    ax.set_title(f'Distribution of Propagated Labels\n'
                 f'({n_modified:,}/{n_total:,} samples modified)')

    # Right: species added per week (aggregated)
    ax = axes[1]
    week_added = {}
    for i in range(len(weeks)):
        wk = int(weeks[i])
        if wk == 0:
            continue
        if wk not in week_added:
            week_added[wk] = 0
        week_added[wk] += added[i]

    if week_added:
        wks = sorted(week_added.keys())
        vals = [week_added[w] for w in wks]
        ax.bar(wks, vals, color='#E8913A', alpha=0.85, edgecolor='white',
               linewidth=0.5)
        ax.set_xlabel('Week')
        ax.set_ylabel('Total species added')
        ax.set_title('Propagated Labels by Week (all cells)')

    plt.tight_layout()
    out_path = Path(outdir) / 'propagation_summary.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize label propagation before/after comparison.')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Combined parquet file')
    parser.add_argument('--lat', type=float, default=None,
                        help='Latitude for per-cell comparison')
    parser.add_argument('--lon', type=float, default=None,
                        help='Longitude for per-cell comparison')
    parser.add_argument('--week', type=int, default=None,
                        help='Show detailed species diff for this week')
    parser.add_argument('--taxonomy', type=str, default=None,
                        help='Taxonomy CSV for species name lookup')
    parser.add_argument('--propagate_k', type=int, default=5,
                        help='Number of env-space neighbors (default: 5)')
    parser.add_argument('--propagate_max_radius', type=float, default=2000.0,
                        help='Max geographic radius in km (default: 2000)')
    parser.add_argument('--propagate_min_obs', type=int, default=3,
                        help='Sparsity threshold (default: 3)')
    parser.add_argument('--propagate_max_spread', type=float, default=2.0,
                        help='Restrict propagation distance by species range radius factor')
    parser.add_argument('--no_yearly', action='store_true',
                        help='Exclude yearly (week 0) samples')
    parser.add_argument('--outdir', type=str, default='outputs/plots/propagation',
                        help='Output directory for plots')
    args = parser.parse_args()

    taxonomy = load_taxonomy(args.taxonomy)
    print(f"Loaded taxonomy: {len(taxonomy):,} species")

    # Load and flatten data
    print("Loading data...")
    loader = H3DataLoader(args.data_path)
    loader.load_data()

    print("Flattening to samples...")
    lats, lons, weeks, species_lists, env_features = loader.flatten_to_samples(
        include_yearly=not args.no_yearly,
    )
    print(f"  {len(species_lists):,} samples "
          f"({len(np.unique(np.column_stack([lats, lons]), axis=0)):,} unique locations)")

    # Deep copy species lists so we have before/after
    species_before = [list(sl) if hasattr(sl, '__iter__') else [] for sl in species_lists]
    species_after = copy.deepcopy(species_before)

    # Run propagation on the copy
    print("Running label propagation...")
    
    H3DataPreprocessor.propagate_env_labels(
        lats, lons, weeks, species_after, env_features,
        k=args.propagate_k,
        max_radius_km=args.propagate_max_radius,
        min_obs_threshold=args.propagate_min_obs,
        max_spread_factor=args.propagate_max_spread,
    )

    # Global summary plot
    print("\nGenerating global summary...")
    plot_global_summary(weeks, species_before, species_after, args.outdir)

    # Per-cell plots if location specified
    if args.lat is not None and args.lon is not None:
        nearest_idx = find_nearest_cell(args.lat, args.lon, lats, lons)
        cell_lat = float(lats[nearest_idx])
        cell_lon = float(lons[nearest_idx])

        print(f"\nNearest cell: ({cell_lat:.4f}°, {cell_lon:.4f}°)")

        cell_indices = get_cell_samples(cell_lat, cell_lon, lats, lons)
        print(f"  {len(cell_indices)} samples at this cell")

        # Weekly comparison bar chart
        plot_weekly_comparison(
            weeks, species_before, species_after,
            cell_indices, cell_lat, cell_lon, args.outdir,
        )

        # Detailed diff for a specific week
        if args.week is not None:
            # Find sample for this week at this cell
            for idx in cell_indices:
                if int(weeks[idx]) == args.week:
                    print_species_diff(
                        args.week, idx, species_before, species_after,
                        taxonomy, cell_lat, cell_lon,
                    )
                    break
            else:
                print(f"  No sample found for week {args.week} at this cell.")
        else:
            # Print summary diffs for all weeks that changed
            changed_weeks = []
            for idx in cell_indices:
                wk = int(weeks[idx])
                before_set = set(species_before[idx])
                after_set = set(species_after[idx])
                n_added = len(after_set - before_set)
                if n_added > 0:
                    changed_weeks.append((wk, n_added, len(before_set), len(after_set)))

            if changed_weeks:
                print(f"\n  Weeks with propagated species:")
                for wk, n_added, n_before, n_after in sorted(changed_weeks):
                    print(f"    Week {wk:2d}: {n_before:3d} → {n_after:3d} (+{n_added})")
                print(f"\n  Use --week N for detailed species diff.")
            else:
                print(f"\n  No propagation occurred at this cell "
                      f"(all weeks have ≥{args.propagate_min_obs} species).")

    print("\nDone.")


if __name__ == '__main__':
    main()
