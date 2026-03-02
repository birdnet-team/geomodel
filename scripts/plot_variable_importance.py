"""
Plot variable importance for species predictions.

For each requested species, computes the Spearman rank correlation between
each variable (lat, lon, week, and all environmental features) and the
model's predicted occurrence probability across training data samples.
Produces one horizontal bar chart per species.

The script loads the training parquet, runs batched inference to obtain
predicted probabilities, then correlates those predictions with the raw
variable values.  This reveals which habitat / location variables the
model has learned to associate with each species — even though
environmental features are not direct model inputs.

Usage:
    python scripts/plot_variable_importance.py --species "Barn Swallow" "House Sparrow"
    python scripts/plot_variable_importance.py --taxon_keys 9750029
    python scripts/plot_variable_importance.py --species "European Robin" \
        --data_path /path/to/data.parquet \
        --max_samples 100000 --top_k 20
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

# Add project root to path so we can import project modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.model import create_model
from predict import load_labels
from utils.data import H3DataLoader


# ── Model loading ──────────────────────────────────────────────────────

def load_model_and_labels(checkpoint_path: str, device: torch.device):
    """Load model checkpoint and labels. Returns model, idx_to_species, labels dict."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = ckpt['model_config']
    species_vocab = ckpt['species_vocab']
    idx_to_species = species_vocab['idx_to_species']

    model = create_model(
        n_species=model_config['n_species'],
        n_env_features=model_config['n_env_features'],
        model_size=model_config['model_size'],
        coord_harmonics=model_config.get('coord_harmonics', 4),
        week_harmonics=model_config.get('week_harmonics', 2),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    labels_path = Path(checkpoint_path).parent / 'labels.txt'
    labels = load_labels(str(labels_path)) if labels_path.exists() else {}

    return model, idx_to_species, labels


# ── Species resolution ─────────────────────────────────────────────────

def resolve_species_indices(
    species_names: Optional[List[str]],
    taxon_keys: Optional[List[int]],
    idx_to_species: Dict,
    labels: Dict[int, Tuple[str, str]],
) -> List[Tuple[int, int, str, str]]:
    """Resolve requested species to model indices.

    Returns list of (model_index, taxonKey, sciName, comName).
    """
    taxon_to_idx = {int(v): int(k) for k, v in idx_to_species.items()}
    results = []

    if taxon_keys:
        for tk in taxon_keys:
            if tk in taxon_to_idx:
                idx = taxon_to_idx[tk]
                sci, com = labels.get(idx, (str(tk), str(tk)))
                results.append((idx, tk, sci, com))
            else:
                print(f"Warning: taxonKey {tk} not found in model vocabulary, skipping.")

    if species_names:
        for name_query in species_names:
            query_lower = name_query.lower().strip()
            found = False
            for idx_key, taxon_key in idx_to_species.items():
                idx = int(idx_key)
                sci, com = labels.get(idx, (str(taxon_key), str(taxon_key)))
                if query_lower in sci.lower() or query_lower in com.lower():
                    tk = int(taxon_key)
                    if not any(r[0] == idx for r in results):
                        results.append((idx, tk, sci, com))
                        found = True
                        break
            if not found:
                print(f"Warning: species '{name_query}' not found in labels, skipping.")

    return results


# ── Data loading ───────────────────────────────────────────────────────

def load_data_samples(
    data_path: str,
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load parquet and flatten to samples.

    Returns:
        lats, lons, weeks: 1-D float arrays
        env_matrix: (n_samples, n_env_features) raw feature values
        env_names: list of environmental feature column names
    """
    loader = H3DataLoader(data_path)
    loader.load_data()
    lats, lons, weeks, _, env_df = loader.flatten_to_samples()

    env_names = list(env_df.columns)
    env_matrix = env_df.values.astype(np.float32)

    # Drop uninformative columns (constant metadata)
    keep = [i for i, n in enumerate(env_names) if n not in DROP_COLUMNS]
    env_names = [env_names[i] for i in keep]
    env_matrix = env_matrix[:, keep]

    # Subsample if requested
    n = len(lats)
    if max_samples and max_samples < n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_samples, replace=False)
        idx.sort()
        lats = lats[idx]
        lons = lons[idx]
        weeks = weeks[idx]
        env_matrix = env_matrix[idx]

    return lats, lons, weeks, env_matrix, env_names


# ── Batched inference ──────────────────────────────────────────────────

def predict_probabilities(
    model: torch.nn.Module,
    lats: np.ndarray,
    lons: np.ndarray,
    weeks: np.ndarray,
    species_indices: List[int],
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    """
    Run inference for all samples and extract probabilities for requested species.

    Returns:
        probs: (n_samples, n_species) array of predicted probabilities
    """
    lat_t = torch.from_numpy(lats.astype(np.float32))
    lon_t = torch.from_numpy(lons.astype(np.float32))
    week_t = torch.from_numpy(weeks.astype(np.float32))

    all_probs = []
    n = len(lats)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        lb = lat_t[start:end].to(device)
        lnb = lon_t[start:end].to(device)
        wb = week_t[start:end].to(device)
        with torch.no_grad():
            output = model(lb, lnb, wb, return_env=False)
            logits = output['species_logits'][:, species_indices]
            probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)

    return np.concatenate(all_probs, axis=0)


# ── Correlation computation ───────────────────────────────────────────

PRETTY_NAMES = {
    'elevation': 'Elevation',
    'temperature_mean': 'Temperature (mean)',
    'temperature_range': 'Temperature (range)',
    'precipitation_mean': 'Precipitation (mean)',
    'precipitation_range': 'Precipitation (range)',
    'water_fraction': 'Water fraction',
    'urban_fraction': 'Urban fraction',
    'ndvi_mean': 'NDVI (mean)',
    'ndvi_range': 'NDVI (range)',
}

# MODIS MCD12Q1 LC_Type1 (IGBP) class names
IGBP_CLASS_NAMES = {
    1: 'Evergreen Needleleaf Forest',
    2: 'Evergreen Broadleaf Forest',
    3: 'Deciduous Needleleaf Forest',
    4: 'Deciduous Broadleaf Forest',
    5: 'Mixed Forest',
    6: 'Closed Shrublands',
    7: 'Open Shrublands',
    8: 'Woody Savannas',
    9: 'Savannas',
    10: 'Grasslands',
    11: 'Permanent Wetlands',
    12: 'Croplands',
    13: 'Urban / Built-up',
    14: 'Cropland / Vegetation Mosaic',
    15: 'Snow / Ice',
    16: 'Barren',
    17: 'Water Bodies',
}

# Columns to exclude — constant metadata, not meaningful for correlation
DROP_COLUMNS = {'h3_resolution', 'target_km'}

# Columns that should be one-hot encoded (categorical integers)
CATEGORICAL_COLUMNS = {'landcover_class'}


def expand_categoricals(
    env_matrix: np.ndarray,
    env_names: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    One-hot encode categorical columns (e.g. landcover_class) into individual
    binary indicator columns.  Non-categorical columns pass through unchanged.

    Returns:
        expanded_matrix, expanded_names
    """
    parts: List[np.ndarray] = []
    names: List[str] = []

    for col_idx, col_name in enumerate(env_names):
        col = env_matrix[:, col_idx]

        if col_name not in CATEGORICAL_COLUMNS:
            parts.append(col.reshape(-1, 1))
            names.append(col_name)
            continue

        # Determine class lookup
        if col_name == 'landcover_class':
            class_names = IGBP_CLASS_NAMES
        else:
            class_names = {}

        # Find unique non-NaN classes present in the data
        valid = col[~np.isnan(col)].astype(int)
        unique_classes = sorted(set(valid.tolist()))

        for cls in unique_classes:
            indicator = (col == cls).astype(np.float32)
            parts.append(indicator.reshape(-1, 1))
            label = class_names.get(cls, f'{col_name}_{cls}')
            names.append(f'LC: {label}' if col_name == 'landcover_class' else label)

    expanded = np.hstack(parts) if parts else env_matrix
    return expanded, names


def compute_correlations(
    variable_values: np.ndarray,
    variable_names: List[str],
    species_probs: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute Spearman rank correlation between each variable and species probs.

    Returns:
        correlations: (n_variables,) array of Spearman rho values
        names: corresponding variable names
    """
    n_vars = variable_values.shape[1]
    correlations = np.zeros(n_vars, dtype=np.float64)

    for i in range(n_vars):
        col = variable_values[:, i]
        # Skip columns with no variance (constant or all-NaN)
        valid = ~np.isnan(col)
        if valid.sum() < 10 or np.nanstd(col) < 1e-12:
            correlations[i] = 0.0
            continue
        rho, _ = stats.spearmanr(col[valid], species_probs[valid])
        correlations[i] = rho if np.isfinite(rho) else 0.0

    return correlations, variable_names


def prettify_name(name: str) -> str:
    """Make a variable name human-readable."""
    if name in PRETTY_NAMES:
        return PRETTY_NAMES[name]
    return name.replace('_', ' ').title()


# ── Plotting ───────────────────────────────────────────────────────────

# Semantic variable groups — order defines the top-to-bottom layout.
# Each entry: (group_label, list of raw variable name prefixes/exact matches)
VARIABLE_GROUPS = [
    ('Location',   ['latitude', 'longitude']),
    ('Climate',    ['temperature', 'precipitation']),
    ('Terrain',    ['elevation', 'canopy']),
    ('Surface',    ['water_fraction', 'urban_fraction']),
    ('Land Cover', ['LC:']),  # one-hot land cover columns start with "LC:"
]


def _group_order(
    variable_names: List[str],
) -> List[int]:
    """
    Return indices into *variable_names* sorted by semantic group.

    Variables within each group keep their original order.
    Any variable that doesn't match a group is appended at the end.
    """
    used: set = set()
    ordered: List[int] = []

    pretty = [prettify_name(n) for n in variable_names]

    for _, prefixes in VARIABLE_GROUPS:
        for idx, (raw, pn) in enumerate(zip(variable_names, pretty)):
            if idx in used:
                continue
            for pfx in prefixes:
                if raw.startswith(pfx) or pn.startswith(pfx):
                    ordered.append(idx)
                    used.add(idx)
                    break

    # Anything unmatched goes at the end
    for idx in range(len(variable_names)):
        if idx not in used:
            ordered.append(idx)

    return ordered


def plot_variable_importance(
    correlations: np.ndarray,
    variable_names: List[str],
    species_info: Tuple[int, int, str, str],
    outdir: str,
):
    """
    Plot a horizontal bar chart of variable–species correlations.

    Variables are arranged in semantic groups (Location, Climate, Terrain,
    Surface, Land Cover) so plots are comparable across species.

    Parameters:
        correlations: Spearman rho per variable
        variable_names: corresponding labels
        species_info: (model_idx, taxonKey, sciName, comName)
        outdir: output directory
    """
    _, taxon_key, sci_name, com_name = species_info

    # Order by semantic groups
    order = _group_order(variable_names)

    # Reverse so first group appears at the top of the horizontal bar chart
    order = order[::-1]

    sorted_corrs = correlations[order]
    sorted_names = [prettify_name(variable_names[i]) for i in order]

    n_bars = len(sorted_corrs)
    fig_height = max(4, 0.35 * n_bars + 1.5)
    fig, ax = plt.subplots(figsize=(8, fig_height))

    colors = ['#2166ac' if c > 0 else '#b2182b' for c in sorted_corrs]

    y_pos = np.arange(n_bars)
    ax.barh(y_pos, sorted_corrs, color=colors, edgecolor='white', linewidth=0.5, height=0.7)

    # Draw thin horizontal lines between groups
    raw_order = order[::-1]  # back to top-first for boundary detection
    pretty_ordered = [prettify_name(variable_names[i]) for i in raw_order]
    group_boundaries: List[Tuple[int, str]] = []  # (bar_index, group_label)
    prev_group = None
    for pos, (idx, pn) in enumerate(zip(raw_order, pretty_ordered)):
        raw = variable_names[idx]
        cur_group = None
        for grp_label, prefixes in VARIABLE_GROUPS:
            for pfx in prefixes:
                if raw.startswith(pfx) or pn.startswith(pfx):
                    cur_group = grp_label
                    break
            if cur_group:
                break
        if cur_group != prev_group and prev_group is not None:
            # pos is top-first index; convert to bar y position (reversed)
            bar_y = n_bars - pos - 0.5
            ax.axhline(bar_y, color='#999999', linewidth=0.6, linestyle='-', zorder=0)
        prev_group = cur_group

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names, fontsize=9)
    ax.set_xlabel('Spearman correlation with predicted probability', fontsize=10)

    title = com_name if com_name != sci_name else sci_name
    ax.set_title(f'{title}\n({sci_name})', fontsize=12, fontweight='bold')

    ax.axvline(0, color='#333333', linewidth=0.8, zorder=0)

    # Light gridlines
    ax.xaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Extend x-axis slightly for breathing room
    xlim = max(0.15, np.max(np.abs(sorted_corrs)) * 1.15)
    ax.set_xlim(-xlim, xlim)

    plt.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    safe_name = com_name.replace(' ', '_').replace('/', '_').lower()
    filename = f'variable_importance_{safe_name}.png'
    filepath = os.path.join(outdir, filename)
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filepath}")


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Plot variable importance (correlation) for species predictions',
    )

    # Species selection
    parser.add_argument('--species', type=str, nargs='+', default=None,
                        help='Species common or scientific names (substring match)')
    parser.add_argument('--taxon_keys', type=int, nargs='+', default=None,
                        help='GBIF taxonKey identifiers')

    # Data
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to training parquet file (with environmental features)')
    parser.add_argument('--max_samples', type=int, default=200_000,
                        help='Max samples to use for correlation (default: 200000)')

    # Model
    parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoint_best.pt',
                        help='Path to model checkpoint')

    # Output
    parser.add_argument('--outdir', type=str, default='outputs/plots/variable_importance',
                        help='Output directory')
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Inference batch size')

    args = parser.parse_args()

    if not args.species and not args.taxon_keys:
        parser.error('Provide at least one of --species or --taxon_keys')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, idx_to_species, labels = load_model_and_labels(args.checkpoint, device)

    # Resolve species
    species_list = resolve_species_indices(
        args.species, args.taxon_keys, idx_to_species, labels,
    )
    if not species_list:
        print("No valid species found. Exiting.")
        return

    print(f"Species: {len(species_list)}")
    for _, tk, sci, com in species_list:
        print(f"  {com} ({sci}) — taxonKey {tk}")

    # Load data
    print(f"\nLoading data from {args.data_path}...")
    lats, lons, weeks, env_matrix, env_names = load_data_samples(
        args.data_path, max_samples=args.max_samples,
    )
    print(f"  Samples: {len(lats):,}  |  Env features: {len(env_names)}")

    # Build combined variable matrix: [lat, lon, env_features...]
    # One-hot encode categorical columns (e.g. landcover_class)
    # Week is excluded — it's a model input, not a habitat variable.
    env_expanded, env_expanded_names = expand_categoricals(env_matrix, env_names)
    all_variable_values = np.column_stack([
        lats.astype(np.float32),
        lons.astype(np.float32),
        env_expanded,
    ])
    all_variable_names = ['latitude', 'longitude'] + env_expanded_names

    # Get model indices for batch inference
    model_indices = [sp[0] for sp in species_list]

    # Run inference
    print(f"\nRunning inference on {len(lats):,} samples...")
    probs = predict_probabilities(
        model, lats, lons, weeks, model_indices, device,
        batch_size=args.batch_size,
    )
    print(f"  Predictions shape: {probs.shape}")

    # Compute correlations and plot for each species
    print(f"\nComputing correlations and plotting...")
    for sp_idx, (model_idx, tk, sci, com) in enumerate(species_list):
        species_probs = probs[:, sp_idx]
        correlations, var_names = compute_correlations(
            all_variable_values, all_variable_names, species_probs,
        )
        plot_variable_importance(
            correlations, var_names,
            species_info=(model_idx, tk, sci, com),
            outdir=args.outdir,
        )

    print(f"\nDone. Plots saved to {args.outdir}/")


if __name__ == '__main__':
    main()
