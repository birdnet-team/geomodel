"""
Inference script for BirdNET Geomodel.

Load a checkpoint and predict species occurrence for a given location and week.

Usage:
    python predict.py --lat 52.5 --lon 13.4 --week 22
    python predict.py --checkpoint checkpoints/checkpoint_best.pt --lat 52.5 --lon 13.4 --week 22 --top_k 10
    python predict.py --lat 52.5 --lon 13.4 --week 22 --threshold 0.1

Common species mode — produce a globally representative species list:
    python predict.py --common_species --num_species 5000 --output common_species.txt
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from model.model import create_model


def load_labels(labels_path: str) -> Dict[int, Tuple[str, str, str]]:
    """
    Load labels.txt (saved by train.py) — one line per species in vocab order.
    Format: speciesCode<TAB>scientificName<TAB>commonName

    Returns:
        Dict mapping model output index → (speciesCode, scientificName, commonName)
    """
    labels: Dict[int, Tuple[str, str, str]] = {}
    path = Path(labels_path)
    if not path.exists():
        return labels

    with open(path, encoding='utf-8') as f:
        for idx, line in enumerate(f):
            parts = line.rstrip('\n').split('\t')
            if len(parts) >= 3:
                code, sci, com = parts[0], parts[1], parts[2]
            elif len(parts) == 2:
                code, sci = parts[0], parts[1]
                com = sci
            else:
                code = sci = com = parts[0]
            labels[idx] = (code, sci, com)
    return labels


def predict(
    checkpoint_path: str,
    lat: float,
    lon: float,
    week: int,
    top_k: Optional[int] = None,
    threshold: Optional[float] = None,
    device: str = 'auto',
) -> List[Tuple[str, str, str, float]]:
    """
    Predict species occurrence for a location and week.

    Returns list of (speciesCode, scientificName, commonName, probability) tuples,
    sorted by probability descending.
    """
    if device == 'auto':
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        dev = torch.device(device)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    model_config = ckpt['model_config']
    species_vocab = ckpt['species_vocab']
    idx_to_species = species_vocab['idx_to_species']

    # Reconstruct model
    model = create_model(
        n_species=model_config['n_species'],
        n_env_features=model_config['n_env_features'],
        model_scale=model_config.get('model_scale', 1.0),
        coord_harmonics=model_config.get('coord_harmonics', 8),
        week_harmonics=model_config.get('week_harmonics', 4),
        habitat_head=model_config.get('habitat_head', False),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(dev)
    model.eval()

    # Yearly (week 0): max predictions across all 48 weeks
    if week == 0:
        lat_batch = torch.full((48,), lat, dtype=torch.float32, device=dev)
        lon_batch = torch.full((48,), lon, dtype=torch.float32, device=dev)
        week_batch = torch.arange(1, 49, dtype=torch.float32, device=dev)
        with torch.no_grad():
            output = model(lat_batch, lon_batch, week_batch, return_env=False)
            probs = torch.sigmoid(output['species_logits']).cpu().numpy()  # (48, n_species)
        probs = probs.max(axis=0)  # yearly = max across weeks
    else:
        # Raw inputs — the model handles circular encoding internally
        lat_t = torch.tensor([lat], dtype=torch.float32, device=dev)
        lon_t = torch.tensor([lon], dtype=torch.float32, device=dev)
        week_t = torch.tensor([week], dtype=torch.float32, device=dev)
        with torch.no_grad():
            output = model(lat_t, lon_t, week_t, return_env=False)
            probs = torch.sigmoid(output['species_logits']).cpu().numpy()[0]

    # Load labels file (auto-detect from checkpoint dir)
    # Try: <checkpoint_stem>_labels.txt, then labels.txt
    ckpt_dir = Path(checkpoint_path).parent
    ckpt_stem = Path(checkpoint_path).stem
    labels_path = ckpt_dir / f'{ckpt_stem}_labels.txt'
    if not labels_path.exists():
        labels_path = ckpt_dir / 'labels.txt'
    labels = load_labels(str(labels_path)) if labels_path.exists() else {}
    if not labels:
        import warnings
        warnings.warn(
            f"No labels file found in {ckpt_dir} — output will use species "
            f"codes only. Expected: {ckpt_stem}_labels.txt or labels.txt",
            stacklevel=2,
        )

    # Build results
    results = []
    for idx_key, species_id in idx_to_species.items():
        idx = int(idx_key)
        species_id = str(species_id)
        prob = float(probs[idx])
        label = labels.get(idx)
        if label:
            code, sci_name, com_name = label
        else:
            code, sci_name, com_name = species_id, species_id, species_id
        results.append((code, sci_name, com_name, prob))

    results.sort(key=lambda x: x[3], reverse=True)

    if threshold is not None:
        results = [r for r in results if r[3] >= threshold]
    if top_k is not None:
        results = results[:top_k]

    return results


def _load_model_and_labels(checkpoint_path: str, device: str = 'auto'):
    """Load model, labels, and move to device. Returns (model, labels, dev)."""
    if device == 'auto':
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        dev = torch.device(device)

    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    model_config = ckpt['model_config']

    model = create_model(
        n_species=model_config['n_species'],
        n_env_features=model_config['n_env_features'],
        model_scale=model_config.get('model_scale', 1.0),
        coord_harmonics=model_config.get('coord_harmonics', 8),
        week_harmonics=model_config.get('week_harmonics', 4),
        habitat_head=model_config.get('habitat_head', False),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(dev)
    model.eval()

    ckpt_dir = Path(checkpoint_path).parent
    ckpt_stem = Path(checkpoint_path).stem
    labels_path = ckpt_dir / f'{ckpt_stem}_labels.txt'
    if not labels_path.exists():
        labels_path = ckpt_dir / 'labels.txt'
    labels = load_labels(str(labels_path)) if labels_path.exists() else {}

    return model, labels, dev


def _load_bird_indices(taxonomy_path: str, labels: Dict[int, Tuple[str, str, str]]) -> set:
    """Return set of label indices that are birds (class_name == 'aves')."""
    code_to_class = {}
    with open(taxonomy_path, encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row['species_code']
            cls = row.get('class_name', '').strip().lower()
            code_to_class[code] = cls

    bird_indices = set()
    for idx, (code, _, _) in labels.items():
        if code_to_class.get(code, '') == 'aves':
            bird_indices.add(idx)
    return bird_indices


def _compute_region_scores(
    model: torch.nn.Module,
    dev: torch.device,
    bbox: Tuple[float, float, float, float],
    n_species: int,
    grid_step: float = 5.0,
    weeks: Tuple[int, ...] = (1, 13, 26, 39),
    batch_size: int = 512,
) -> np.ndarray:
    """Compute mean species probabilities over a region's grid × weeks.

    Returns:
        np.ndarray of shape (n_species,) — mean probability per species.
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    lats = np.arange(lat_min + grid_step / 2, lat_max, grid_step)
    lons = np.arange(lon_min + grid_step / 2, lon_max, grid_step)
    if len(lats) == 0:
        lats = np.array([(lat_min + lat_max) / 2])
    if len(lons) == 0:
        lons = np.array([(lon_min + lon_max) / 2])

    grid_lat, grid_lon = np.meshgrid(lats, lons, indexing='ij')
    coords = np.stack([grid_lat.ravel(), grid_lon.ravel()], axis=1)  # (G, 2)
    n_points = len(coords)

    # Build full batch: every grid point × every week
    all_lat = np.tile(coords[:, 0], len(weeks))
    all_lon = np.tile(coords[:, 1], len(weeks))
    all_week = np.repeat(np.array(weeks, dtype=np.float32), n_points)

    accum = np.zeros(n_species, dtype=np.float64)
    total = len(all_lat)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        lat_t = torch.tensor(all_lat[start:end], dtype=torch.float32, device=dev)
        lon_t = torch.tensor(all_lon[start:end], dtype=torch.float32, device=dev)
        week_t = torch.tensor(all_week[start:end], dtype=torch.float32, device=dev)
        with torch.no_grad():
            out = model(lat_t, lon_t, week_t, return_env=False)
            probs = torch.sigmoid(out['species_logits']).cpu().numpy()
        accum += probs.sum(axis=0)

    return accum / total


def generate_common_species(
    checkpoint_path: str,
    num_species: int = 5000,
    grid_step: float = 5.0,
    taxonomy_path: Optional[str] = None,
    device: str = 'auto',
    output_path: Optional[str] = None,
) -> List[Tuple[str, str, str]]:
    """Generate a globally representative common species list.

    Algorithm:
        1. For each of ~24 non-overlapping global regions, compute mean species
           probabilities over a lat/lon grid at 4 representative weeks.
        2. Split species into birds and non-birds using taxonomy.
        3. Round-robin across regions: pick 4 birds + 1 non-bird per region per
           round until the target count is reached.

    Returns:
        List of (speciesCode, scientificName, commonName).
    """
    from utils.regions import GLOBAL_SAMPLING_REGIONS
    from utils.taxonomy import find_taxonomy_csv

    model, labels, dev = _load_model_and_labels(checkpoint_path, device)
    n_species = len(labels)

    if taxonomy_path is None:
        found = find_taxonomy_csv()
        if found is None:
            raise FileNotFoundError(
                "No taxonomy CSV found in ./ or data/. Pass --taxonomy "
                "explicitly (needed for the bird/non-bird split)."
            )
        taxonomy_path = str(found)

    print(f"Loading taxonomy from {taxonomy_path}")
    bird_indices = _load_bird_indices(taxonomy_path, labels)
    n_birds = len(bird_indices)
    n_other = n_species - n_birds
    print(f"  {n_birds} birds, {n_other} non-birds in vocabulary")

    regions = list(GLOBAL_SAMPLING_REGIONS.items())
    print(f"\nScoring {len(regions)} regions (grid_step={grid_step}°, weeks=1/13/26/39)")

    # Per-region ranked lists: birds and non-birds
    presence_threshold = 0.01  # species with mean prob > this are "present"
    region_bird_ranks: List[List[int]] = []
    region_other_ranks: List[List[int]] = []
    region_present: List[set] = []  # species indices present per region

    for i, (name, bbox) in enumerate(regions):
        print(f"  [{i+1}/{len(regions)}] {name} ...", end=" ", flush=True)
        scores = _compute_region_scores(model, dev, bbox, n_species, grid_step)

        present = {idx for idx in range(n_species) if scores[idx] > presence_threshold}
        region_present.append(present)

        bird_scores = [(idx, scores[idx]) for idx in range(n_species) if idx in bird_indices]
        other_scores = [(idx, scores[idx]) for idx in range(n_species) if idx not in bird_indices]

        bird_scores.sort(key=lambda x: x[1], reverse=True)
        other_scores.sort(key=lambda x: x[1], reverse=True)

        region_bird_ranks.append([idx for idx, _ in bird_scores])
        region_other_ranks.append([idx for idx, _ in other_scores])
        print(f"done ({len(present)} species present, "
              f"top bird: {labels[bird_scores[0][0]][2]}, "
              f"top other: {labels[other_scores[0][0]][2]})")

    # Round-robin selection: 4 birds + 1 other per region per round
    selected: set = set()
    result_order: List[int] = []
    region_contrib = [0] * len(regions)  # count of species contributed per region
    # Track position in each region's ranked list
    bird_pos = [0] * len(regions)
    other_pos = [0] * len(regions)

    print(f"\nRound-robin selection (target: {num_species} species) ...")
    while len(result_order) < num_species:
        made_progress = False
        for r in range(len(regions)):
            # Pick up to 4 birds from this region
            added_birds = 0
            while added_birds < 4 and bird_pos[r] < len(region_bird_ranks[r]):
                idx = region_bird_ranks[r][bird_pos[r]]
                bird_pos[r] += 1
                if idx not in selected:
                    selected.add(idx)
                    result_order.append(idx)
                    region_contrib[r] += 1
                    added_birds += 1
                    made_progress = True
                    if len(result_order) >= num_species:
                        break
            if len(result_order) >= num_species:
                break

            # Pick 1 non-bird from this region
            while other_pos[r] < len(region_other_ranks[r]):
                idx = region_other_ranks[r][other_pos[r]]
                other_pos[r] += 1
                if idx not in selected:
                    selected.add(idx)
                    result_order.append(idx)
                    region_contrib[r] += 1
                    made_progress = True
                    break
            if len(result_order) >= num_species:
                break

        if not made_progress:
            print(f"  Exhausted all regions at {len(result_order)} species")
            break

    total = len(result_order)
    print(f"\n  Regional coverage (species selected / species present):")
    for r, (name, _) in enumerate(regions):
        present = region_present[r]
        covered = len(present & selected)
        n_present = len(present)
        pct = 100.0 * covered / n_present if n_present else 0
        print(f"    {name:20s}  {covered:5d} / {n_present:5d}  ({pct:5.1f}%)")

    species_list = []
    for idx in result_order:
        code, sci, com = labels[idx]
        species_list.append((code, sci, com))

    print(f"  Selected {len(species_list)} species "
          f"({sum(1 for idx in result_order if idx in bird_indices)} birds, "
          f"{sum(1 for idx in result_order if idx not in bird_indices)} non-birds)")

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            for code, sci, com in species_list:
                f.write(f"{code}\t{sci}\t{com}\n")
        print(f"  Written to {output_path}")

    return species_list


def main():
    parser = argparse.ArgumentParser(description='Predict species with BirdNET Geomodel')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoint_best.pt', help='Path to model checkpoint')

    # Common species mode
    parser.add_argument('--common_species', action='store_true', help='Generate a globally representative common species list')
    parser.add_argument('--num_species', type=int, default=5000, help='Target number of species for --common_species')
    parser.add_argument('--grid_step', type=float, default=5.0, help='Grid spacing in degrees for --common_species')
    parser.add_argument('--taxonomy', type=str, default=None,
                        help='Path to taxonomy CSV for bird/non-bird split '
                             '(auto-detected from ./ or data/ if omitted)')
    parser.add_argument('--output', type=str, default=None, help='Output file path for --common_species')

    # Single-location mode
    parser.add_argument('--lat', type=float, default=None, help='Latitude (-90 to 90)')
    parser.add_argument('--lon', type=float, default=None, help='Longitude (-180 to 180)')
    parser.add_argument('--week', type=int, default=None, help='Week number (1-48, or -1/0 for yearly)')
    parser.add_argument('--top_k', type=int, default=100, help='Show top K species')
    parser.add_argument('--threshold', type=float, default=0.15, help='Min probability threshold')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    args = parser.parse_args()

    if args.common_species:
        generate_common_species(
            checkpoint_path=args.checkpoint,
            num_species=args.num_species,
            grid_step=args.grid_step,
            taxonomy_path=args.taxonomy,
            device=args.device,
            output_path=args.output,
        )
        return

    # Single-location mode: require lat, lon, week
    if args.lat is None or args.lon is None or args.week is None:
        parser.error("--lat, --lon, and --week are required (unless using --common_species)")

    if not (args.week in (-1, 0) or 1 <= args.week <= 48):
        parser.error("Week must be between 1 and 48, or -1/0 for yearly")

    # Map CLI -1 → internal week 0 (yearly)
    internal_week = 0 if args.week == -1 else args.week

    results = predict(
        checkpoint_path=args.checkpoint,
        lat=args.lat, lon=args.lon, week=internal_week,
        top_k=args.top_k, threshold=args.threshold,
        device=args.device,
    )

    # Print results
    week_label = "yearly" if args.week == -1 else f"week={args.week}"
    print(f"\nPredictions for lat={args.lat}, lon={args.lon}, {week_label}")
    print(f"{'Rank':<5} {'Code':<12} {'Probability':<12} {'Common Name':<30} {'Scientific Name'}")
    print("-" * 100)
    for i, (code, sci_name, com_name, prob) in enumerate(results, 1):
        print(f"{i:<5} {code:<12} {prob:<12.4f} {com_name:<30} {sci_name}")

    if not results:
        print("  (no species above threshold)")


if __name__ == '__main__':
    main()
