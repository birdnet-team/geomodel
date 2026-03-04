"""
Inference script for BirdNET Geomodel.

Load a checkpoint and predict species occurrence for a given location and week.

Usage:
    python predict.py --lat 52.5 --lon 13.4 --week 22
    python predict.py --checkpoint checkpoints/checkpoint_best.pt --lat 52.5 --lon 13.4 --week 22 --top_k 10
    python predict.py --lat 52.5 --lon 13.4 --week 22 --threshold 0.1
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from model.model import create_model


def load_labels(labels_path: str) -> Dict[int, Tuple[str, str]]:
    """
    Load labels.txt (saved by train.py) — one line per species in vocab order.
    Format: taxonKey<TAB>scientificName<TAB>commonName

    Returns:
        Dict mapping model output index → (scientificName, commonName)
    """
    labels: Dict[int, Tuple[str, str]] = {}
    path = Path(labels_path)
    if not path.exists():
        return labels

    with open(path, encoding='utf-8') as f:
        for idx, line in enumerate(f):
            parts = line.rstrip('\n').split('\t')
            if len(parts) >= 3:
                _, sci, com = parts[0], parts[1], parts[2]
            elif len(parts) == 2:
                _, sci = parts[0], parts[1]
                com = sci
            else:
                sci = com = parts[0]
            labels[idx] = (sci, com)
    return labels


def predict(
    checkpoint_path: str,
    lat: float,
    lon: float,
    week: int,
    top_k: Optional[int] = None,
    threshold: Optional[float] = None,
    device: str = 'auto',
) -> List[Tuple[int, str, str, float]]:
    """
    Predict species occurrence for a location and week.

    Returns list of (taxonKey, scientificName, commonName, probability) tuples,
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
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(dev)
    model.eval()

    # Yearly (week 0): max predictions across all 48 weeks
    if week == 0:
        import numpy as np
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
    labels_path = Path(checkpoint_path).parent / 'labels.txt'
    labels = load_labels(str(labels_path)) if labels_path.exists() else {}

    # Build results
    results = []
    for idx_key, taxon_key in idx_to_species.items():
        idx = int(idx_key)
        taxon_key = int(taxon_key)
        prob = float(probs[idx])
        sci_name, com_name = labels.get(idx, (str(taxon_key), str(taxon_key)))
        results.append((taxon_key, sci_name, com_name, prob))

    results.sort(key=lambda x: x[3], reverse=True)

    if threshold is not None:
        results = [r for r in results if r[3] >= threshold]
    if top_k is not None:
        results = results[:top_k]

    return results


def main():
    parser = argparse.ArgumentParser(description='Predict species with BirdNET Geomodel')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoint_best.pt', help='Path to model checkpoint')
    parser.add_argument('--lat', type=float, required=True, help='Latitude (-90 to 90)')
    parser.add_argument('--lon', type=float, required=True, help='Longitude (-180 to 180)')
    parser.add_argument('--week', type=int, required=True, help='Week number (1-48, or -1 for yearly)')
    parser.add_argument('--top_k', type=int, default=100, help='Show top K species')
    parser.add_argument('--threshold', type=float, default=0.15, help='Min probability threshold')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    args = parser.parse_args()

    if not (args.week == -1 or 1 <= args.week <= 48):
        parser.error("Week must be between 1 and 48, or -1 for yearly")

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
    print(f"{'Rank':<5} {'TaxonKey':<12} {'Probability':<12} {'Common Name':<30} {'Scientific Name'}")
    print("-" * 100)
    for i, (taxon_key, sci_name, com_name, prob) in enumerate(results, 1):
        print(f"{i:<5} {taxon_key:<12} {prob:<12.4f} {com_name:<30} {sci_name}")

    if not results:
        print("  (no species above threshold)")


if __name__ == '__main__':
    main()
