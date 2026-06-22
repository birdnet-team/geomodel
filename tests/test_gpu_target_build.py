"""COO + GPU-scatter species target must equal the legacy dense CPU collate."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from utils.data import _make_sparse_collate_fn, _make_coo_collate_fn

N = 200  # species
def make_batch(B=16, region=False, seed=0):
    g = np.random.default_rng(seed)
    batch = []
    for i in range(B):
        k = int(g.integers(1, 8))
        idx = np.sort(g.choice(N, size=k, replace=False)).astype(np.int32)
        inp = {'lat': torch.tensor(float(g.uniform(-60,60))),
               'lon': torch.tensor(float(g.uniform(-180,180))),
               'week': torch.tensor(float(g.integers(1,49)))}
        if region:
            inp['region_id'] = int(g.integers(0, 60))
        batch.append((inp, {'species_indices': idx, 'env_features': torch.zeros(3)}))
    return batch

def rebuild_coo(targets):  # mirrors Trainer._build_species_target (deterministic part, cpu)
    B, ns = int(targets['species_B']), int(targets['species_n'])
    sp = torch.zeros(B, ns)
    if targets['species_rows'].numel() > 0:
        sp[targets['species_rows'], targets['species_cols']] = targets['species_vals']
    return sp

def check(name, **kw):
    b = make_batch(**{k:v for k,v in kw.items() if k in ('B','region','seed')})
    dense = _make_sparse_collate_fn(N, **{k:v for k,v in kw.items() if k.startswith(('species','ubi'))})(b)[1]['species']
    coo = rebuild_coo(_make_coo_collate_fn(N, **{k:v for k,v in kw.items() if k.startswith(('species','ubi'))})(b)[1])
    d = (dense - coo).abs().max().item()
    print(f"{name:<24} max|dense-coo| = {d:.3e}  {'OK' if d==0.0 else 'FAIL'}")
    assert d == 0.0

print("=== COO target vs legacy dense (deterministic, ubiquitous off) ===")
check("binary")
fw = torch.rand(N)*0.9 + 0.1
check("freq_weights", species_freq_weights=fw)
rw = torch.rand(60, N)*0.9 + 0.1
check("region_weights", region=True, species_region_weights=rw)
print("All COO target-build equivalence tests passed.")
