# Species

```bash
python -m pipeline.taxonomy taxonomy.csv data/species.csv --cache data/gbif_keys.json
```

The model's species are the species of the BirdNET taxonomy, the same list the
audio models use. Every output carries a BirdNET id (`BN…`), so apps can join the
geomodel with any BirdNET audio model without name matching. Sound classes
(record type other than `species`) have no geography
and are skipped.

The taxonomy is published as `BirdNET+_Taxonomy_v0.5-Nov2026.csv` in the
[BirdNET model converter](https://github.com/birdnet-team/model-converter) repository
(stored with Git LFS). Each row has, among others, `birdnet_id`, `scientific_name`,
`common_name`, `taxon_group`, `record_type`, `gbif_id`, `ebird_code` and
`scientific_name_aliases`. A JSON export of the same taxonomy (`{"rows": [...]}`)
works too. The commands in this documentation call the file `taxonomy.csv`.

## Matching species to GBIF records

GBIF files each occurrence under the accepted species key of its name. A taxonomy
species is matched as follows:

1. If it has a `gbif_id`, the accepted species key of that name usage. Synonyms
   move to their accepted species this way.
2. Otherwise an exact match of the scientific name at species rank, then of each
   alias.

A key goes to one species only. Two taxonomy species on the same key usually mean
that GBIF still lumps a recent split. The species whose own `gbif_id` is the key
keeps it; the split gets no key. Borrowing the parent's records would give the split
the parent's whole range, which is worse than leaving it out. Species without a key
have no training data and are not in the model.

Lookups go to the public GBIF API (no account needed) and are cached in `--cache`,
so later runs only ask for new names.

## Output

`species.csv`, one row per taxonomy species:

| column | meaning |
|---|---|
| `birdnet_id` | BirdNET id |
| `scientific_name`, `common_name` | from the taxonomy |
| `taxon_group` | taxon group; cartilaginous fish are merged into `Actinopterygii` |
| `ebird_code` | eBird species code (birds), used by the benchmark |
| `gbif_key` | accepted GBIF species key, empty if none |
| `how` | how the key was found (`gbif_id`, `gbif_id, accepted key`, `name`, `alias`), or why there is none |

## Humans and domestic animals

`data/ubiquitous.csv` lists the species that are always in the model, whatever the
data say: humans (`kind = human`) and domestic animals (`kind = domestic`: dog, cat,
cattle, sheep, goat, horse, chicken, rats, honey bee). The apps use the human class
to detect speech, so it must pass everywhere; domestic animals can turn up anywhere
people live. How they are scored is described in [Scores](../model/scores.md).
