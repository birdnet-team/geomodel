import pandas as pd
import logging
from pathlib import Path

def generate_taxonomy(metadata_path: str, output_path: str):
    logging.info(f"Generating legacy-style taxonomy from {metadata_path}...")

    # Load metadata
    df = pd.read_csv(metadata_path)

    # 1. Authority logic for species_code: eBird code for birds, iNat ID for others
    def get_species_code(row):
        is_bird = str(row.get('taxon_group', '')).strip().lower() == 'aves'
        ebird = str(row.get('ebird_code', '')).strip()
        inat = row.get('inat_id')

        if is_bird and ebird and ebird.lower() != 'nan' and ebird != '':
            return str(ebird)

        # Handle iNat ID: convert to int then string to avoid .0 suffix
        if pd.notna(inat) and str(inat).lower() != 'nan' and inat != '':
            try:
                return str(int(float(inat)))
            except (ValueError, TypeError):
                return str(inat).strip()

        return ''

    def clean_name_casing(name):
        """Ensure names are not forced to lowercase."""
        if not name or not isinstance(name, str) or name.lower() == 'nan':
            return ""
        return name.strip()

    # 2. Locale columns to include (all available from metadata)
    metadata_cols = df.columns.tolist()
    _non_locale = {'common_name_en', 'common_name_alt', 'common_name_aliases'}
    locales = [c for c in metadata_cols if c.startswith('common_name_') and c not in _non_locale]

    # 3. Process records
    processed_data = []

    # Sort by scientific name for consistent indexing
    df = df.sort_values(by='scientific_name')

    for _, row in df.iterrows():
        sci_name = clean_name_casing(row.get('scientific_name', ''))
        if not sci_name:
            continue

        # Determine base English common name for 'com_name'
        com_en = clean_name_casing(row.get('common_name_en', ''))
        if not com_en:
            # Fallback to the general 'common_name' field, then to scientific name
            com_en = clean_name_casing(row.get('common_name', ''))
            if not com_en:
                com_en = sci_name

        entry = {
            'sci_name': sci_name,
            'com_name': com_en,
            'species_code': get_species_code(row),
            'class_name': str(row.get('taxon_group', '')).strip().lower()
        }

        # Add other locales with fallbacks to the primary common name (English)
        for loc in locales:
            val = clean_name_casing(row.get(loc, ''))
            # If specific locale is missing, use English common name (preserve case)
            if not val:
                entry[loc] = com_en
            else:
                entry[loc] = val

        processed_data.append(entry)

    # Final DataFrame
    tax_df = pd.DataFrame(processed_data)

    # Add incremental index
    tax_df.insert(0, 'idx', range(len(tax_df)))

    # Save to CSV
    tax_df.to_csv(output_path, index=False)
    logging.info(f"New taxonomy.csv generated with {len(tax_df)} species at {output_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_taxonomy("species-data/species_metadata.csv", "taxonomy_v0.2-Jun2026.csv")
