"""Stage 1: stream a GBIF occurrence download into compact per-record parquet files.

The input is the ``occurrence.txt`` table of a GBIF Darwin Core Archive, read from
stdin so the archive never has to be unpacked on disk::

    unzip -p 0001234-260101000000000.zip occurrence.txt \\
        | python -m pipeline.gbif_map data/gbif_map --workers 8

Each record keeps only what the later stages need: where (H3 cell at resolution 6;
coarser cells are its parents), when (year, month, day, model week), what (GBIF
species and class key), and who (dataset, a hash of the observer, coordinate
uncertainty, individual count). Keeping year, dataset and observer means that the
grid, the effort definition, the year window and the recency weighting can all be
changed later by re-running stage 2, without another pass over the archive.

Output: ``OUT_DIR/part_NNNNNN.parquet`` with the columns

    cell6        uint64   H3 cell, resolution 6 (~36 km²)
    year         int16    0 = unknown
    month, day   int8
    week         int8     model week 1-48 (four per month, see ``model_week``)
    species_key  int32    GBIF species key
    class_key    int32    GBIF class key (212 = Aves), 0 = unknown
    dataset_key  string   GBIF dataset key (dictionary-encoded)
    observer     uint32   CRC32 of recordedByID or recordedBy, 0 = unknown
    unc_m        uint32   coordinateUncertaintyInMeters, 0 = unknown
    n_ind        int32    individualCount, 0 = unknown

plus ``OUT_DIR/_stats/part_NNNNNN.json`` with row counts per chunk. Records without
coordinates, species, month or day, and records flagged with geospatial issues are
dropped. Chunk boundaries depend only on ``--chunk-mb``, so an interrupted run
resumes where it stopped.
"""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import os
import pathlib
import sys
import zlib
from typing import BinaryIO, Iterator

import numpy as np

COLUMNS = [
    'recordedBy', 'recordedByID', 'individualCount', 'year', 'month', 'day',
    'decimalLatitude', 'decimalLongitude', 'coordinateUncertaintyInMeters',
    'datasetKey', 'hasGeospatialIssues', 'classKey', 'speciesKey',
]
H3_RES = 6


def model_week(month: np.ndarray, day: np.ndarray) -> np.ndarray:
    """Model week 1-48: four weeks per month, days 1-7, 8-14, 15-21 and 22-31."""
    month = month.astype(np.int16)
    day = day.astype(np.int16)
    return ((month - 1) * 4 + np.minimum((day - 1) // 7, 3) + 1).astype(np.int8)


def observer_hash(names: list[str | None]) -> np.ndarray:
    """Stable 32-bit id per observer (case and surrounding space ignored); 0 = unknown."""
    return np.fromiter(
        (0 if n is None else (zlib.crc32(n.strip().lower().encode()) or 1) for n in names),
        dtype=np.uint32, count=len(names))


def process_chunk(job: tuple[int, list[str], bytes, pathlib.Path]) -> dict:
    """Parse one block of tab-separated records and write its parquet part."""
    import h3.api.basic_int as h3
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.csv as pcsv
    import pyarrow.parquet as pq

    index, header, data, out = job
    malformed = 0

    def skip_row(_row):
        nonlocal malformed
        malformed += 1
        return 'skip'

    table = pcsv.read_csv(
        io.BytesIO(data),
        read_options=pcsv.ReadOptions(column_names=header, block_size=64 << 20),
        parse_options=pcsv.ParseOptions(delimiter='\t', quote_char=False, invalid_row_handler=skip_row),
        convert_options=pcsv.ConvertOptions(
            include_columns=COLUMNS, column_types={k: pa.string() for k in COLUMNS},
            strings_can_be_null=True, null_values=['']))
    rows_in = table.num_rows

    def number(column: str, dtype: pa.DataType) -> pa.ChunkedArray:
        return pc.cast(pc.cast(table[column], pa.float64(), safe=False), dtype, safe=False)

    def clipped_uint32(column: str) -> pa.ChunkedArray:
        value = pc.fill_null(pc.cast(table[column], pa.float64(), safe=False), 0)
        return pc.cast(pc.min_element_wise(pc.max_element_wise(value, 0), 4e9), pa.uint32(), safe=False)

    try:
        lat = number('decimalLatitude', pa.float64())
        lon = number('decimalLongitude', pa.float64())
        year = number('year', pa.int16())
        month = number('month', pa.int8())
        day = number('day', pa.int8())
        species = number('speciesKey', pa.int32())
    except pa.ArrowInvalid as error:
        raise RuntimeError(f'chunk {index}: {error}') from error

    keep = pc.and_(pc.and_(pc.is_valid(lat), pc.is_valid(lon)),
                   pc.and_(pc.is_valid(species), pc.and_(pc.is_valid(month), pc.is_valid(day))))
    keep = pc.and_(keep, pc.and_(pc.and_(pc.greater_equal(month, 1), pc.less_equal(month, 12)),
                                 pc.and_(pc.greater_equal(day, 1), pc.less_equal(day, 31))))
    keep = pc.and_(keep, pc.and_(pc.less_equal(pc.abs(lat), 90), pc.less_equal(pc.abs(lon), 180)))
    keep = pc.and_(keep, pc.fill_null(pc.not_equal(table['hasGeospatialIssues'], 'true'), True))
    keep = keep.fill_null(False)

    table = table.filter(keep)
    lat, lon, year, month, day, species = (a.filter(keep) for a in (lat, lon, year, month, day, species))

    lat_np, lon_np = lat.to_numpy(), lon.to_numpy()
    cells = np.fromiter((h3.latlng_to_cell(a, b, H3_RES) for a, b in zip(lat_np, lon_np)),
                        dtype=np.uint64, count=len(lat_np))
    observers = observer_hash(pc.coalesce(table['recordedByID'], table['recordedBy']).to_pylist())

    out_table = pa.table({
        'cell6': cells,
        'year': pc.fill_null(year, 0),
        'month': month,
        'day': day,
        'week': model_week(month.to_numpy(), day.to_numpy()),
        'species_key': species,
        'class_key': pc.fill_null(number('classKey', pa.int32()), 0),
        'dataset_key': pc.dictionary_encode(table['datasetKey'].combine_chunks()),
        'observer': observers,
        'unc_m': clipped_uint32('coordinateUncertaintyInMeters'),
        'n_ind': pc.cast(pc.min_element_wise(clipped_uint32('individualCount'), 2_000_000_000), pa.int32()),
    })
    tmp = out / f'part_{index:06d}.parquet.tmp'
    pq.write_table(out_table, tmp, compression='zstd')
    os.replace(tmp, out / f'part_{index:06d}.parquet')

    stats = dict(chunk=index, rows_in=rows_in, rows_out=out_table.num_rows, malformed=malformed, bytes=len(data))
    (out / '_stats' / f'part_{index:06d}.json').write_text(json.dumps(stats))
    return stats


def read_chunks(stream: BinaryIO, size: int, out: pathlib.Path,
                resume: bool) -> Iterator[tuple[int, list[str], bytes, pathlib.Path]]:
    """Split the stream into blocks of whole lines; skip blocks that already have stats."""
    header = stream.readline().decode().rstrip('\n').split('\t')
    rest = b''
    index = 0
    while True:
        block = stream.read(size)
        if not block:
            if rest:
                yield index, header, rest, out
            return
        block = rest + block
        cut = block.rfind(b'\n') + 1
        data, rest = block[:cut], block[cut:]
        if not (resume and (out / '_stats' / f'part_{index:06d}.json').exists()):
            yield index, header, data, out
        index += 1


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('out', help='output directory for the parquet parts')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--chunk-mb', type=int, default=256, help='size of one block of input text')
    parser.add_argument('--no-resume', action='store_true', help='redo chunks that are already written')
    args = parser.parse_args(argv)

    out = pathlib.Path(args.out)
    (out / '_stats').mkdir(parents=True, exist_ok=True)
    backlog = mp.Semaphore(args.workers * 2)  # bounds RAM to about three blocks per worker

    def jobs():
        for job in read_chunks(sys.stdin.buffer, args.chunk_mb << 20, out, not args.no_resume):
            backlog.acquire()
            yield job

    total = dict(rows_in=0, rows_out=0, malformed=0, bytes=0, chunks=0)
    # No maxtasksperchild: recycling workers deadlocked the pool on long runs.
    with mp.Pool(args.workers) as pool:
        for stats in pool.imap_unordered(process_chunk, jobs()):
            backlog.release()
            total['chunks'] += 1
            for key in ('rows_in', 'rows_out', 'malformed', 'bytes'):
                total[key] += stats[key]
            if total['chunks'] % 20 == 0:
                print(json.dumps(total), flush=True)
    print('done', json.dumps(total), flush=True)


if __name__ == '__main__':
    main()
