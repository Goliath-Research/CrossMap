"""BAM/SAM liftover must not store a read only as a secondary alignment.

A read whose alignment overlaps more than one chain block, and a paired
read whose mate does, is still one read. These tests run the ``bam``
command and then apply the filter variant callers use by default: skip
secondary (flag 0x100), unmapped (0x4), QC-fail (0x200), and duplicate
(0x400) records. Each affected read must still yield a mapped record that
carries its sequence.

They fail while that record is written with the secondary flag and no
primary record is written beside it.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import pysam

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cmmodule import crossmap  # noqa: E402

# Source chr1 is 1000 bp. One chain leaves a gap between two blocks:
#   source [100, 150) -> target chr1 [500, 550)
#   source [160, 210) -> target chr1 [555, 605)
# A read covering source [140, 180) overlaps both blocks.
GAP_CHAIN = """\
chain 1000 chr1 1000 + 100 210 chr1 1000 + 500 605 1
50 10 5
50
"""

# Two chains cover the same source interval, so a read there lifts to two
# target places. A third chain is a unique block used as a control mate.
#   source [100, 150) -> target chr1 [500, 550)
#   source [100, 150) -> target chr2 [10, 60)
#   source [400, 450) -> target chr1 [700, 750)
OVERLAP_CHAIN = """\
chain 1000 chr1 1000 + 100 150 chr1 1000 + 500 550 1
50
chain 900 chr1 1000 + 100 150 chr2 800 + 10 60 2
50
chain 800 chr1 1000 + 400 450 chr1 1000 + 700 750 3
50
"""

# Flags excluded by common variant callers and viewers.
_SKIP = 0x100 | 0x4 | 0x200 | 0x400


def _qualities(sequence):
    return pysam.qualitystring_to_array("I" * len(sequence))


def _write_alignment(out, name, sequence, flag, start=None, cigar=None,
                     mate_start=None, unmapped=False, mate_unmapped=False):
    record = pysam.AlignedSegment()
    record.query_name = name
    record.query_sequence = sequence
    record.flag = flag
    record.query_qualities = _qualities(sequence)
    if unmapped:
        record.reference_id = -1
        record.reference_start = -1
        record.mapping_quality = 0
    else:
        record.reference_id = 0
        record.reference_start = start
        record.mapping_quality = 40
        record.cigarstring = cigar
    if mate_unmapped or mate_start is None:
        record.next_reference_id = -1
        record.next_reference_start = -1
    else:
        record.next_reference_id = 0
        record.next_reference_start = mate_start
    out.write(record)


def _header():
    return {
        "HD": {"VN": "1.6", "SO": "unsorted"},
        "SQ": [{"SN": "chr1", "LN": 1000}, {"SN": "chr2", "LN": 800}],
    }


def _write_file(path, records):
    mode = "w" if path.endswith(".sam") else "wb"
    with pysam.AlignmentFile(path, mode, header=_header()) as out:
        for record in records:
            _write_alignment(out, **record)


def _run_bam(chain_text, alignment_path):
    work = tempfile.mkdtemp(prefix="crossmap-bam-")
    chain_path = os.path.join(work, "lift.chain")
    with open(chain_path, "w", encoding="ascii") as handle:
        handle.write(chain_text)
    prefix = os.path.join(work, "lifted")
    argv = ["CrossMap", "bam", chain_path, alignment_path, prefix]
    previous = sys.argv
    sys.argv = argv
    try:
        crossmap()
    finally:
        sys.argv = previous
    suffix = ".sam" if alignment_path.endswith(".sam") else ".bam"
    output = prefix + suffix
    if not os.path.exists(output):
        raise AssertionError("bam command wrote no alignment file at %s" % output)
    with pysam.AlignmentFile(output, "r") as aligned:
        return list(aligned)


def _kept(records, name, read1=None):
    """Records a secondary-excluding caller would still use."""
    kept = []
    for record in records:
        if record.query_name != name:
            continue
        if record.flag & _SKIP:
            continue
        if read1 is True and not record.is_read1:
            continue
        if read1 is False and not record.is_read2:
            continue
        kept.append(record)
    return kept


def _assert_kept(records, name, sequence, read1=None, contigs=None):
    kept = _kept(records, name, read1=read1)
    end = "read1" if read1 is True else "read2" if read1 is False else "read"
    if not kept:
        present = [
            (rec.flag, rec.is_unmapped, rec.reference_name, rec.reference_start)
            for rec in records
            if rec.query_name == name and (read1 is None or rec.is_read1 == read1)
        ]
        raise AssertionError(
            "%s %s is dropped when secondary alignments are excluded; "
            "written records (flag, unmapped, contig, start): %s"
            % (name, end, present)
        )
    sequences = {rec.query_sequence for rec in kept}
    if sequence not in sequences:
        raise AssertionError(
            "%s %s kept records do not carry the original sequence %s; got %s"
            % (name, end, sequence, sequences)
        )
    if contigs is not None:
        seen = {rec.reference_name for rec in kept}
        if not seen & set(contigs):
            raise AssertionError(
                "%s %s kept records are on %s, expected one of %s"
                % (name, end, seen, contigs)
            )


class BamMultiblockSecondaryTests(unittest.TestCase):
    def test_single_end_gap_read_survives_secondary_filter(self):
        sequence = "ACGT" * 10
        path = os.path.join(tempfile.mkdtemp(), "in.bam")
        _write_file(path, [{
            "name": "span_se",
            "sequence": sequence,
            "flag": 0,
            "start": 140,
            "cigar": "40M",
        }])
        lifted = _run_bam(GAP_CHAIN, path)
        _assert_kept(lifted, "span_se", sequence, contigs=("chr1",))

    def test_reverse_single_end_gap_read_survives_secondary_filter(self):
        sequence = "TGCA" * 10
        path = os.path.join(tempfile.mkdtemp(), "in.bam")
        _write_file(path, [{
            "name": "span_rev",
            "sequence": sequence,
            "flag": 0x10,
            "start": 140,
            "cigar": "40M",
        }])
        lifted = _run_bam(GAP_CHAIN, path)
        _assert_kept(lifted, "span_rev", sequence, contigs=("chr1",))

    def test_sam_single_end_gap_read_survives_secondary_filter(self):
        sequence = "ACGT" * 10
        path = os.path.join(tempfile.mkdtemp(), "in.sam")
        _write_file(path, [{
            "name": "span_sam",
            "sequence": sequence,
            "flag": 0,
            "start": 140,
            "cigar": "40M",
        }])
        lifted = _run_bam(GAP_CHAIN, path)
        _assert_kept(lifted, "span_sam", sequence, contigs=("chr1",))

    def test_unique_single_end_read_stays_primary_at_lifted_position(self):
        sequence = "T" * 20
        path = os.path.join(tempfile.mkdtemp(), "in.bam")
        _write_file(path, [{
            "name": "inside_se",
            "sequence": sequence,
            "flag": 0,
            "start": 110,
            "cigar": "20M",
        }])
        lifted = _run_bam(GAP_CHAIN, path)
        _assert_kept(lifted, "inside_se", sequence, contigs=("chr1",))
        kept = _kept(lifted, "inside_se")
        self.assertEqual(
            [(rec.reference_name, rec.reference_start) for rec in kept],
            [("chr1", 510)],
        )

    def test_overlapping_chains_single_end_survives_secondary_filter(self):
        sequence = "AC" * 20
        path = os.path.join(tempfile.mkdtemp(), "in.bam")
        _write_file(path, [{
            "name": "overlap_se",
            "sequence": sequence,
            "flag": 0,
            "start": 100,
            "cigar": "40M",
        }])
        lifted = _run_bam(OVERLAP_CHAIN, path)
        _assert_kept(lifted, "overlap_se", sequence, contigs=("chr1", "chr2"))

    def test_paired_read_spanning_gap_survives_secondary_filter(self):
        # Read 2 overlaps both gap blocks. Read 1 sits inside the first
        # block, so its own liftover is unique. Both ends must remain
        # visible after secondary alignments are excluded.
        left = "G" * 20
        right = "C" * 40
        path = os.path.join(tempfile.mkdtemp(), "in.bam")
        _write_file(path, [
            {
                "name": "pair_gap",
                "sequence": left,
                "flag": 0x1 | 0x40,
                "start": 110,
                "cigar": "20M",
                "mate_start": 140,
            },
            {
                "name": "pair_gap",
                "sequence": right,
                "flag": 0x1 | 0x80,
                "start": 140,
                "cigar": "40M",
                "mate_start": 110,
            },
        ])
        lifted = _run_bam(GAP_CHAIN, path)
        _assert_kept(lifted, "pair_gap", left, read1=True, contigs=("chr1",))
        _assert_kept(lifted, "pair_gap", right, read1=False, contigs=("chr1",))
        kept_left = _kept(lifted, "pair_gap", read1=True)
        self.assertEqual(kept_left[0].reference_start, 510)

    def test_paired_gap_read_with_unmapped_mate_survives_secondary_filter(self):
        mate = "A" * 20
        spanning = "ACGT" * 10
        path = os.path.join(tempfile.mkdtemp(), "in.bam")
        _write_file(path, [
            {
                "name": "pair_mn",
                "sequence": mate,
                "flag": 0x1 | 0x4 | 0x40,
                "unmapped": True,
                "mate_start": 140,
            },
            {
                "name": "pair_mn",
                "sequence": spanning,
                "flag": 0x1 | 0x8 | 0x80,
                "start": 140,
                "cigar": "40M",
                "mate_unmapped": True,
            },
        ])
        lifted = _run_bam(GAP_CHAIN, path)
        _assert_kept(lifted, "pair_mn", spanning, read1=False, contigs=("chr1",))

    def test_unique_read_not_secondary_when_mate_hits_two_chains(self):
        # Read 1 lifts through a single chain block. Read 2 starts in a
        # region covered by two chains. Read 1 must not be stored only as
        # a secondary alignment because of its mate.
        unique = "G" * 30
        other = "C" * 40
        path = os.path.join(tempfile.mkdtemp(), "in.bam")
        _write_file(path, [
            {
                "name": "pair_um",
                "sequence": unique,
                "flag": 0x1 | 0x40,
                "start": 400,
                "cigar": "30M",
                "mate_start": 100,
            },
            {
                "name": "pair_um",
                "sequence": other,
                "flag": 0x1 | 0x80,
                "start": 100,
                "cigar": "40M",
                "mate_start": 400,
            },
        ])
        lifted = _run_bam(OVERLAP_CHAIN, path)
        _assert_kept(lifted, "pair_um", unique, read1=True, contigs=("chr1",))
        _assert_kept(lifted, "pair_um", other, read1=False, contigs=("chr1", "chr2"))
        kept = _kept(lifted, "pair_um", read1=True)
        self.assertEqual(
            (kept[0].reference_name, kept[0].reference_start),
            ("chr1", 700),
        )

    def test_both_ends_on_overlapping_chains_each_survive_secondary_filter(self):
        left = "A" * 30
        right = "T" * 30
        path = os.path.join(tempfile.mkdtemp(), "in.bam")
        _write_file(path, [
            {
                "name": "pair_mm",
                "sequence": left,
                "flag": 0x1 | 0x40,
                "start": 100,
                "cigar": "30M",
                "mate_start": 110,
            },
            {
                "name": "pair_mm",
                "sequence": right,
                "flag": 0x1 | 0x80,
                "start": 110,
                "cigar": "30M",
                "mate_start": 100,
            },
        ])
        lifted = _run_bam(OVERLAP_CHAIN, path)
        _assert_kept(lifted, "pair_mm", left, read1=True, contigs=("chr1", "chr2"))
        _assert_kept(lifted, "pair_mm", right, read1=False, contigs=("chr1", "chr2"))

    def test_unmapped_read_not_marked_secondary_for_multiblock_mate(self):
        # The unmapped end does not itself overlap a chain. Its only record
        # must not carry the secondary flag just because the mate start
        # overlaps two chains. Callers that keep unmapped reads and skip
        # secondary alignments would otherwise drop it.
        unmapped = "A" * 20
        mapped = "C" * 40
        path = os.path.join(tempfile.mkdtemp(), "in.bam")
        _write_file(path, [
            {
                "name": "pair_nm",
                "sequence": unmapped,
                "flag": 0x1 | 0x4 | 0x40,
                "unmapped": True,
                "mate_start": 100,
            },
            {
                "name": "pair_nm",
                "sequence": mapped,
                "flag": 0x1 | 0x8 | 0x80,
                "start": 100,
                "cigar": "40M",
                "mate_unmapped": True,
            },
        ])
        lifted = _run_bam(OVERLAP_CHAIN, path)
        read1 = [
            rec for rec in lifted
            if rec.query_name == "pair_nm" and rec.is_read1
        ]
        self.assertTrue(read1, "unmapped read was not written")
        self.assertTrue(
            all(rec.flag & 0x100 == 0 for rec in read1),
            "unmapped read is stored only as a secondary alignment: %s"
            % [(rec.flag,) for rec in read1],
        )
        _assert_kept(lifted, "pair_nm", mapped, read1=False, contigs=("chr1", "chr2"))


if __name__ == "__main__":
    unittest.main()
