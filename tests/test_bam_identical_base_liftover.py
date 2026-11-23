"""BAM liftover must keep going when a read sequence contains '='.

SAM encodes the identical-to-reference base as '='. pysam preserves that
base in query_sequence. When liftover changes the strand, CrossMap
reverse-complements the sequence. The complement table has no entry for
'=', and three call sites do not catch the lookup error:

- paired-end, uniquely lifted read (mapbam.crossmap_bam_file)
- paired-end, multiply lifted read
- single-end, multiply lifted read

The single-end unique path already catches the lookup error. These tests
are the spec for a fix: the bam command exits successfully and every
alignment, including alignments after the read that contains '=', is
written. They fail while the uncaught lookup is still present.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pysam


ROOT = Path(__file__).resolve().parents[1]
CHROM = "chr1"
CHROM_LEN = 1000
# Reverse complement of ACGTAC, the sequence on the read that follows '='.
AFTER_SEQUENCE = "ACGTAC"
AFTER_REVCOMP = "GTACGT"


def _minus_start(start, length):
    """Target start for a plus-strand interval lifted by the minus chain."""
    return CHROM_LEN - start - length


def _header():
    return pysam.AlignmentHeader.from_dict(
        {
            "HD": {"VN": "1.6", "SO": "unsorted"},
            "SQ": [{"SN": CHROM, "LN": CHROM_LEN}],
        }
    )


def _alignment(header, name, sequence, start, paired):
    record = pysam.AlignedSegment(header)
    record.query_name = name
    record.query_sequence = sequence
    # Paired read1 with an unmapped mate still takes the paired-end branch.
    record.flag = 0x1 | 0x40 | 0x8 if paired else 0
    record.reference_id = 0
    record.reference_start = start
    record.mapping_quality = 40
    record.cigartuples = [(0, len(sequence))]
    record.next_reference_id = -1
    record.next_reference_start = -1
    record.template_length = 0
    record.query_qualities = pysam.qualitystring_to_array("I" * len(sequence))
    return record


def _write_reads(path, mode, reads, paired, reference_filename=None):
    header = _header()
    kwargs = {"header": header}
    if reference_filename is not None:
        kwargs["reference_filename"] = reference_filename
    with pysam.AlignmentFile(path, mode, **kwargs) as handle:
        for name, sequence, start in reads:
            handle.write(_alignment(header, name, sequence, start, paired))


def _reads():
    return [
        ("before", "AACGTT", 10),
        ("bad", "AC=GT", 100),
        ("after", AFTER_SEQUENCE, 200),
    ]


class IdenticalBaseLiftoverTests(unittest.TestCase):
    def _chain(self, strand, multiple):
        blocks = [
            "chain 1000 chr1 1000 + 0 1000 chr1 1000 %s 0 1000 1\n1000\n"
            % strand
        ]
        if multiple:
            blocks.append(
                "chain 900 chr1 1000 + 0 1000 chr2 1000 %s 0 1000 2\n1000\n"
                % strand
            )
        return "".join(blocks)

    def _run_bam(self, directory, chain_text, kind, paired, multiple):
        chain_path = Path(directory) / "flip.chain"
        chain_path.write_text(chain_text)
        reads = _reads()
        reference = None
        if kind == "bam":
            infile = Path(directory) / "in.bam"
            _write_reads(str(infile), "wb", reads, paired)
            out_path = Path(directory) / "lifted.bam"
        elif kind == "sam":
            infile = Path(directory) / "in.sam"
            _write_reads(str(infile), "wh", reads, paired)
            out_path = Path(directory) / "lifted.sam"
        elif kind == "cram":
            reference = str(Path(directory) / "ref.fa")
            Path(reference).write_text(">%s\n%s\n" % (CHROM, "A" * CHROM_LEN))
            pysam.faidx(reference)
            infile = Path(directory) / "in.cram"
            _write_reads(
                str(infile), "wc", reads, paired, reference_filename=reference
            )
            # CRAM input is written out as BAM.
            out_path = Path(directory) / "lifted.bam"
        else:
            raise AssertionError(kind)

        prefix = str(Path(directory) / "lifted")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get(
            "PYTHONPATH", ""
        )
        # A local REF_PATH keeps htslib from using its default CRAM reference URL.
        env["REF_PATH"] = directory
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "from cmmodule import crossmap; crossmap()",
                "bam",
                str(chain_path),
                str(infile),
                prefix,
            ],
            cwd=directory,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        records = []
        if out_path.exists() and out_path.stat().st_size > 0:
            read_mode = "r" if kind == "sam" else "rb"
            open_kwargs = {}
            if kind == "cram":
                open_kwargs["reference_filename"] = reference
            try:
                with pysam.AlignmentFile(str(out_path), read_mode, **open_kwargs) as handle:
                    records = list(handle)
            except (ValueError, OSError):
                records = []
        return proc, records

    def _assert_continues(self, proc, records, strand_flipped, multiple, paired):
        names = [record.query_name for record in records]
        self.assertEqual(
            proc.returncode,
            0,
            "bam command stopped before later alignments were written.\n"
            "names=%s\nstdout:\n%s\nstderr:\n%s"
            % (names, proc.stdout, proc.stderr),
        )
        self.assertEqual(names, ["before", "bad", "after"])
        before = records[0]
        after = records[2]
        self.assertEqual(before.reference_name, CHROM)
        self.assertEqual(after.reference_name, CHROM)
        self.assertEqual(before.is_paired, paired)
        self.assertEqual(after.is_paired, paired)
        if strand_flipped:
            self.assertEqual(before.reference_start, _minus_start(10, 6))
            self.assertEqual(after.reference_start, _minus_start(200, 6))
            self.assertTrue(before.is_reverse)
            self.assertTrue(after.is_reverse)
            self.assertEqual(after.query_sequence, AFTER_REVCOMP)
        else:
            self.assertEqual(before.reference_start, 10)
            self.assertEqual(after.reference_start, 200)
            self.assertFalse(before.is_reverse)
            self.assertFalse(after.is_reverse)
            self.assertEqual(after.query_sequence, AFTER_SEQUENCE)
            self.assertEqual(records[1].query_sequence, "AC=GT")
        if multiple:
            self.assertTrue(before.is_secondary)
            self.assertTrue(after.is_secondary)
        else:
            self.assertFalse(before.is_secondary)
            self.assertFalse(after.is_secondary)

    def test_paired_unique_bam_writes_alignments_after_equal_base(self):
        with tempfile.TemporaryDirectory() as directory:
            proc, records = self._run_bam(
                directory,
                self._chain("-", multiple=False),
                "bam",
                paired=True,
                multiple=False,
            )
            self._assert_continues(
                proc, records, strand_flipped=True, multiple=False, paired=True
            )

    def test_paired_multiple_bam_writes_alignments_after_equal_base(self):
        with tempfile.TemporaryDirectory() as directory:
            proc, records = self._run_bam(
                directory,
                self._chain("-", multiple=True),
                "bam",
                paired=True,
                multiple=True,
            )
            self._assert_continues(
                proc, records, strand_flipped=True, multiple=True, paired=True
            )

    def test_single_multiple_bam_writes_alignments_after_equal_base(self):
        with tempfile.TemporaryDirectory() as directory:
            proc, records = self._run_bam(
                directory,
                self._chain("-", multiple=True),
                "bam",
                paired=False,
                multiple=True,
            )
            self._assert_continues(
                proc, records, strand_flipped=True, multiple=True, paired=False
            )

    def test_paired_unique_sam_writes_alignments_after_equal_base(self):
        with tempfile.TemporaryDirectory() as directory:
            proc, records = self._run_bam(
                directory,
                self._chain("-", multiple=False),
                "sam",
                paired=True,
                multiple=False,
            )
            self._assert_continues(
                proc, records, strand_flipped=True, multiple=False, paired=True
            )

    def test_paired_unique_cram_writes_alignments_after_equal_base(self):
        with tempfile.TemporaryDirectory() as directory:
            proc, records = self._run_bam(
                directory,
                self._chain("-", multiple=False),
                "cram",
                paired=True,
                multiple=False,
            )
            self._assert_continues(
                proc, records, strand_flipped=True, multiple=False, paired=True
            )

    def test_paired_same_strand_keeps_equal_base_and_later_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            proc, records = self._run_bam(
                directory,
                self._chain("+", multiple=False),
                "bam",
                paired=True,
                multiple=False,
            )
            self._assert_continues(
                proc, records, strand_flipped=False, multiple=False, paired=True
            )

    def test_single_unique_strand_flip_keeps_equal_base_and_later_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            proc, records = self._run_bam(
                directory,
                self._chain("-", multiple=False),
                "bam",
                paired=False,
                multiple=False,
            )
            self._assert_continues(
                proc, records, strand_flipped=True, multiple=False, paired=False
            )


if __name__ == "__main__":
    unittest.main()
