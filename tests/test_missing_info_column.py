"""Liftover must keep going when a variant has no INFO column.

A gVCF or VCF record needs eight columns before the converter rewrites INFO.
These tests describe that behavior:

- A record that already has INFO, including a sample column, is lifted.
- A record with only five, six, or seven columns is copied to the unmapped
  file and does not abort the command.
- Records that follow the short line are still written to the output.
- The same rule applies to a gVCF reference block (a line containing END=)
  that has no INFO column, and to both commands when reference-allele
  checking is turned on and the reference base matches.

They fail while the converter indexes a missing column and stops.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
# Source chr1 [0, 40) lifts onto target chr1 [100, 140).
# A 1-based source position P therefore becomes P + 100.
CHAIN = "chain 100 chr1 40 + 0 40 chr1 200 + 100 140 1\n40\n"
FASTA_SEQUENCE = "ACGT" * 50  # 200 bp, indexed as A, C, G, T, ...


def lifted_position(source_position):
    return source_position + 100


def target_base(source_position):
    """Base the target FASTA holds at the lifted coordinate of this position."""
    target_start = 100 + (source_position - 1)
    return FASTA_SEQUENCE[target_start]


class MissingInfoColumnTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="crossmap-info-")
        root = Path(self.tmp.name)
        self.chain = root / "tiny.chain"
        self.fasta = root / "target.fa"
        self.chain.write_text(CHAIN)
        self.fasta.write_text(">chr1\n" + FASTA_SEQUENCE + "\n")
        self.case_number = 0

    def tearDown(self):
        self.tmp.cleanup()

    def test_complete_records_are_lifted(self):
        before = "chr1\t5\tbefore_ok\tA\tT\t.\tPASS\t."
        after = "chr1\t25\tafter_ok\tA\tG\t.\tPASS\t.\tGT\t0/1"
        for command in ("gvcf", "vcf"):
            with self.subTest(command=command):
                result, output, _unmap = self.run_liftover(
                    command, [before, after])
                self.assert_completed(result)
                self.assert_lifted(output, before, after)

    def test_complete_gvcf_block_is_lifted(self):
        # Only the gvcf command rewrites a reference block. The vcf command
        # lifts the same line as an ordinary variant.
        block = "chr1\t8\tblock_ok\tA\t<NON_REF>\t.\tPASS\tEND=12"
        after = "chr1\t25\tafter_ok\tA\tG\t.\tPASS\t.\tGT\t0/1"
        result, output, _unmap = self.run_liftover("gvcf", [block, after])
        self.assert_completed(result)
        rows = data_rows(output)
        self.assertIn(
            "chr1\t108\tblock_ok\tA\t<NON_REF>\t.\tPASS\tEND=112",
            rows)
        self.assertIn(self.lifted_row(after), rows)

    def test_short_variant_is_unmapped_and_later_record_is_kept(self):
        # Five, six, and seven columns all stop before INFO.
        short_lines = {
            5: "chr1\t10\tmissing_info_5\tA\tC",
            6: "chr1\t15\tmissing_info_6\tA\tC\t30",
            7: "chr1\t20\tmissing_info_7\tA\tC\t30\tPASS",
        }
        before = "chr1\t5\tbefore_ok\tA\tT\t.\tPASS\t."
        after = "chr1\t25\tafter_ok\tA\tG\t.\tPASS\t.\tGT\t0/1"
        for command in ("gvcf", "vcf"):
            for width, short in short_lines.items():
                with self.subTest(command=command, columns=width):
                    self.assert_short_line_skipped(
                        command, [before, short, after], short, after)

    def test_two_short_variants_in_one_file_both_skip(self):
        before = "chr1\t5\tbefore_ok\tA\tT\t.\tPASS\t."
        first = "chr1\t10\tmissing_info_5\tA\tC"
        second = "chr1\t20\tmissing_info_7\tA\tC\t30\tPASS"
        after = "chr1\t25\tafter_ok\tA\tG\t.\tPASS\t.\tGT\t0/1"
        for command in ("gvcf", "vcf"):
            with self.subTest(command=command):
                result, output, unmap = self.run_liftover(
                    command, [before, first, second, after])
                self.assert_completed(result)
                self.assert_lifted(output, before, after)
                self.assert_unmapped(unmap, first)
                self.assert_unmapped(unmap, second)
                self.assert_not_mapped(output, "missing_info_5")
                self.assert_not_mapped(output, "missing_info_7")

    def test_short_gvcf_block_is_unmapped_and_later_record_is_kept(self):
        # END= selects the reference-block path, which also rewrites column 8.
        before = "chr1\t5\tbefore_ok\tA\tT\t.\tPASS\t."
        after = "chr1\t25\tafter_ok\tA\tG\t.\tPASS\t.\tGT\t0/1"
        short_blocks = {
            6: "chr1\t8\tshort_block_6\tA\t<NON_REF>\tEND=12",
            7: "chr1\t8\tshort_block_7\tA\t<NON_REF>\t.\tEND=12",
        }
        for width, short in short_blocks.items():
            with self.subTest(columns=width):
                self.assert_short_line_skipped(
                    "gvcf", [before, short, after], short, after)

    def test_ref_consistent_short_variant_still_skips(self):
        # Target base at source position 10 is C, so the consistency check
        # matches and execution continues into the INFO rewrite.
        self.assertEqual(target_base(10), "C")
        before = "chr1\t5\tbefore_ok\tA\tT\t.\tPASS\t."
        short = "chr1\t10\tmissing_info_5\tC\tA"
        after = "chr1\t25\tafter_ok\tA\tG\t.\tPASS\t.\tGT\t0/1"
        for command in ("gvcf", "vcf"):
            with self.subTest(command=command):
                self.assert_short_line_skipped(
                    command,
                    [before, short, after],
                    short,
                    after,
                    extra_args=("--ref-consistent",))

    def test_short_line_on_unmapped_chromosome_keeps_later_record(self):
        before = "chr1\t5\tbefore_ok\tA\tT\t.\tPASS\t."
        short = "chr9\t10\tmissing_info_5\tA\tC"
        after = "chr1\t25\tafter_ok\tA\tG\t.\tPASS\t.\tGT\t0/1"
        for command in ("gvcf", "vcf"):
            with self.subTest(command=command):
                self.assert_short_line_skipped(
                    command, [before, short, after], short, after)

    def assert_short_line_skipped(self, command, body, short, after,
                                  extra_args=()):
        result, output, unmap = self.run_liftover(
            command, body, extra_args=extra_args)
        self.assert_completed(result)
        self.assert_lifted(output, body[0], after)
        self.assert_unmapped(unmap, short)
        self.assert_not_mapped(output, short.split("\t")[2])

    def assert_not_mapped(self, output, record_id):
        mapped_ids = [row.split("\t")[2] for row in data_rows(output)]
        self.assertNotIn(record_id, mapped_ids)

    def assert_completed(self, result):
        self.assertEqual(
            result.returncode,
            0,
            msg="liftover exited %s\n%s\n%s" % (
                result.returncode, result.stdout, result.stderr),
        )
        self.assertNotIn("Traceback", result.stderr)
        self.assertNotIn("Traceback", result.stdout)

    def assert_lifted(self, output, before, after):
        rows = data_rows(output)
        self.assertIn(self.lifted_row(before), rows)
        self.assertIn(self.lifted_row(after), rows)

    def lifted_row(self, line):
        fields = line.split("\t")
        fields[1] = str(lifted_position(int(fields[1])))
        return "\t".join(fields)

    def assert_unmapped(self, unmap_text, original_line):
        rows = data_rows(unmap_text)
        self.assertTrue(
            any(
                row == original_line or row.startswith(original_line + "\t")
                for row in rows
            ),
            msg="short line was not written to the unmapped file:\n%s\n%s"
            % (original_line, unmap_text),
        )

    def run_liftover(self, command, body, extra_args=()):
        self.case_number += 1
        root = Path(self.tmp.name)
        infile = root / ("%s-%d.%s" % (command, self.case_number, command))
        outfile = root / ("%s-%d.out" % (command, self.case_number))
        infile.write_text(variant_file(body))
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(REPO / "src"), env.get("PYTHONPATH", "")])
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from cmmodule import crossmap; crossmap()",
                command,
                *extra_args,
                str(self.chain),
                str(infile),
                str(self.fasta),
                str(outfile),
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        output = outfile.read_text() if outfile.exists() else ""
        unmap_path = Path(str(outfile) + ".unmap")
        unmap = unmap_path.read_text() if unmap_path.exists() else ""
        return result, output, unmap


def variant_file(body):
    # A contig header is required for the gVCF header rewrite when the
    # target FASTA uses a chr-prefixed name. Without it the command stops
    # before any variant is read.
    header = [
        "##fileformat=VCFv4.2",
        "##contig=<ID=chr1,length=40>",
        "##INFO=<ID=END,Number=1,Type=Integer,Description=\"Stop position of the interval\">",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
    ]
    return "\n".join(header + list(body)) + "\n"


def data_rows(text):
    return [
        line for line in text.splitlines()
        if line and not line.startswith("#")
    ]


if __name__ == "__main__":
    unittest.main()
