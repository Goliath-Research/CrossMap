"""VCF and gVCF liftover must keep going when a variant has no INFO column.

A record with CHROM, POS, ID, REF, and ALT (and optionally QUAL or FILTER)
but no eighth INFO column is still a line the vcf and gvcf commands read.
After a successful coordinate lift, the converters rewrite INFO in column 8.
These tests run that command path. They fail while a short mappable line
raises and leaves every following variant out of the output, and they pass
once that line is written to the unmapped file and later variants are lifted.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

CHAIN = "\n".join(
    [
        "chain 100 chr1 200 + 0 200 chr1 200 + 0 200 1",
        "200",
        "",
    ]
)
FASTA = ">chr1\n" + ("A" * 200) + "\n"
VCF_HEADER = "\n".join(
    [
        "##fileformat=VCFv4.2",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
        "",
    ]
)
GVCF_HEADER = "\n".join(
    [
        "##fileformat=VCFv4.2",
        "##contig=<ID=chr1,length=200>",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
        "",
    ]
)


def run_crossmap(args):
    """Run the installed entry point against this tree, not a stub."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "from cmmodule import crossmap; crossmap()",
            *args,
        ],
        env=env,
        capture_output=True,
        text=True,
    )


def write_assembly(directory):
    chain = directory / "mini.chain"
    fasta = directory / "target.fa"
    chain.write_text(CHAIN)
    fasta.write_text(FASTA)
    return chain, fasta


def variant(chrom, pos, name, alt="G", extra=()):
    columns = [chrom, str(pos), name, "A", alt, *extra]
    return "\t".join(columns)


def data_lines(text, name):
    return [
        line
        for line in text.splitlines()
        if f"\t{name}\t" in f"\t{line}\t"
    ]


class MissingInfoColumnTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.chain, self.fasta = write_assembly(self.directory)

    def lift(self, command, body, outfile, extra_args=()):
        infile = self.directory / f"input.{command}"
        header = VCF_HEADER if command == "vcf" else GVCF_HEADER
        infile.write_text(header + body)
        out = self.directory / outfile
        result = run_crossmap(
            [
                command,
                str(self.chain),
                str(infile),
                str(self.fasta),
                *extra_args,
                str(out),
            ]
        )
        mapped = out.read_text() if out.exists() else ""
        unmap_path = Path(str(out) + ".unmap")
        unmapped = unmap_path.read_text() if unmap_path.exists() else ""
        return result, mapped, unmapped

    def assert_lift_continues(self, command, body, short_id, extra_args=()):
        result, mapped, unmapped = self.lift(
            command, body, f"{command}-{short_id}.out", extra_args
        )
        detail = (
            f"returncode={result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}\n"
            f"mapped:\n{mapped}\n"
            f"unmapped:\n{unmapped}"
        )
        # The record after the short line has to be in the mapped file.
        # A traceback that stops the loop leaves this record out.
        self.assertNotEqual(mapped, "", detail)
        self.assertTrue(data_lines(mapped, "AFTER"), detail)
        after = data_lines(mapped, "AFTER")[0].split("\t")
        self.assertGreaterEqual(len(after), 8, detail)
        self.assertEqual(result.returncode, 0, detail)
        self.assertNotIn("Traceback", result.stderr, detail)
        self.assertNotIn("IndexError", result.stderr, detail)
        self.assertTrue(data_lines(mapped, "BEFORE"), detail)
        # The short record is not liftable as written. It goes to the
        # unmapped file, and the mapped file does not stop on it.
        self.assertTrue(data_lines(unmapped, short_id), detail)
        self.assertFalse(data_lines(mapped, short_id), detail)

    def test_eight_column_vcf_lifts_every_variant(self):
        body = "\n".join(
            [
                variant("chr1", 10, "BEFORE", "G", (".", ".", "END=99")),
                variant("chr1", 20, "MID", "C", (".", ".", "DP=2")),
                variant("chr1", 30, "AFTER", "T", (".", "PASS", ".")),
                "",
            ]
        )
        result, mapped, unmapped = self.lift("vcf", body, "complete.vcf")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        for name in ("BEFORE", "MID", "AFTER"):
            lines = data_lines(mapped, name)
            self.assertEqual(len(lines), 1, mapped)
            self.assertGreaterEqual(len(lines[0].split("\t")), 8)
        # END= lives in the INFO column and is rewritten from the lifted end.
        before = data_lines(mapped, "BEFORE")[0]
        self.assertIn("END=10", before)
        self.assertNotIn("END=99", before)
        self.assertEqual(
            [line for line in unmapped.splitlines() if not line.startswith("#")],
            [],
        )

    def test_vcf_sample_columns_stay_with_the_info_field(self):
        body = (
            variant(
                "chr1",
                10,
                "BEFORE",
                "G",
                (".", ".", "DP=1", "GT", "0/1"),
            )
            + "\n"
            + variant("chr1", 30, "AFTER", "T", (".", ".", "."))
            + "\n"
        )
        result, mapped, _unmapped = self.lift("vcf", body, "samples.vcf")
        self.assertEqual(result.returncode, 0, result.stderr)
        before = data_lines(mapped, "BEFORE")[0].split("\t")
        # maxsplit keeps INFO and the sample columns in the tail.
        self.assertEqual(before[7:], ["DP=1", "GT", "0/1"])
        self.assertTrue(data_lines(mapped, "AFTER"))

    def test_short_line_on_an_unmapped_contig_does_not_drop_later_variants(self):
        body = "\n".join(
            [
                variant("chr1", 10, "BEFORE", "G", (".", ".", ".")),
                variant("chr9", 20, "SHORT", "C"),
                variant("chr1", 30, "AFTER", "T", (".", ".", ".")),
                "",
            ]
        )
        self.assert_lift_continues("vcf", body, "SHORT")

    def test_mappable_vcf_line_without_info_column_keeps_later_variants(self):
        # 5 columns: the attack input (no QUAL, FILTER, or INFO).
        # 6 columns: QUAL present, FILTER and INFO absent.
        # 7 columns: QUAL and FILTER present, INFO absent.
        widths = {
            5: (),
            6: (".",),
            7: (".", "PASS"),
        }
        for width, extra in widths.items():
            with self.subTest(columns=width):
                body = "\n".join(
                    [
                        variant("chr1", 10, "BEFORE", "G", (".", ".", ".")),
                        variant("chr1", 20, "SHORT", "C", extra),
                        variant("chr1", 30, "AFTER", "T", (".", ".", ".")),
                        "",
                    ]
                )
                self.assert_lift_continues("vcf", body, "SHORT")

    def test_mappable_short_line_with_no_comp_alleles_keeps_later_variants(self):
        body = "\n".join(
            [
                variant("chr1", 10, "BEFORE", "G", (".", ".", ".")),
                variant("chr1", 20, "SHORT", "C"),
                variant("chr1", 30, "AFTER", "T", (".", ".", ".")),
                "",
            ]
        )
        self.assert_lift_continues(
            "vcf", body, "SHORT", extra_args=("--no-comp-alleles",)
        )

    def test_mappable_short_line_with_ref_consistent_keeps_later_variants(self):
        body = "\n".join(
            [
                variant("chr1", 10, "BEFORE", "G", (".", ".", ".")),
                variant("chr1", 20, "SHORT", "A"),
                variant("chr1", 30, "AFTER", "T", (".", ".", ".")),
                "",
            ]
        )
        # REF matches the target base, so the consistency check does not
        # reject the short line before the INFO column is rewritten.
        self.assert_lift_continues(
            "vcf", body, "SHORT", extra_args=("--ref-consistent",)
        )

    def test_gvcf_variant_without_info_column_keeps_later_variants(self):
        widths = {
            5: (),
            6: (".",),
            7: (".", "PASS"),
        }
        for width, extra in widths.items():
            with self.subTest(columns=width):
                body = "\n".join(
                    [
                        variant("chr1", 10, "BEFORE", "G", (".", ".", ".")),
                        variant("chr1", 20, "SHORT", "C", extra),
                        variant("chr1", 30, "AFTER", "T", (".", ".", ".")),
                        "",
                    ]
                )
                self.assert_lift_continues("gvcf", body, "SHORT")

    def test_gvcf_reference_block_without_info_column_keeps_later_variants(self):
        # END= selects the non-variant block path, which also rewrites
        # column 8 after the block lifts. These lines contain END= but
        # still have no INFO column.
        short_blocks = {
            6: variant("chr1", 20, "BLOCK", "<NON_REF>", ("END=40",)),
            7: variant("chr1", 20, "BLOCK", "<NON_REF>", (".", "END=40")),
        }
        for width, block in short_blocks.items():
            with self.subTest(columns=width):
                body = "\n".join(
                    [
                        variant("chr1", 10, "BEFORE", "G", (".", ".", ".")),
                        block,
                        variant("chr1", 50, "AFTER", "T", (".", ".", ".")),
                        "",
                    ]
                )
                self.assert_lift_continues("gvcf", body, "BLOCK")

    def test_gvcf_reference_block_with_info_column_lifts(self):
        body = "\n".join(
            [
                variant("chr1", 10, "BEFORE", "G", (".", ".", ".")),
                variant(
                    "chr1", 20, "BLOCK", "<NON_REF>", (".", ".", "END=40")
                ),
                variant("chr1", 50, "AFTER", "T", (".", ".", ".")),
                "",
            ]
        )
        result, mapped, unmapped = self.lift("gvcf", body, "block.gvcf")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        block = data_lines(mapped, "BLOCK")
        self.assertEqual(len(block), 1, mapped)
        self.assertIn("END=40", block[0])
        self.assertTrue(data_lines(mapped, "BEFORE"))
        self.assertTrue(data_lines(mapped, "AFTER"))
        self.assertFalse(data_lines(unmapped, "BLOCK"))


if __name__ == "__main__":
    unittest.main()
