"""Genotype indexes must follow alternate alleles that liftover drops.

CrossMap rewrites a variant's REF from the target FASTA and then deletes any
alternate whose sequence is identical to that new reference. The written ALT
list is shorter, but the rest of the record is copied as it stood: GT indexes,
Number=R sample arrays (AD), Number=G sample arrays (PL), and Number=A INFO
counts (AC, AF).

These tests are the required behavior for both the ``gvcf`` and ``vcf``
commands. They fail while that copy is left unchanged.

Allele map used below, after the same alt rewrite the liftover already
performs (reverse-complement on the minus strand; otherwise the allele is
unchanged):

- an alternate that equals the new reference becomes index 0
- each surviving alternate keeps its relative order and is numbered from 1
- index 0 stays 0 when the genotype used the reference column
- ``.`` stays ``.``, and ``/`` versus ``|`` stays as it was

Allelic depth is moved with the sequence it belongs to. The depth written for
the new reference is the depth of the allele that now has that sequence (the
alternate that was removed), not the depth of the previous reference base.
Phred likelihoods are the diploid Number=G array from the VCF specification:
the value for alleles i <= j is at index ``j * (j + 1) / 2 + i``, taken from
the old allele numbers.

Number=A INFO fields AC and AF must be omitted or rewritten so each remaining
value lines up with a remaining alternate. A rewritten value is either the
original count for that alternate, or the count implied by the remapped
genotypes (diploid or haploid alleles that are not ``.``). NS is not an
allele count and must be kept. When no alternate is removed, AC, AF, AD, and
GT stay as they were, including after a minus-strand reverse complement.

A site whose every alternate matched the new reference may be left out of the
mapped file (the REF==ALT rejection) or written with ALT ``.`` or ALT equal
to REF, GT indexes of only 0, and AD reduced to the surviving reference
depth. ``--no-comp-alleles`` keeps that site, so the mapped record is
required. ``--ref-consistent`` still drops a duplicate alternate when the
submitted REF already matches the target base, and the genotype has to move
with it.

Not covered here: indel alleles (the target reference written for a variant
is the single base at the lifted position, so this filter does not remove an
indel alternate), MAF (no genotype indexes), and gVCF lines that contain
``END=`` (those take the reference-block path and do not filter alternates).
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"

HEADER = "\n".join([
    "##fileformat=VCFv4.2",
    "##contig=<ID=chr1,length=30>",
    '##INFO=<ID=NS,Number=1,Type=Integer,Description="Number of samples">',
    '##INFO=<ID=AC,Number=A,Type=Integer,Description="Allele count">',
    '##INFO=<ID=AF,Number=A,Type=Float,Description="Allele frequency">',
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
    '##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths">',
    '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read depth">',
    '##FORMAT=<ID=PL,Number=G,Type=Integer,Description="Phred likelihoods">',
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2",
])


def _chain(strand):
    return "chain 100 chr1 30 + 0 30 chr1 30 %s 0 30 1\n30\n" % strand


def _fasta(bases):
    seq = ["A"] * 30
    for index, base in bases.items():
        seq[index] = base
    return ">chr1\n" + "".join(seq) + "\n"


def _write(path, text):
    path.write_text(text)
    return path


def _run_crossmap(command, chain, vcf_text, fasta_text, flags):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        chain_path = _write(root / "liftover.chain", chain)
        vcf_path = _write(root / "input.vcf", HEADER + "\n" + vcf_text)
        fasta_path = _write(root / "target.fa", fasta_text)
        out_path = root / "lifted.vcf"
        argv = [
            "CrossMap", command, str(chain_path), str(vcf_path),
            str(fasta_path), str(out_path),
        ] + list(flags)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(SRC) + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        script = (
            "import sys\n"
            "from cmmodule import crossmap\n"
            "sys.argv = %r\n"
            "crossmap()\n" % (argv,)
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=tmp,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        mapped = out_path.read_text() if out_path.exists() else ""
        unmap_path = Path(str(out_path) + ".unmap")
        unmapped = unmap_path.read_text() if unmap_path.exists() else ""
        return completed, mapped, unmapped


def _parse(text):
    samples = []
    records = []
    for line in text.splitlines():
        if line.startswith("#CHROM"):
            samples = line.split("\t")[9:]
        elif not line or line.startswith("#"):
            continue
        else:
            cols = line.split("\t")
            record = {
                "chrom": cols[0],
                "pos": cols[1],
                "id": cols[2],
                "ref": cols[3],
                "alt": cols[4],
                "info": _info(cols[7]),
                "line": line,
                "samples": {},
            }
            if len(cols) > 8:
                keys = cols[8].split(":")
                for name, raw in zip(samples, cols[9:]):
                    record["samples"][name] = dict(zip(keys, raw.split(":")))
            records.append(record)
    return records


def _info(field):
    if field == ".":
        return {}
    parsed = {}
    for item in field.split(";"):
        if "=" in item:
            key, value = item.split("=", 1)
            parsed[key] = value
        else:
            parsed[item] = True
    return parsed


def _numbers(text):
    return [float(part) for part in text.split(",")]


def _gt_indexes(genotype):
    indexes = []
    for part in genotype.replace("|", "/").split("/"):
        if part != ".":
            indexes.append(int(part))
    return indexes


def _allele_count(alt):
    if alt in (".", ""):
        return 0
    return len(alt.split(","))


class AlleleIndexLiftoverTests(unittest.TestCase):
    maxDiff = None

    def test_forward_liftover_remaps_indexes_of_dropped_alts(self):
        """First or middle alternate equals the target base; the call used another allele."""
        for command in ("gvcf", "vcf"):
            with self.subTest(command=command):
                completed, mapped, unmapped = _run_crossmap(
                    command, _chain("+"), _forward_body(command),
                    _fasta({10: "C", 11: "T", 12: "C"}), [])
                self.assertEqual(
                    completed.returncode, 0,
                    "stdout:\n%s\nstderr:\n%s" % (
                        completed.stdout, completed.stderr))
                expected = _forward_expected()
                if command != "gvcf":
                    expected = [
                        spec for spec in expected if spec["id"] != "refblock"]
                self._assert_expectations(
                    _parse(mapped), unmapped, expected,
                    allow_only_match_unmap=True)

    def test_reverse_strand_remaps_indexes_after_reverse_complement(self):
        for command in ("gvcf", "vcf"):
            with self.subTest(command=command):
                completed, mapped, unmapped = _run_crossmap(
                    command, _chain("-"), _reverse_body(),
                    _fasta({19: "G", 18: "T", 17: "G"}), [])
                self.assertEqual(
                    completed.returncode, 0,
                    "stdout:\n%s\nstderr:\n%s" % (
                        completed.stdout, completed.stderr))
                self._assert_expectations(
                    _parse(mapped), unmapped, _reverse_expected(),
                    allow_only_match_unmap=False)

    def test_no_comp_alleles_still_remaps_and_keeps_monomorphic_site(self):
        for command in ("gvcf", "vcf"):
            with self.subTest(command=command):
                completed, mapped, unmapped = _run_crossmap(
                    command, _chain("+"), _forward_body(command),
                    _fasta({10: "C", 11: "T", 12: "C"}),
                    ["--no-comp-alleles"])
                self.assertEqual(
                    completed.returncode, 0,
                    "stdout:\n%s\nstderr:\n%s" % (
                        completed.stdout, completed.stderr))
                expected = _forward_expected()
                if command != "gvcf":
                    expected = [
                        spec for spec in expected if spec["id"] != "refblock"]
                self._assert_expectations(
                    _parse(mapped), unmapped, expected,
                    allow_only_match_unmap=False)

    def test_ref_consistent_remaps_when_submitted_ref_matches_target(self):
        """REF already equals the target base and the first ALT repeats it."""
        body = "\n".join([
            "chr1\t11\tconsistent\tC\tC,G\t.\t.\tNS=2;AC=1,4;AF=0.1,0.4\t"
            "GT:AD:DP:PL\t2/2:3,5,9:17:60,50,40,30,20,10\t"
            "1|2:3,5,9:17:6,5,4,3,2,1",
        ]) + "\n"
        expected = [_forward_expected()[0]]
        expected[0] = dict(expected[0])
        expected[0]["id"] = "consistent"
        for command in ("gvcf", "vcf"):
            with self.subTest(command=command):
                completed, mapped, unmapped = _run_crossmap(
                    command, _chain("+"), body, _fasta({10: "C"}),
                    ["--ref-consistent"])
                self.assertEqual(
                    completed.returncode, 0,
                    "stdout:\n%s\nstderr:\n%s" % (
                        completed.stdout, completed.stderr))
                self._assert_expectations(
                    _parse(mapped), unmapped, expected,
                    allow_only_match_unmap=False)

    def _assert_expectations(self, records, unmapped, expected, allow_only_match_unmap):
        by_id = {}
        for record in records:
            self.assertNotEqual(
                record["alt"], "",
                "mapped ALT is empty: %s" % record["line"])
            n_alt = _allele_count(record["alt"])
            for name, sample in record["samples"].items():
                if "GT" not in sample:
                    continue
                for index in _gt_indexes(sample["GT"]):
                    self.assertGreaterEqual(index, 0, record["line"])
                    self.assertLessEqual(
                        index, n_alt,
                        "genotype index is outside the written alleles: %s"
                        % record["line"])
            self.assertNotIn(
                record["id"], by_id,
                "duplicate mapped id %s" % record["id"])
            by_id[record["id"]] = record

        for spec in expected:
            with self.subTest(record=spec["id"]):
                record = by_id.get(spec["id"])
                if record is None:
                    self.assertTrue(
                        spec.get("unmap_ok") and allow_only_match_unmap,
                        "record %s was not mapped\n%s" % (spec["id"], unmapped))
                    self.assertIn("\t%s\t" % spec["id"], "\t" + unmapped)
                    continue
                if spec.get("unmap_ok") and allow_only_match_unmap:
                    self.assertNotIn(
                        "\t%s\t" % spec["id"], "\t" + unmapped,
                        "record %s was written and also unmapped" % spec["id"])
                self.assertEqual(record["chrom"], "chr1")
                self.assertEqual(record["pos"], spec["pos"], record["line"])
                self.assertEqual(record["ref"], spec["ref"], record["line"])
                if "alt" in spec:
                    self.assertEqual(record["alt"], spec["alt"], record["line"])
                else:
                    self.assertIn(record["alt"], spec["alt_in"], record["line"])
                if "ns" in spec:
                    self.assertEqual(record["info"].get("NS"), spec["ns"])
                if "end" in spec:
                    self.assertEqual(record["info"].get("END"), spec["end"])
                self._assert_optional_count(record, "AC", spec.get("ac"))
                self._assert_optional_count(record, "AF", spec.get("af"))
                self.assertEqual(set(record["samples"]), set(spec["samples"]))
                for name, fields in spec["samples"].items():
                    for key, value in fields.items():
                        self.assertEqual(
                            record["samples"][name].get(key), value,
                            "%s %s %s" % (spec["id"], name, record["line"]))

    def _assert_optional_count(self, record, key, allowed):
        if not allowed:
            return
        if key not in record["info"]:
            return
        got = _numbers(record["info"][key])
        matches = any(
            len(got) == len(choice) and all(
                abs(left - right) < 1e-6 for left, right in zip(got, choice))
            for choice in allowed)
        self.assertTrue(
            matches,
            "%s %s=%s is not one of %s (%s)" % (
                record["id"], key, record["info"][key], allowed, record["line"]))


def _forward_body(command):
    rows = [
        "chr1\t11\tsecond_alt\tA\tC,G\t.\t.\tNS=2;AC=1,4;AF=0.1,0.4\t"
        "GT:AD:DP:PL\t2/2:3,5,9:17:60,50,40,30,20,10\t"
        "1|2:3,5,9:17:6,5,4,3,2,1",
        "chr1\t11\tcall_removed\tA\tC,G\t.\t.\tNS=2\t"
        "GT:AD:DP\t1/1:3,5,9:17\t0/2:3,5,9:17",
        "chr1\t13\tmiddle\tA\tT,C,G\t.\t.\tNS=2;AC=5,6,7;AF=0.2,0.3,0.4\t"
        "GT:AD:DP\t3/3:1,2,3,4:10\t1/3:1,2,3,4:10",
        "chr1\t13\tmiddle_called\tA\tT,C,G\t.\t.\tNS=2\t"
        "GT:AD:DP\t2|2:1,2,3,4:10\t2/1:1,2,3,4:10",
        "chr1\t13\tsymbolic\tA\tT,C,<NON_REF>\t.\t.\tNS=2\t"
        "GT:AD\t2/2:8,9,1,4\t3:8,9,1,4",
        "chr1\t11\tmissing_mix\tA\tC,G\t.\t.\tNS=2\t"
        "GT:AD\t./2:3,5,9\t.|2:3,5,9",
        "chr1\t11\ttail_missing\tA\tC,G\t.\t.\tNS=2\t"
        "GT:AD\t2/.:3,5,9\t2|.:3,5,9",
        "chr1\t11\tpolyploid\tA\tC,G\t.\t.\tNS=2\t"
        "GT:AD\t2/1/2:3,5,9\t././.:3,5,9",
        "chr1\t11\tonly_match\tA\tC\t.\t.\tNS=2\t"
        "GT:AD\t1/1:3,5\t./.:3,5",
        "chr1\t12\tkeep_both\tA\tC,G\t.\t.\tNS=2;AC=1,4\t"
        "GT:AD\t2/2:1,2,3\t1/1:1,2,3",
    ]
    if command == "gvcf":
        rows.append(
            "chr1\t1\trefblock\tN\t<NON_REF>\t.\t.\tEND=5\t"
            "GT:DP\t0/0:10\t0/0:10")
    return "\n".join(rows) + "\n"


def _forward_expected():
    # Target bases: POS 11 = C, POS 12 = T, POS 13 = C.
    # second_alt / call_removed / missing / polyploid: ALT C matches C and is
    # removed; G remains allele 1. Depths 3,5,9 belong to A,C,G so the written
    # alleles C,G take 5,9. PL for old alleles 0,1,2 at 2/2 and 1/2 moves to
    # the new diploid genotypes 1/1 and 0/1.
    # middle: T,C,G with C removed leaves T,G. Depths 1,2,3,4 (A,T,C,G) become
    # 3,2,4 for C,T,G. AC/AF subset keeps T and G (5,7 and 0.2,0.4). The two
    # samples 2/2 + 1/2 contribute T=1, G=3.
    second = {
        "id": "second_alt",
        "pos": "11",
        "ref": "C",
        "alt": "G",
        "ns": "2",
        "ac": ([4], [3]),
        "af": ([0.4], [0.75]),
        "samples": {
            "S1": {"GT": "1/1", "AD": "5,9", "DP": "17", "PL": "40,20,10"},
            "S2": {"GT": "0|1", "AD": "5,9", "DP": "17", "PL": "4,2,1"},
        },
    }
    return [
        second,
        {
            "id": "call_removed",
            "pos": "11",
            "ref": "C",
            "alt": "G",
            "ns": "2",
            "samples": {
                "S1": {"GT": "0/0", "AD": "5,9", "DP": "17"},
                "S2": {"GT": "0/1", "AD": "5,9", "DP": "17"},
            },
        },
        {
            "id": "middle",
            "pos": "13",
            "ref": "C",
            "alt": "T,G",
            "ns": "2",
            "ac": ([5, 7], [1, 3]),
            "af": ([0.2, 0.4], [0.25, 0.75]),
            "samples": {
                "S1": {"GT": "2/2", "AD": "3,2,4", "DP": "10"},
                "S2": {"GT": "1/2", "AD": "3,2,4", "DP": "10"},
            },
        },
        {
            "id": "middle_called",
            "pos": "13",
            "ref": "C",
            "alt": "T,G",
            "ns": "2",
            "samples": {
                "S1": {"GT": "0|0", "AD": "3,2,4", "DP": "10"},
                "S2": {"GT": "0/1", "AD": "3,2,4", "DP": "10"},
            },
        },
        {
            "id": "symbolic",
            "pos": "13",
            "ref": "C",
            "alt": "T,<NON_REF>",
            "ns": "2",
            "samples": {
                "S1": {"GT": "0/0", "AD": "1,9,4"},
                "S2": {"GT": "2", "AD": "1,9,4"},
            },
        },
        {
            "id": "missing_mix",
            "pos": "11",
            "ref": "C",
            "alt": "G",
            "ns": "2",
            "samples": {
                "S1": {"GT": "./1", "AD": "5,9"},
                "S2": {"GT": ".|1", "AD": "5,9"},
            },
        },
        {
            "id": "tail_missing",
            "pos": "11",
            "ref": "C",
            "alt": "G",
            "ns": "2",
            "samples": {
                "S1": {"GT": "1/.", "AD": "5,9"},
                "S2": {"GT": "1|.", "AD": "5,9"},
            },
        },
        {
            "id": "polyploid",
            "pos": "11",
            "ref": "C",
            "alt": "G",
            "ns": "2",
            "samples": {
                "S1": {"GT": "1/0/1", "AD": "5,9"},
                "S2": {"GT": "././.", "AD": "5,9"},
            },
        },
        {
            "id": "only_match",
            "pos": "11",
            "ref": "C",
            "alt_in": [".", "C"],
            "ns": "2",
            "unmap_ok": True,
            "samples": {
                "S1": {"GT": "0/0", "AD": "5"},
                "S2": {"GT": "./.", "AD": "5"},
            },
        },
        {
            "id": "keep_both",
            "pos": "12",
            "ref": "T",
            "alt": "C,G",
            "ns": "2",
            "ac": ([1, 4],),
            "samples": {
                "S1": {"GT": "2/2", "AD": "1,2,3"},
                "S2": {"GT": "1/1", "AD": "1,2,3"},
            },
        },
        {
            "id": "refblock",
            "pos": "1",
            "ref": "N",
            "alt": "<NON_REF>",
            "samples": {
                "S1": {"GT": "0/0", "DP": "10"},
                "S2": {"GT": "0/0", "DP": "10"},
            },
            "end": "5",
        },
    ]


def _reverse_body():
    return "\n".join([
        "chr1\t11\trev_second\tA\tC,G\t.\t.\tNS=2;AC=1,4;AF=0.1,0.4\t"
        "GT:AD:DP:PL\t2/2:3,5,9:17:60,50,40,30,20,10\t"
        "1|2:3,5,9:17:6,5,4,3,2,1",
        "chr1\t11\trev_removed\tA\tC,G\t.\t.\tNS=2\t"
        "GT:AD:DP\t1/1:3,5,9:17\t0/2:3,5,9:17",
        "chr1\t13\trev_which\tA\tT,C,G\t.\t.\tNS=2;AC=5,6,7;AF=0.2,0.3,0.4\t"
        "GT:AD:DP\t2/2:1,2,3,4:10\t3/1:1,2,3,4:10",
        "chr1\t12\trev_keep\tA\tC,G\t.\t.\tNS=2;AC=1,4\t"
        "GT:AD\t2/2:1,2,3\t1/1:1,2,3",
    ]) + "\n"


def _reverse_expected():
    # Minus strand, chromosome length 30. Source POS 11, 12, 13 land on
    # target POS 20, 19, 18, whose bases are G, T, G.
    # rev_second: C,G complement to G,C. G equals the new reference, so C
    # remains allele 1. Depths follow C then the surviving complement of G.
    # rev_which: T,C,G complement to A,G,C. G equals the new reference, so
    # the middle input allele is the one removed. Survivors are A,C.
    return [
        {
            "id": "rev_second",
            "pos": "20",
            "ref": "G",
            "alt": "C",
            "ns": "2",
            "ac": ([4], [3]),
            "af": ([0.4], [0.75]),
            "samples": {
                "S1": {"GT": "1/1", "AD": "5,9", "DP": "17", "PL": "40,20,10"},
                "S2": {"GT": "0|1", "AD": "5,9", "DP": "17", "PL": "4,2,1"},
            },
        },
        {
            "id": "rev_removed",
            "pos": "20",
            "ref": "G",
            "alt": "C",
            "ns": "2",
            "samples": {
                "S1": {"GT": "0/0", "AD": "5,9", "DP": "17"},
                "S2": {"GT": "0/1", "AD": "5,9", "DP": "17"},
            },
        },
        {
            "id": "rev_which",
            "pos": "18",
            "ref": "G",
            "alt": "A,C",
            "ns": "2",
            "ac": ([5, 7], [1, 1]),
            "af": ([0.2, 0.4], [0.25, 0.25]),
            "samples": {
                "S1": {"GT": "0/0", "AD": "3,2,4", "DP": "10"},
                "S2": {"GT": "2/1", "AD": "3,2,4", "DP": "10"},
            },
        },
        {
            "id": "rev_keep",
            "pos": "19",
            "ref": "T",
            "alt": "G,C",
            "ns": "2",
            "ac": ([1, 4],),
            "samples": {
                "S1": {"GT": "2/2", "AD": "1,2,3"},
                "S2": {"GT": "1/1", "AD": "1,2,3"},
            },
        },
    ]


if __name__ == "__main__":
    unittest.main()
