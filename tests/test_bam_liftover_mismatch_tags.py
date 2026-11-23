"""Stale NM and MD tags after BAM/SAM/CRAM liftover.

CrossMap's bam command rewrites an alignment's chromosome, position, and, when
the chain flips the strand, its sequence and CIGAR. NM (edit distance) and MD
(mismatch string) still describe the read against the source genome.

These tests are the expected behavior: after a successful liftover, NM and MD
are either absent or equal to the edit distance and mismatch string of the
lifted alignment against the target reference. They fail while the tags are
copied unchanged.

Run from the repository root:

    pip install pysam bx-python pyBigWig pytest
    PYTHONPATH=src python3 -m pytest tests/test_bam_liftover_mismatch_tags.py

Reads that CrossMap emits as unmapped are not asserted. That includes QC
failures, alignments with no chain hit, and paired records whose own alignment
does not lift (the mate-only NU path, and the append-tags path that overwrites
NM while the read stays unmapped).
"""

import contextlib
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pysam
from cmmodule import crossmap


SRC_LEN = 1000
TGT_LEN = 2000
SRC_CHROM = "chr1"
# Source assembly is poly-A. Target assemblies are poly-C, so a read that
# matched the source is not an exact match after liftover.
SRC_BASE = "A"
TGT_BASE = "C"

MATCH, INS, DEL = 0, 1, 2


@dataclass(frozen=True)
class Block:
    s_start: int
    s_end: int
    t_name: str
    t_strand: str
    t_forward_start: int
    cid: int


@dataclass(frozen=True)
class Read:
    name: str
    start: int
    sequence: str
    cigar: tuple
    nm: int
    md: str
    mate_start: int | None = None  # None => single-end; -1 => mate unmapped

    @property
    def paired(self):
        return self.mate_start is not None

    @property
    def mate_unmapped(self):
        return self.mate_start == -1


@dataclass(frozen=True)
class Case:
    name: str
    blocks: tuple
    reads: tuple
    fmt: str = "bam"  # bam, sam, cram
    append_tags: bool = False


def _ref_len(cigar):
    return sum(length for op, length in cigar if op in (MATCH, DEL))


def _chains_covering(start, end, blocks):
    return [
        block for block in blocks
        if not (end <= block.s_start or start >= block.s_end)
    ]


def _path_expectation(read, blocks):
    """Which successful-liftover branch this read should hit."""
    end = read.start + _ref_len(read.cigar)
    covering = _chains_covering(read.start, end, blocks)
    if len(covering) == 0:
        raise AssertionError("%s is not covered by the chain" % read.name)
    strands = {block.t_strand for block in covering}
    if len(strands) != 1:
        raise AssertionError("%s is covered by mixed strands" % read.name)
    n_self = len(covering)
    if not read.paired or read.mate_unmapped:
        n_mate = 0
    else:
        n_mate = len(_chains_covering(read.mate_start, read.mate_start + 1, blocks))
        if n_mate == 0:
            raise AssertionError("%s mate is not covered by the chain" % read.name)

    expect_reverse = strands.pop() == "-"
    expect_secondary = n_self > 1 or n_mate > 1
    if not read.paired:
        crossmap_tag = "SM" if n_self > 1 else "SU"
    elif n_mate == 0:
        crossmap_tag = "MN" if n_self > 1 else "UN"
    elif n_self == 1 and n_mate == 1:
        crossmap_tag = "UU"
    elif n_self == 1 and n_mate > 1:
        crossmap_tag = "UM"
    elif n_self > 1 and n_mate == 1:
        crossmap_tag = "MU"
    else:
        crossmap_tag = "MM"
    return expect_reverse, expect_secondary, crossmap_tag


def _revcomp(sequence):
    return sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def compute_nm_md(sequence, cigar, reference, ref_start):
    """Edit distance and MD string of one alignment against a reference.

    Covers the CIGAR operations these tests write: M, I, and D. MD records
    match lengths, the reference base at each mismatch, and deleted reference
    bases. Insertions increment NM and do not appear in MD. This matches
    samtools calmd on the alignments below; test setup checks that.
    """
    nm = 0
    parts = []
    matches = 0
    query_index = 0
    ref_index = ref_start

    def flush():
        nonlocal matches
        parts.append(str(matches))
        matches = 0

    for op, length in cigar:
        if op == MATCH:
            for _ in range(length):
                query_base = sequence[query_index].upper()
                ref_base = reference[ref_index].upper()
                if query_base == ref_base:
                    matches += 1
                else:
                    flush()
                    parts.append(ref_base)
                    nm += 1
                query_index += 1
                ref_index += 1
        elif op == INS:
            query_index += length
            nm += length
        elif op == DEL:
            flush()
            deleted = reference[ref_index:ref_index + length].upper()
            parts.append("^" + deleted)
            nm += length
            ref_index += length
        else:
            raise AssertionError("unsupported CIGAR op %s" % op)
    flush()
    return nm, "".join(parts)


def _write_fasta(path, contigs):
    with path.open("w") as handle:
        for name, sequence in contigs:
            handle.write(">%s\n%s\n" % (name, sequence))
    pysam.faidx(str(path))


def _write_chain(path, blocks):
    lines = []
    for block in blocks:
        size = block.s_end - block.s_start
        if block.t_strand == "+":
            t_start = block.t_forward_start
            t_end = t_start + size
        else:
            # Chain-direction start. map_coordinates converts this to forward
            # target coordinates when the target strand is '-'.
            t_start = 0
            t_end = size
        lines.append(
            "chain %d %s %d + %d %d %s %d %s %d %d %d"
            % (
                1000 - block.cid,
                SRC_CHROM,
                SRC_LEN,
                block.s_start,
                block.s_end,
                block.t_name,
                TGT_LEN,
                block.t_strand,
                t_start,
                t_end,
                block.cid,
            )
        )
        lines.append(str(size))
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def _alignment(read, reference_id):
    record = pysam.AlignedSegment()
    record.query_name = read.name
    record.query_sequence = read.sequence
    record.query_qualities = pysam.qualitystring_to_array("I" * len(read.sequence))
    record.reference_id = reference_id
    record.reference_start = read.start
    record.cigartuples = list(read.cigar)
    record.mapping_quality = 40
    if not read.paired:
        record.flag = 0
    else:
        record.flag = 0x1 | 0x40
        if read.mate_unmapped:
            record.flag |= 0x8
            record.next_reference_id = -1
            record.next_reference_start = -1
        else:
            record.next_reference_id = reference_id
            record.next_reference_start = read.mate_start
    # Tags a standard aligner writes, plus two tags the liftover must keep.
    record.set_tag("NM", read.nm, "i")
    record.set_tag("MD", read.md, "Z")
    record.set_tag("AS", 60, "i")
    record.set_tag("RG", "sample1", "Z")
    return record


def _write_input(path, reads, fmt, source_fasta):
    header = {
        "HD": {"VN": "1.6", "SO": "unsorted"},
        "SQ": [{"SN": SRC_CHROM, "LN": SRC_LEN}],
        "RG": [{"ID": "sample1", "SM": "sample1"}],
    }
    records = [_alignment(read, 0) for read in reads]
    if fmt == "bam":
        with pysam.AlignmentFile(str(path), "wb", header=header) as out:
            for record in records:
                out.write(record)
    elif fmt == "sam":
        with pysam.AlignmentFile(str(path), "w", header=header) as out:
            for record in records:
                out.write(record)
    elif fmt == "cram":
        with pysam.AlignmentFile(
            str(path), "wc", header=header, reference_filename=str(source_fasta)
        ) as out:
            for record in records:
                out.write(record)
    else:
        raise AssertionError(fmt)


def _output_files(prefix, fmt):
    if fmt == "sam":
        produced = [prefix.with_suffix(".sam")]
    else:
        produced = [Path(str(prefix) + ".bam"), Path(str(prefix) + ".sorted.bam")]
    missing = [path for path in produced if not path.exists()]
    if missing:
        raise AssertionError("liftover did not write %s" % ", ".join(map(str, missing)))
    return produced


def _records_by_name(path):
    with pysam.AlignmentFile(str(path)) as handle:
        found = {}
        for record in handle:
            found[record.query_name] = record
        return found


def _run_crossmap(case, work, infile, chain):
    prefix = work / "lifted"
    argv = ["CrossMap", "bam", str(chain), str(infile), str(prefix)]
    if case.append_tags:
        argv.append("--append-tags")
    previous = sys.argv
    log = work / "crossmap.log"
    sys.argv = argv
    try:
        with log.open("w") as handle, contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
            crossmap()
    except SystemExit as exc:
        raise AssertionError(
            "CrossMap bam exited %s\n%s" % (exc.code, log.read_text())
        )
    finally:
        sys.argv = previous
    return _output_files(prefix, case.fmt)


_ORACLE_CHECKED = False


def _check_oracle_against_calmd(work):
    """Fail the run if the MD/NM calculator disagrees with samtools calmd."""
    global _ORACLE_CHECKED
    if _ORACLE_CHECKED:
        return
    fasta = work / "oracle.fa"
    reference = TGT_BASE * 200
    _write_fasta(fasta, [("chrT", reference)])
    samples = [
        ("polyA", 5, "A" * 10, ((MATCH, 10),)),
        ("mismatch", 20, "A" * 9 + "G", ((MATCH, 10),)),
        ("deletion", 40, "A" * 10, ((MATCH, 5), (DEL, 2), (MATCH, 5))),
        ("insertion", 60, "A" * 10, ((MATCH, 2), (INS, 3), (MATCH, 5))),
        ("rev_ins", 80, _revcomp("A" * 10), ((MATCH, 5), (INS, 3), (MATCH, 2))),
    ]
    header = {"HD": {"VN": "1.6"}, "SQ": [{"SN": "chrT", "LN": len(reference)}]}
    raw = work / "oracle.bam"
    with pysam.AlignmentFile(str(raw), "wb", header=header) as out:
        for name, start, sequence, cigar in samples:
            record = pysam.AlignedSegment()
            record.query_name = name
            record.query_sequence = sequence
            record.query_qualities = pysam.qualitystring_to_array("I" * len(sequence))
            record.reference_id = 0
            record.reference_start = start
            record.cigartuples = list(cigar)
            record.flag = 0
            record.mapping_quality = 40
            out.write(record)
    calculated = work / "oracle.calmd.bam"
    pysam.calmd(
        "-b",
        str(raw),
        str(fasta),
        save_stdout=str(calculated),
        catch_stdout=False,
    )
    with pysam.AlignmentFile(str(calculated), "rb") as handle:
        for record in handle:
            sample = next(item for item in samples if item[0] == record.query_name)
            expected_nm, expected_md = compute_nm_md(
                sample[2], sample[3], reference, sample[1]
            )
            assert int(record.get_tag("NM")) == expected_nm
            assert record.get_tag("MD") == expected_md
    _ORACLE_CHECKED = True


def _assert_read(read, record, blocks, references, append_tags):
    expect_reverse, expect_secondary, crossmap_tag = _path_expectation(read, blocks)
    assert not record.is_unmapped, read.name
    assert record.reference_name != SRC_CHROM, read.name
    assert record.reference_name in references, read.name
    assert record.is_reverse == expect_reverse, read.name
    assert record.is_secondary == expect_secondary, read.name
    assert record.is_paired == read.paired, read.name
    expected_sequence = _revcomp(read.sequence) if expect_reverse else read.sequence
    expected_cigar = list(reversed(read.cigar)) if expect_reverse else list(read.cigar)
    assert record.query_sequence == expected_sequence, read.name
    assert list(record.cigartuples) == expected_cigar, read.name
    # Unrelated aligner tags have to survive a fix that only drops NM and MD.
    assert int(record.get_tag("AS")) == 60, read.name
    assert record.get_tag("RG") == "sample1", read.name
    if append_tags:
        assert record.has_tag(crossmap_tag), (
            "%s did not reach the %s liftover branch" % (read.name, crossmap_tag)
        )

    reference = references[record.reference_name]
    expected_nm, expected_md = compute_nm_md(
        record.query_sequence, record.cigartuples, reference, record.reference_start
    )
    if record.cigarstring == "10M" and set(record.query_sequence) in ({"A"}, {"T"}):
        # poly-A/poly-T against poly-C is ten mismatches. Locks the oracle
        # to the value samtools calmd produced for this fixture.
        assert (expected_nm, expected_md) == (
            10,
            "0C0C0C0C0C0C0C0C0C0C0",
        ), read.name
    assert (expected_nm, expected_md) != (read.nm, read.md), (
        "%s fixture does not change the mismatch pattern" % read.name
    )

    got_nm = int(record.get_tag("NM")) if record.has_tag("NM") else None
    got_md = record.get_tag("MD") if record.has_tag("MD") else None
    if isinstance(got_md, bytes):
        got_md = got_md.decode()
    problems = []
    where = "%s at %s:%d" % (
        read.name, record.reference_name, record.reference_start + 1
    )
    if got_nm is not None and got_nm != expected_nm:
        problems.append(
            "%s keeps edit distance NM:i:%s; the lifted alignment against "
            "the target reference has NM:i:%d" % (where, got_nm, expected_nm)
        )
    if got_md is not None and got_md != expected_md:
        problems.append(
            "%s keeps mismatch tag MD:Z:%s; the target reference at the "
            "lifted alignment has MD:Z:%s" % (where, got_md, expected_md)
        )
    if problems:
        raise AssertionError("; ".join(problems))


def _prepare(case, work):
    work.mkdir(parents=True, exist_ok=True)
    _check_oracle_against_calmd(work)
    for read in case.reads:
        _path_expectation(read, case.blocks)
        query_ops = sum(length for op, length in read.cigar if op in (MATCH, INS))
        if query_ops != len(read.sequence):
            raise AssertionError("%s CIGAR does not match the sequence" % read.name)
    source = SRC_BASE * SRC_LEN
    targets = []
    seen = set()
    for block in case.blocks:
        if block.t_name not in seen:
            seen.add(block.t_name)
            targets.append((block.t_name, TGT_BASE * TGT_LEN))
    source_fasta = work / "source.fa"
    _write_fasta(source_fasta, [(SRC_CHROM, source)])
    references = {name: sequence for name, sequence in targets}
    chain = work / "liftover.chain"
    _write_chain(chain, case.blocks)
    suffix = {"bam": ".bam", "sam": ".sam", "cram": ".cram"}[case.fmt]
    infile = work / ("input" + suffix)
    _write_input(infile, case.reads, case.fmt, source_fasta)
    outputs = _run_crossmap(case, work, infile, chain)
    loaded = [(path, _records_by_name(path)) for path in outputs]
    for path, found in loaded:
        missing = [read.name for read in case.reads if read.name not in found]
        if missing:
            raise AssertionError("%s is missing reads %s" % (path, missing))
    return loaded, references


EXACT = Read("exact", 10, "A" * 10, ((MATCH, 10),), 0, "10")
MISMATCH = Read("mismatch", 40, "A" * 9 + "G", ((MATCH, 10),), 1, "9A0")
DELETION = Read(
    "deletion", 80, "A" * 10, ((MATCH, 5), (DEL, 2), (MATCH, 5)), 2, "5^AA5"
)
INSERTION = Read(
    "insertion", 120, "A" * 10, ((MATCH, 2), (INS, 3), (MATCH, 5)), 3, "7"
)
SINGLE_READS = (EXACT, MISMATCH, DELETION, INSERTION)

UN_EXACT = Read("un_exact", 200, "A" * 10, ((MATCH, 10),), 0, "10", mate_start=-1)
UU_EXACT = Read("uu_exact", 300, "A" * 10, ((MATCH, 10),), 0, "10", mate_start=400)
UN_DELETION = Read(
    "un_deletion",
    150,
    "A" * 10,
    ((MATCH, 5), (DEL, 2), (MATCH, 5)),
    2,
    "5^AA5",
    mate_start=-1,
)
PAIRED_UNIQUE_READS = (UN_EXACT, UN_DELETION, UU_EXACT)

UM_EXACT = Read("um_exact", 10, "A" * 10, ((MATCH, 10),), 0, "10", mate_start=800)
MU_EXACT = Read("mu_exact", 800, "A" * 10, ((MATCH, 10),), 0, "10", mate_start=10)
MU_DELETION = Read(
    "mu_deletion",
    820,
    "A" * 10,
    ((MATCH, 5), (DEL, 2), (MATCH, 5)),
    2,
    "5^AA5",
    mate_start=10,
)
PAIRED_SPLIT_READS = (UM_EXACT, MU_EXACT, MU_DELETION)

MN_EXACT = Read("mn_exact", 800, "A" * 10, ((MATCH, 10),), 0, "10", mate_start=-1)
MM_EXACT = Read("mm_exact", 850, "A" * 10, ((MATCH, 10),), 0, "10", mate_start=900)
MN_DELETION = Read(
    "mn_deletion",
    910,
    "A" * 10,
    ((MATCH, 5), (DEL, 2), (MATCH, 5)),
    2,
    "5^AA5",
    mate_start=-1,
)
PAIRED_MULTI_READS = (MN_EXACT, MN_DELETION, MM_EXACT)

UNIQUE_PLUS = (
    Block(0, 1000, "chr2", "+", 1000, 1),
)
UNIQUE_MINUS = (
    Block(0, 1000, "chr2", "-", 0, 1),
)
MULTI_PLUS = (
    Block(0, 1000, "chr2", "+", 1000, 1),
    Block(0, 1000, "chr3", "+", 1000, 2),
)
MULTI_MINUS = (
    Block(0, 1000, "chr2", "-", 0, 1),
    Block(0, 1000, "chr3", "-", 0, 2),
)
SPLIT_PLUS = (
    Block(0, 500, "chr2", "+", 1000, 1),
    Block(700, 1000, "chr2", "+", 1700, 2),
    Block(700, 1000, "chr3", "+", 1700, 3),
)
SPLIT_MINUS = (
    Block(0, 500, "chr2", "-", 0, 1),
    Block(700, 1000, "chr2", "-", 0, 2),
    Block(700, 1000, "chr3", "-", 0, 3),
)

CASES = (
    Case("single_unique_same_strand", UNIQUE_PLUS, SINGLE_READS),
    Case("single_unique_opposite_strand", UNIQUE_MINUS, SINGLE_READS),
    Case("single_multiple_same_strand", MULTI_PLUS, SINGLE_READS),
    Case("single_multiple_opposite_strand", MULTI_MINUS, SINGLE_READS),
    Case("paired_unique_same_strand", UNIQUE_PLUS, PAIRED_UNIQUE_READS),
    Case("paired_unique_opposite_strand", UNIQUE_MINUS, PAIRED_UNIQUE_READS),
    Case("paired_split_same_strand", SPLIT_PLUS, PAIRED_SPLIT_READS),
    Case("paired_split_opposite_strand", SPLIT_MINUS, PAIRED_SPLIT_READS),
    Case("paired_multiple_same_strand", MULTI_PLUS, PAIRED_MULTI_READS),
    Case("paired_multiple_opposite_strand", MULTI_MINUS, PAIRED_MULTI_READS),
    Case("sam_unique_same_strand", UNIQUE_PLUS, SINGLE_READS, fmt="sam"),
    Case("cram_unique_same_strand", UNIQUE_PLUS, SINGLE_READS, fmt="cram"),
    Case(
        "append_tags_unique_same_strand",
        UNIQUE_PLUS,
        SINGLE_READS + PAIRED_UNIQUE_READS,
        append_tags=True,
    ),
    Case(
        "append_tags_unique_opposite_strand",
        UNIQUE_MINUS,
        SINGLE_READS + PAIRED_UNIQUE_READS,
        append_tags=True,
    ),
)


@pytest.fixture(scope="module")
def lifted(tmp_path_factory):
    base = tmp_path_factory.mktemp("liftover")
    prepared = {}
    for case in CASES:
        prepared[case.name] = _prepare(case, base / case.name)
    return prepared


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_lifted_nm_md_match_target_or_are_absent(case, lifted):
    """NM and MD from the source genome must not be written at the new locus."""
    loaded, references = lifted[case.name]
    failures = []
    for path, found in loaded:
        for read in case.reads:
            try:
                _assert_read(
                    read,
                    found[read.name],
                    case.blocks,
                    references,
                    case.append_tags,
                )
            except AssertionError as exc:
                failures.append("%s: %s" % (path.name, exc))
    if failures:
        raise AssertionError("\n".join(failures))
