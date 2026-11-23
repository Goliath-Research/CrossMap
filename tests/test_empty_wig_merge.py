"""Wiggle and bigWig liftover must finish when every interval is dropped.

An input whose records all fail to lift (chromosome missing from the chain,
span crossing more than one chain block, or span lying in a chain gap) makes
the intermediate bedGraph empty. Merging that file must not raise. The command
must exit 0 and write a readable bigWig that carries the chain target
chromosome header and no signal. Intervals that lift through a single chain
block must still be written.

These tests fail while the empty-group merge indexes a missing last line.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pyBigWig


REPO = Path(__file__).resolve().parents[1]

# Source chr1 [0, 100) -> target chrT [0, 100), then a 50 bp source gap,
# then source chr1 [150, 250) -> target chrT [100, 200).
CHAIN = """\
chain 100 chr1 1000 + 0 1000 chrT 1000 + 0 1000 1
100 50 0
100
"""

TARGET_CHROMS = {"chrT": 1000}


def run_crossmap(argv, cwd):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-c", "from cmmodule import crossmap; crossmap()", *argv],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )


def write_bigwig(path, chrom, size, intervals):
    bw = pyBigWig.open(str(path), "w")
    bw.addHeader([(chrom, size)])
    if intervals:
        starts = [start for start, _end, _value in intervals]
        ends = [end for _start, end, _value in intervals]
        values = [value for _start, _end, value in intervals]
        bw.addEntries([chrom] * len(intervals), starts, ends=ends, values=values)
    bw.close()


class EmptyWigMergeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.work = Path(self._tmp.name)
        (self.work / "map.chain").write_text(CHAIN)

    def tearDown(self):
        self._tmp.cleanup()

    def assert_finished_track(self, result, prefix, intervals):
        self.assertEqual(
            result.returncode,
            0,
            "CrossMap exited %s\nstdout:\n%s\nstderr:\n%s"
            % (result.returncode, result.stdout, result.stderr),
        )
        self.assertNotIn("Traceback", result.stderr)
        self.assertNotIn("IndexError", result.stderr)
        self.assertNotIn("error while finishing writing", result.stderr)

        bw_path = self.work / (prefix + ".bw")
        bw = pyBigWig.open(str(bw_path))
        try:
            self.assertEqual(bw.chroms(), TARGET_CHROMS)
            self.assertEqual(bw.intervals("chrT"), intervals)
        finally:
            bw.close()

        self.assertFalse(
            (self.work / (prefix + ".bgr")).exists(),
            "unmerged bedGraph was left behind",
        )
        self.assertTrue((self.work / (prefix + ".sorted.bgr")).is_file())
        leftover = sorted(path.name for path in self.work.iterdir())
        allowed = {"map.chain", prefix + ".bw", prefix + ".sorted.bgr"}
        allowed.update(
            path.name for path in self.work.iterdir()
            if path.suffix in {".wig", ".bw"} and not path.name.startswith(prefix)
        )
        # The input bigWig uses the .bw suffix but so does the output. Inputs
        # are recorded by the caller via extra names below.
        unexpected = [
            name for name in leftover
            if name not in allowed and name not in getattr(self, "_inputs", set())
        ]
        self.assertEqual(
            unexpected,
            [],
            "scratch or intermediate files left in %s: %s" % (self.work, leftover),
        )

    def _run(self, command, infile_name, prefix):
        self._inputs = {infile_name, "map.chain"}
        return run_crossmap(
            [command, "map.chain", infile_name, prefix],
            self.work,
        )

    def test_wig_bedgraph_chromosome_absent_from_chain_writes_empty_track(self):
        (self.work / "in.wig").write_text("chr99\t0\t10\t1.5\nchr99\t20\t30\t2.5\n")
        result = self._run("wig", "in.wig", "out")
        self.assert_finished_track(result, "out", None)
        sorted_bgr = (self.work / "out.sorted.bgr").read_text()
        self.assertEqual(sorted_bgr.strip(), "")

    def test_wig_bedgraph_span_crossing_two_chain_blocks_writes_empty_track(self):
        (self.work / "in.wig").write_text("chr1\t0\t200\t1.5\nchr1\t10\t220\t2.5\n")
        result = self._run("wig", "in.wig", "out")
        self.assert_finished_track(result, "out", None)

    def test_wig_bedgraph_intervals_in_chain_gap_writes_empty_track(self):
        (self.work / "in.wig").write_text("chr1\t100\t150\t1.5\nchr1\t120\t140\t2.5\n")
        result = self._run("wig", "in.wig", "out")
        self.assert_finished_track(result, "out", None)

    def test_wig_variablestep_chromosome_absent_writes_empty_track(self):
        (self.work / "in.wig").write_text(
            "variableStep chrom=chr99 span=10\n1 1.5\n20 2.5\n"
        )
        result = self._run("wig", "in.wig", "out")
        self.assert_finished_track(result, "out", None)

    def test_wig_fixedstep_span_crossing_two_chain_blocks_writes_empty_track(self):
        (self.work / "in.wig").write_text(
            "fixedStep chrom=chr1 start=1 step=50 span=200\n1.5\n"
        )
        result = self._run("wig", "in.wig", "out")
        self.assert_finished_track(result, "out", None)

    def test_bigwig_chromosome_absent_from_chain_writes_empty_track(self):
        write_bigwig(
            self.work / "in.bw",
            "chr99",
            1000,
            [(0, 10, 1.5), (20, 30, 2.5)],
        )
        result = self._run("bigwig", "in.bw", "out")
        self.assert_finished_track(result, "out", None)

    def test_bigwig_span_crossing_two_chain_blocks_writes_empty_track(self):
        # One interval is enough: pyBigWig rejects overlapping input entries,
        # and this span overlaps both chain blocks on chr1.
        write_bigwig(
            self.work / "in.bw",
            "chr1",
            1000,
            [(0, 200, 1.5)],
        )
        result = self._run("bigwig", "in.bw", "out")
        self.assert_finished_track(result, "out", None)

    def test_wig_keeps_interval_that_lifts_to_one_block(self):
        (self.work / "in.wig").write_text(
            "chr99\t0\t10\t9.0\nchr1\t0\t50\t3.5\nchr1\t0\t200\t8.0\n"
        )
        result = self._run("wig", "in.wig", "out")
        self.assert_finished_track(result, "out", ((0, 50, 3.5),))

    def test_bigwig_keeps_interval_that_lifts_to_one_block(self):
        write_bigwig(
            self.work / "in.bw",
            "chr1",
            1000,
            [(10, 40, 4.0)],
        )
        result = self._run("bigwig", "in.bw", "out")
        self.assert_finished_track(result, "out", ((10, 40, 4.0),))


if __name__ == "__main__":
    unittest.main()
