#!/usr/bin/env python3

import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO

TOOLS_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TOOLS_SRC)

import tau_prof2callgrind
import tau_prof2json


HEADER = """{} templated_functions_MULTI_TIME
# Name Calls Subrs Excl Incl ProfileCalls # <metadata><attribute><name>Metric Name</name><value>TIME</value></attribute><attribute><name>Command Line</name><value>./matmult</value></attribute></metadata>
"""

# From the examples/salt-fm/serial/f90 run with TAU_CALLPATH=1.
CALLPATH_TIMERS = """".TAU application" 1 1 371 442982 0 GROUP="TAU_UTILITY"
"MATMULT [{matmult.f90} {22,1}-{35,19}]" 1 2 4176 442611 0 GROUP="TAU_DEFAULT"
".TAU application => MATMULT [{matmult.f90} {22,1}-{35,19}]" 1 2 4176 442611 0 GROUP="TAU_CALLPATH|TAU_DEFAULT"
"compute [{matmult_impl.f90} {16,3}-{26,23}]" 1 0 337940 337940 0 GROUP="TAU_DEFAULT"
".TAU application => MATMULT [{matmult.f90} {22,1}-{35,19}] => compute [{matmult_impl.f90} {16,3}-{26,23}]" 1 0 337940 337940 0 GROUP="TAU_CALLPATH|TAU_DEFAULT"
"compute_interchange [{matmult_impl.f90} {28,3}-{38,35}]" 1 0 100495 100495 0 GROUP="TAU_DEFAULT"
".TAU application => MATMULT [{matmult.f90} {22,1}-{35,19}] => compute_interchange [{matmult_impl.f90} {28,3}-{38,35}]" 1 0 100495 100495 0 GROUP="TAU_CALLPATH|TAU_DEFAULT"
0 aggregates
"""

# init in two modules, called twice from main by two call paths.
SAME_NAME_TIMERS = """"main" 1 3 10 110 0 GROUP="TAU_DEFAULT"
"init [{a.f90} {3,1}-{9,1}]" 2 0 40 40 0 GROUP="TAU_DEFAULT"
"init [{b.f90} {5,1}-{8,1}]" 1 0 60 60 0 GROUP="TAU_DEFAULT"
"main => init [{a.f90} {3,1}-{9,1}]" 2 0 40 40 0 GROUP="TAU_CALLPATH"
"main => init [{b.f90} {5,1}-{8,1}]" 1 0 60 60 0 GROUP="TAU_CALLPATH"
0 aggregates
"""

FLAT_TIMERS = """"main" 1 0 10 10 0 GROUP="TAU_DEFAULT"
0 aggregates
"""


class ConvertTest(unittest.TestCase):
    def convert(self, timers):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "profile.0.0.0")
            with open(path, "w", encoding="utf-8") as profile:
                profile.write(HEADER.format(timers.count("\n") - 1) + timers)
            stderr = StringIO()
            with redirect_stderr(stderr):
                text = tau_prof2callgrind.convert(tau_prof2json.parse_profile(path))
        return text, stderr.getvalue()

    def test_callpath_edges(self):
        text, stderr = self.convert(CALLPATH_TIMERS)
        self.assertEqual(stderr, "")
        self.assertIn("cmd: ./matmult\n", text)
        self.assertIn("events: TIME\n", text)
        self.assertIn(
            "fl=(2) matmult.f90\nfn=(2) MATMULT\n22 4176\n"
            "cfl=(3) matmult_impl.f90\ncfn=(3) compute\ncalls=1 16\n22 337940\n"
            "cfl=(3) matmult_impl.f90\ncfn=(4) compute_interchange\n"
            "calls=1 28\n22 100495\n",
            text,
        )
        self.assertIn("fl=(1) ???\nfn=(1) .TAU application\n0 371\n", text)

    def test_same_name_in_two_files(self):
        text, _ = self.convert(SAME_NAME_TIMERS)
        self.assertIn("cfn=(2) init [a.f90:3]\ncalls=2 3\n0 40\n", text)
        self.assertIn("cfn=(3) init [b.f90:5]\ncalls=1 5\n0 60\n", text)

    def test_flat_profile_warns(self):
        text, stderr = self.convert(FLAT_TIMERS)
        self.assertIn("TAU_CALLPATH=1", stderr)
        self.assertIn("fn=(1) main\n0 10\n", text)


class DotTest(unittest.TestCase):
    def test_color_matches_gprof2dot(self):
        # Colors gprof2dot 2025.4.14 gave the CALLPATH_TIMERS nodes.
        total = 442982
        self.assertEqual(tau_prof2callgrind.color(442611 / total), "#ff0100")
        self.assertEqual(tau_prof2callgrind.color(337940 / total), "#dbd005")
        self.assertEqual(tau_prof2callgrind.color(100495 / total), "#0c8490")

    def test_style(self):
        dot = (
            "digraph {\n"
            "\tnode [shape=box, style=filled, width=0];\n"
            '\ta [color="#ff0000", label="a"];\n'
            "}\n"
        )
        styled = tau_prof2callgrind.style(dot)
        self.assertIn('style="filled,rounded"', styled)
        self.assertIn("labelloc=t;", styled)
        self.assertIn('<TD BGCOLOR="#ff0000" WIDTH="36">', styled)
        self.assertTrue(styled.endswith(">;\n}\n"))

    def dot(self, options):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "profile.0.0.0")
            with open(path, "w", encoding="utf-8") as profile:
                profile.write(HEADER.format(7) + CALLPATH_TIMERS)
            output = os.path.join(directory, "callgraph.dot")
            status = tau_prof2callgrind.main(["--dot", "-o", output] + options + [path])
            with open(output, encoding="utf-8") as dot:
                text = dot.read()
        self.assertEqual(status, 0)
        return text

    @unittest.skipIf(shutil.which("gprof2dot") is None, "gprof2dot not installed")
    def test_prune_and_exclusive_default(self):
        text = self.dot(["-n", "50"])
        # compute_interchange is 22.69% of the total, below the 50% cutoff.
        self.assertIn("compute", text)
        self.assertNotIn("compute_interchange", text)
        self.assertIn("Exclusive time", text)
        # gprof2dot colors the largest exclusive time (compute, 76.29%) red
        # and MATMULT (0.94% exclusive) relative to it; the legend's top cell
        # is the 76.29%.
        self.assertIn('compute [color="#ff0000"', text)
        self.assertIn(
            'MATMULT [color="{}"'.format(tau_prof2callgrind.color(0.94 / 76.29)),
            text,
        )
        self.assertIn('<FONT COLOR="white">76.3%</FONT>', text)

    @unittest.skipIf(shutil.which("gprof2dot") is None, "gprof2dot not installed")
    def test_inclusive(self):
        text = self.dot(["--inclusive"])
        self.assertIn("Inclusive time", text)
        self.assertIn('MATMULT [color="#ff0100"', text)
        self.assertIn('<FONT COLOR="white">100%</FONT>', text)

    def test_options_need_dot(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            tau_prof2callgrind.main(["--inclusive", "profile.0.0.0"])

    def test_exclusive_and_inclusive_conflict(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            tau_prof2callgrind.main(
                ["--dot", "--exclusive", "--inclusive", "profile.0.0.0"]
            )


if __name__ == "__main__":
    unittest.main()
