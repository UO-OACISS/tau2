#!/usr/bin/env python3
"""Convert a TAU call-path profile to callgrind format.

The input is one profile.<node>.<context>.<thread> file from a run with
TAU_CALLPATH=1. Each "A => B => C" timer gives the call edge B -> C with C's
call count and inclusive value; flat timers give each routine's exclusive
value, placed on the routine's first line. To see more of each routine in
QCachegrind's source view, raise "Context lines in annotations" in its
settings. Open the output with QCachegrind/KCachegrind, or draw it with
gprof2dot:

  tau_prof2callgrind.py -o callgrind.out.0.0.0 profile.0.0.0
  qcachegrind callgrind.out.0.0.0
  gprof2dot -f callgrind callgrind.out.0.0.0 | dot -Tsvg -o callgraph.svg

With --dot, it runs gprof2dot itself and writes a Graphviz graph with
rounded nodes and a color legend:

  tau_prof2callgrind.py --dot -o callgraph.dot profile.0.0.0
  xdot callgraph.dot
  dot -Tsvg -o callgraph.svg callgraph.dot

Routines are colored by exclusive time, with the largest in red, or by
inclusive time with --inclusive. Calls are always colored by inclusive time.
--node-thres and --edge-thres prune routines and calls below a percentage of
the total time.

Callgrind costs are integers, so values are rounded. Callgrind (and
gprof2dot) show one node per routine, so routines reached by different call
paths are merged. Routines with the same name in different source files get
the location appended to keep them apart.

Other tools (not part of TAU):
  gprof2dot      needed by --dot: uv tool install gprof2dot,
                 pip install gprof2dot, or brew install gprof2dot
  Graphviz       dot, to render images: apt install graphviz or
                 brew install graphviz
  xdot           interactive dot viewer: apt install xdot or brew install xdot
  KCachegrind    callgrind viewer: apt install kcachegrind, or
                 brew install qcachegrind (QCachegrind, the Qt-only version)
"""

import argparse
import colorsys
import re
import subprocess
import sys
from collections import OrderedDict

import tau_prof2json

CALLPATH_SEPARATOR = " => "
# gprof2dot's default "color" colormap: (hue, saturation, lightness) for 0%
# (dark blue) and 100% (red), interpolated linearly.
COLD_HSL = (2.0 / 3.0, 0.8, 0.25)
HOT_HSL = (0.0, 1.0, 0.5)
# A node's self percentage in a gprof2dot label: "name\n100.00%\n(0.08%)\n0×".
SELF_PERCENT_RE = re.compile(r"\\n\(([0-9.]+)%\)\\n")


def _routine(timer_name):
    """Return (name, file, line) for one element of a TAU timer name."""
    name, filename, line, _ = tau_prof2json._split_function(timer_name.strip())
    return name, filename, line


def convert(profile):
    """Return callgrind text for a profile from tau_prof2json.parse_profile."""
    exclusive = OrderedDict()
    edges = OrderedDict()
    for timer in profile["Timers"]:
        path = [_routine(name) for name in timer["Function"].split(CALLPATH_SEPARATOR)]
        for routine in path:
            exclusive.setdefault(routine, 0)
        if len(path) == 1:
            exclusive[path[0]] += timer["Exclusive Time"] or 0
        else:
            # Each call is counted in exactly one path timer, so summing the
            # last edge of every path gives the caller -> callee totals.
            calls, inclusive = edges.get((path[-2], path[-1]), (0, 0))
            edges[(path[-2], path[-1])] = (
                calls + int(timer["Calls"] or 0),
                inclusive + (timer["Inclusive Time"] or 0),
            )
    if not edges:
        sys.stderr.write(
            "warning: no call paths in profile; run with TAU_CALLPATH=1\n"
        )

    name_count = {}
    for name, _, _ in exclusive:
        name_count[name] = name_count.get(name, 0) + 1
    files = OrderedDict()
    functions = OrderedDict()
    for routine in exclusive:
        name, filename, line = routine
        files.setdefault(filename or "???", len(files) + 1)
        if name_count[name] > 1 and filename:
            name = "{} [{}:{}]".format(name, filename, line)
        functions[routine] = (len(functions) + 1, name)

    def file_spec(key, routine):
        filename = routine[1] or "???"
        return "{}=({}) {}".format(key, files[filename], filename)

    def function_spec(key, routine):
        return "{}=({}) {}".format(key, *functions[routine])

    metric = profile["Metric"]
    lines = [
        "# callgrind format",
        "version: 1",
        "creator: tau_prof2callgrind.py",
    ]
    command = profile["Metadata"].get("Command Line")
    if command:
        lines.append("cmd: " + command)
    lines += [
        "thread: {}".format(profile["thread index"] + 1),
        "positions: line",
        "events: " + metric,
    ]
    if metric in tau_prof2json.TIME_METRICS:
        lines.append("event: {} : {} (microseconds)".format(metric, metric))
    for routine, value in exclusive.items():
        caller_line = routine[2] or 0
        lines += [
            "",
            file_spec("fl", routine),
            function_spec("fn", routine),
            "{} {}".format(caller_line, round(value)),
        ]
        for (caller, callee), (calls, inclusive) in edges.items():
            if caller != routine:
                continue
            lines += [
                file_spec("cfl", callee),
                function_spec("cfn", callee),
                "calls={} {}".format(calls, callee[2] or 0),
                "{} {}".format(caller_line, round(inclusive)),
            ]
    return "\n".join(lines) + "\n"


def color(weight):
    """Return gprof2dot's default colormap color for a weight in [0, 1]."""
    hue, saturation, lightness = (
        cold + weight * (hot - cold) for cold, hot in zip(COLD_HSL, HOT_HSL)
    )
    rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
    return "#" + "".join("%02x" % int(255.0 * value + 0.5) for value in rgb)


def style(dot, metric="Inclusive time", scale=1.0):
    """Round the node corners of gprof2dot output and add a color legend.

    The legend maps colors to percentages of the total; scale is the fraction
    of the total that gprof2dot colors red.
    """
    dot = dot.replace("style=filled", 'style="filled,rounded"', 1)
    cells = "".join(
        '<TD BGCOLOR="{}" WIDTH="36"><FONT COLOR="white">{:g}%</FONT></TD>'.format(
            color(percent / 100.0), round(percent * scale, 1)
        )
        for percent in range(0, 101, 10)
    )
    legend = (
        '\tlabelloc=t;\n\tfontsize="9.00";\n\tlabel=<<TABLE BORDER="0" '
        'CELLBORDER="0" CELLSPACING="1" CELLPADDING="3"><TR><TD COLSPAN="11">'
        "{}</TD></TR><TR>{}</TR></TABLE>>;\n"
    ).format(metric, cells)
    end = dot.rstrip().rfind("}")
    return dot[:end] + legend + "}\n"


def gprof2dot(text, node_thres=None, edge_thres=None, exclusive=True):
    """Return the styled gprof2dot graph for callgrind text.

    node_thres and edge_thres are percentages of the total time below which
    gprof2dot drops nodes and edges (None keeps gprof2dot's defaults).
    exclusive colors nodes by exclusive (self) time; False colors them by
    inclusive time.
    """
    command = ["gprof2dot", "-f", "callgrind"]
    if node_thres is not None:
        command += ["--node-thres", str(node_thres)]
    if edge_thres is not None:
        command += ["--edge-thres", str(edge_thres)]
    if exclusive:
        command.append("--color-nodes-by-selftime")
    try:
        result = subprocess.run(
            command,
            input=text,
            stdout=subprocess.PIPE,
            universal_newlines=True,
            check=True,
        )
    except FileNotFoundError as error:
        raise OSError(
            "gprof2dot not found; install it with 'uv tool install gprof2dot' "
            "or 'brew install gprof2dot'"
        ) from error
    except subprocess.CalledProcessError as error:
        raise OSError("gprof2dot failed with status {}".format(error.returncode))
    if not exclusive:
        return style(result.stdout)
    # gprof2dot colors the largest exclusive time red, so scale the legend to
    # the largest "(self%)" in the node labels.
    largest = max(
        (float(value) for value in SELF_PERCENT_RE.findall(result.stdout)),
        default=100.0,
    )
    return style(result.stdout, "Exclusive time", largest / 100.0)


def main(arguments=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("profile", help="profile.<node>.<context>.<thread> file")
    parser.add_argument(
        "-o", "--output", help="output file (default: standard output)"
    )
    parser.add_argument(
        "--dot",
        action="store_true",
        help="write a Graphviz call graph made with gprof2dot instead",
    )
    parser.add_argument(
        "-n",
        "--node-thres",
        type=float,
        metavar="PERCENTAGE",
        help="with --dot, drop routines below this percentage of the total "
        "inclusive time (gprof2dot default: 0.5)",
    )
    parser.add_argument(
        "-e",
        "--edge-thres",
        type=float,
        metavar="PERCENTAGE",
        help="with --dot, drop calls below this percentage of the total time "
        "(gprof2dot default: 0.1)",
    )
    color_by = parser.add_mutually_exclusive_group()
    color_by.add_argument(
        "--exclusive",
        dest="exclusive",
        action="store_const",
        const=True,
        help="with --dot, color routines by exclusive (self) time (default)",
    )
    color_by.add_argument(
        "--inclusive",
        dest="exclusive",
        action="store_const",
        const=False,
        help="with --dot, color routines by inclusive time",
    )
    args = parser.parse_args(arguments)
    if not args.dot and (
        args.node_thres is not None
        or args.edge_thres is not None
        or args.exclusive is not None
    ):
        parser.error(
            "--node-thres, --edge-thres, --exclusive, and --inclusive need --dot"
        )
    try:
        text = convert(tau_prof2json.parse_profile(args.profile))
        if args.dot:
            text = gprof2dot(
                text, args.node_thres, args.edge_thres, args.exclusive is not False
            )
    except (OSError, ValueError) as error:
        sys.stderr.write("tau_prof2callgrind.py: {}\n".format(error))
        return 1
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output:
            output.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
