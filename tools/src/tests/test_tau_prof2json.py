#!/usr/bin/env python3

import argparse
import json
import html
import os
import re
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

try:
    import jsonschema
except ImportError:
    jsonschema = None


TOOLS_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(TOOLS_SRC)))
sys.path.insert(0, TOOLS_SRC)

import tau_prof2json


PROFILE = """2 templated_functions_MULTI_TIME
# Name Calls Subrs Excl Incl ProfileCalls # <metadata><attribute><name>Metric Name</name><value>TIME</value></attribute><attribute><name>Executable</name><value>demo</value></attribute><attribute><name>Starting Timestamp</name><value>1000</value></attribute><attribute><name>Ending Timestamp</name><value>2000</value></attribute><attribute><name>Local Time</name><value>2026-01-01T00:00:00Z</value></attribute><attribute><name>pid</name><value>7</value></attribute></metadata>
"main" 1 2 3.5 4.5 6 GROUP="TAU_USER"
"MPI_Send()" 2 0 7 8 0 GROUP="MPI"
1 aggregates
"timer aggregate" 3 0 9 10 0 GROUP="TAU_DEFAULT"
3 userevents
# eventname numevents max min mean sumsqr
"Message size sent to all nodes" 2 8 4 6 80
"Message size received from all nodes" 3 5 2 4 60
"quoted event" 1 3 3 3 9
future-section-data
"""


def _profile_filenames(profile_dir):
    pattern = re.compile(r"^profile\.(\d+)\.(\d+)\.(\d+)$")
    filenames = [
        filename
        for filename in os.listdir(profile_dir)
        if pattern.fullmatch(filename)
        and os.path.isfile(os.path.join(profile_dir, filename))
    ]
    return sorted(
        filenames,
        key=lambda name: tuple(map(int, pattern.fullmatch(name).groups())),
    )


def _reference_record(line, dimensions, event=False):
    match = re.fullmatch(r'"(.*?)"\s+(.*?)(?:\s+GROUP="(.*)")?\s*', line)
    if not match:
        raise AssertionError("Unrecognized TAU record: {!r}".format(line))
    name, value_text, group = match.groups()
    values = [float(value) for value in value_text.split()]
    result = dict(dimensions)
    if event:
        result["Counter"] = name
        result.update(
            zip(
                ("Num Events", "Max Value", "Min Value", "Mean Value", "SumSqr Value"),
                values,
            )
        )
    else:
        result["Function"] = name
        result.update(
            zip(
                ("Calls", "Subroutines", "Exclusive Time", "Inclusive Time"),
                values[:4],
            )
        )
        if len(values) > 4:
            result["Profile Calls"] = values[4]
        if group is not None:
            result["Group"] = group
        if len(values) > 5:
            result["Additional Values"] = values[5:]
    return result


def _reference_profile(path):
    """Read source records independently of the converter's parsing code."""
    basename = os.path.basename(path)
    _, node, context, thread = basename.split(".")
    with open(path, encoding="utf-8", errors="replace") as profile_file:
        lines = profile_file.read().splitlines()
    header_tokens = lines[0].split()
    timer_count = int(header_tokens[0])
    profile_format = " ".join(header_tokens[1:])
    metric = "TIME" if profile_format == "templated_functions" else profile_format
    format_metric = re.fullmatch(
        r"templated_functions(?:_MULTI)?_(.+)", profile_format
    )
    if format_metric:
        metric = format_metric.group(1)
    metadata = {
        html.unescape(name): html.unescape(value)
        for name, value in re.findall(
            r"<attribute><name>(.*?)</name><value>(.*?)</value></attribute>",
            lines[1],
        )
    }
    metric = metadata.get("Metric Name", metric)
    dimensions = {
        "process index": int(node),
        "context index": int(context),
        "thread index": int(thread),
        "Metric": metric,
    }
    index = 2
    timers = [
        _reference_record(lines[index + offset], dimensions)
        for offset in range(timer_count)
    ]
    index += timer_count
    aggregate_count = int(lines[index].split()[0])
    index += 1
    aggregates = [
        _reference_record(lines[index + offset], dimensions)
        for offset in range(aggregate_count)
    ]
    index += aggregate_count
    counters = []
    if index < len(lines):
        event_count = int(lines[index].split()[0])
        index += 1
        if index < len(lines) and lines[index].startswith("#"):
            index += 1
        counters = [
            _reference_record(lines[index + offset], dimensions, event=True)
            for offset in range(event_count)
        ]
        index += event_count
    return {
        "Filename": basename,
        **dimensions,
        "Profile Format": profile_format,
        "Metadata": metadata,
        "Timers": timers,
        "Aggregates": aggregates,
        "Counters": counters,
        "Additional Profile Data": lines[index:],
    }


def _profile_sources(profile_dir):
    """(display name, path) of every source profile, MULTI__ metrics included."""
    multi = sorted(
        name
        for name in os.listdir(profile_dir)
        if name.startswith("MULTI_")
        and os.path.isdir(os.path.join(profile_dir, name))
    )
    if not multi:
        return [
            (name, os.path.join(profile_dir, name))
            for name in _profile_filenames(profile_dir)
        ]
    if _profile_filenames(profile_dir):
        raise AssertionError(
            "{} mixes profile.* files and MULTI__ directories".format(profile_dir)
        )
    return [
        (os.path.join(directory, name), os.path.join(profile_dir, directory, name))
        for directory in multi
        for name in _profile_filenames(os.path.join(profile_dir, directory))
    ]


def _primary_metric(expected_profiles):
    metrics = list(dict.fromkeys(p["Metric"] for p in expected_profiles))
    time_metrics = {"TIME", "P_WALL_CLOCK_TIME", "P_VIRTUAL_TIME"}
    return next((m for m in metrics if m.upper() in time_metrics), metrics[0])


def _expected_counters(expected_profiles):
    """User events are metric independent: kept only on the time metric."""
    primary = _primary_metric(expected_profiles)
    return [
        record
        for profile in expected_profiles
        if profile["Metric"] == primary
        for record in profile["Counters"]
    ]


def validate_external_directory(profile_dir):
    profile_dir = os.path.abspath(profile_dir)
    sources = _profile_sources(profile_dir)
    filenames = [name for name, _ in sources]
    if not filenames:
        raise AssertionError(
            "{} contains no profile.<node>.<context>.<thread> files".format(
                profile_dir
            )
        )

    with tempfile.TemporaryDirectory() as temporary_directory:
        output_path = os.path.join(temporary_directory, "profiles.json")
        subprocess.run(
            [
                sys.executable,
                os.path.join(TOOLS_SRC, "tau_prof2json.py"),
                "-o",
                output_path,
                profile_dir,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        with open(output_path, encoding="utf-8") as output_file:
            result = json.load(output_file)

        schema_path = os.path.join(TOOLS_SRC, "tau_profile_json.schema.json")
        if jsonschema is not None:
            with open(schema_path, encoding="utf-8") as schema_file:
                jsonschema.validate(result, json.load(schema_file))

        compact_path = os.path.join(temporary_directory, "compact.json")
        subprocess.run(
            [
                sys.executable,
                os.path.join(TOOLS_SRC, "tau_prof2json.py"),
                "--compact",
                "-o",
                compact_path,
                profile_dir,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        with open(compact_path, encoding="utf-8") as compact_file:
            compact = json.load(compact_file)
        if jsonschema is not None:
            compact_schema = os.path.join(
                TOOLS_SRC, "tau_profile_json_compact.schema.json"
            )
            with open(compact_schema, encoding="utf-8") as schema_file:
                jsonschema.validate(compact, json.load(schema_file))
        if tau_prof2json.expand_compact(compact) != result:
            raise AssertionError("compact JSON does not expand to the full JSON")

        application = result[profile_dir]
    expected_profiles = [_reference_profile(path) for _, path in sources]
    primary = _primary_metric(expected_profiles)
    for entry, expected in zip(application["Profiles"], expected_profiles):
        if entry["Filename"] != next(n for n, _ in sources if n == entry["Filename"]):
            raise AssertionError("unexpected file {}".format(entry["Filename"]))
        for key in ("process index", "context index", "thread index", "Metric"):
            if entry[key] != expected[key]:
                raise AssertionError("JSON {} differs for {}".format(key, entry["Filename"]))
        for section in ("Timer", "Aggregate", "Counter"):
            count = len(expected[section + "s"])
            if section == "Counter" and expected["Metric"] != primary:
                count = 0
            if entry[section + " Count"] != count:
                raise AssertionError("JSON {} count differs".format(section))
    for section in ("Timers", "Aggregates", "Counters"):
        if section == "Counters":
            expected_records = _expected_counters(expected_profiles)
        else:
            expected_records = [
                record
                for profile in expected_profiles
                for record in profile[section]
            ]
        if application[section] != expected_records:
            raise AssertionError(
                "JSON {} list differs from source profiles".format(section)
            )
    if len(application["Profiles"]) != len(filenames):
        raise AssertionError("JSON did not include every source profile")
    expected_metrics = {profile["Metric"] for profile in expected_profiles}
    if set(application["Metrics"]) != expected_metrics:
        raise AssertionError("JSON metric list differs from source profiles")
    return (
        len(filenames),
        sum(len(profile["Timers"]) for profile in expected_profiles),
        sum(len(profile["Aggregates"]) for profile in expected_profiles),
        len(_expected_counters(expected_profiles)),
    )


class TauProfileToJsonTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.profile_dir = os.path.join(self.temp_dir.name, "profiles")
        os.mkdir(self.profile_dir)
        self.profile_path = os.path.join(self.profile_dir, "profile.2.1.3")
        with open(self.profile_path, "w", encoding="utf-8") as profile:
            profile.write(PROFILE)

    def test_profile_sections_and_dimensions_are_preserved(self):
        result = tau_prof2json.parse_profile(self.profile_path)

        self.assertEqual((2, 1, 3), (
            result["process index"],
            result["context index"],
            result["thread index"],
        ))
        self.assertEqual("TIME", result["Metric"])
        self.assertEqual("demo", result["Metadata"]["Executable"])
        self.assertEqual("TIME", result["Timers"][0]["Metric"])
        self.assertEqual(3.5, result["Timers"][0]["Exclusive Time"])
        self.assertEqual(6, result["Timers"][0]["Profile Calls"])
        self.assertEqual("TAU_USER", result["Timers"][0]["Group"])
        self.assertEqual("timer aggregate", result["Aggregates"][0]["Function"])
        self.assertEqual(80, result["Counters"][0]["SumSqr Value"])
        self.assertIsNone(tau_prof2json._number("nan", self.profile_path, 1))
        self.assertEqual(["future-section-data"], result["Additional Profile Data"])

    def test_profile_without_user_events_is_supported(self):
        with open(self.profile_path, "w", encoding="utf-8") as profile:
            profile.write(
                "\n".join(
                    (
                        "1 templated_functions_MULTI_TIME",
                        "# Name Calls Subrs Excl Incl ProfileCalls",
                        '"main" 1 0 2 3 0 GROUP="TAU_USER"',
                        "0 aggregates",
                    )
                )
                + "\n"
            )

        result = tau_prof2json.parse_profile(self.profile_path)
        self.assertEqual([], result["Counters"])
        self.assertEqual([], result["Aggregates"])

    def test_directory_conversion_emits_valid_json_to_stdout(self):
        output = StringIO()
        errors = StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            tau_prof2json.main([self.profile_dir])

        result = json.loads(output.getvalue())
        application = result[self.profile_dir]
        self.assertEqual(["TIME"], application["Metrics"])
        self.assertEqual(2, application["Timers"][0]["process index"])
        component = result["Workflow metadata"]["Workflow Component"][0]
        self.assertEqual(3.5, component["aggr_execution_time"])
        self.assertEqual(12, component["aggr_communication_sent_bytes"])
        self.assertEqual(12, component["aggr_communication_recv_bytes"])
        self.assertIn("Processing:", errors.getvalue())

    def test_compact_layout_is_lossless(self):
        output = StringIO()
        with redirect_stdout(output), redirect_stderr(StringIO()):
            tau_prof2json.main([self.profile_dir])
        full = json.loads(output.getvalue())
        compact = json.loads(json.dumps(tau_prof2json.compact_output(full)))
        application = compact[self.profile_dir]
        self.assertEqual(
            "1.0-compact", compact["Workflow metadata"]["profile-json-version"]
        )
        self.assertLessEqual(
            {"main", "MPI_Send()", "TAU_USER", "MPI"}, set(application["strings"])
        )
        self.assertEqual(["Function", "Group"], application["Timers"][0]["indexed"])
        self.assertEqual(full, tau_prof2json.expand_compact(compact))

    def test_compact_tables_split_when_record_keys_differ(self):
        records = [{"a": 1, "b": 2}, {"a": 1}, {"a": 3}]
        tables = tau_prof2json._encode_tables(records, set(), {})
        self.assertEqual(2, len(tables))
        self.assertEqual(records, tau_prof2json._decode_tables(tables, []))

    def test_quotes_in_names_are_preserved(self):
        for line, name, group in (
            ('"say "hi" [{f.c} {1,2}]" 3 0 1 2 0 GROUP="TAU_USER"', 'say "hi" [{f.c} {1,2}]', "TAU_USER"),
            ('"a" b" 1 0 2 3 0 GROUP="TAU_PARAM | TAU_USER" ', 'a" b', "TAU_PARAM | TAU_USER"),
            ('"event "x"" 4 9 1 5 30', 'event "x"', None),
        ):
            parsed = tau_prof2json._quoted_record(line, "p", 1)
            self.assertEqual((name, group), (parsed[0], parsed[2]))

    def test_timer_categories_use_groups_and_adios_name(self):
        def timer(name, group):
            return {"Function": name, "Group": group}

        category = tau_prof2json._timer_category
        self.assertEqual("MPI", category(timer("MPI_Send()", "MPI")))
        self.assertEqual("IO", category(timer("read()", "TAU_IO")))
        self.assertEqual("ADIOS", category(timer("adios_open", "TAU_IO")))
        self.assertEqual("USER", category(timer("f", "TAU_PARAM | TAU_USER")))
        self.assertEqual("OTHER", category(timer("g", "OpenMP")))
        self.assertEqual("OTHER", category(timer("MPI_not_a_group", "TAU_MPI_X")))

    def test_summary_is_precomputed_and_schema_valid(self):
        for extra in ([], ["--compact"]):
            output = StringIO()
            with redirect_stdout(output), redirect_stderr(StringIO()):
                tau_prof2json.main(["--summary"] + extra + [self.profile_dir])
            result = json.loads(output.getvalue())
            summary = result[self.profile_dir]["Summary"]
            self.assertEqual((1, 1), (summary["processes"], summary["threads"]))
            metric = summary["by_metric"]["TIME"]
            self.assertEqual("main", metric["top_exclusive"][1]["Function"])
            self.assertAlmostEqual(7 / 10.5, metric["exclusive_fraction"]["MPI"], 5)
            if jsonschema is not None:
                name = "tau_profile_json%s.schema.json" % (
                    "_compact" if extra else ""
                )
                with open(os.path.join(TOOLS_SRC, name), encoding="utf-8") as f:
                    jsonschema.validate(result, json.load(f))
        without = StringIO()
        with redirect_stdout(without), redirect_stderr(StringIO()):
            tau_prof2json.main([self.profile_dir])
        self.assertNotIn("Summary", json.loads(without.getvalue())[self.profile_dir])

    def test_hardware_counter_is_not_reported_as_execution_time(self):
        with open(self.profile_path, "w", encoding="utf-8") as profile:
            profile.write(PROFILE.replace("<value>TIME</value>", "<value>CPU_CYCLES</value>"))

        output = StringIO()
        with redirect_stdout(output), redirect_stderr(StringIO()):
            tau_prof2json.main([self.profile_dir])

        result = json.loads(output.getvalue())
        component = result["Workflow metadata"]["Workflow Component"][0]
        self.assertNotIn("aggr_execution_time", component)

    def test_workflow_instance_list_and_component_names_are_preserved(self):
        workflow_path = os.path.join(self.temp_dir.name, "workflow.json")
        with open(workflow_path, "w", encoding="utf-8") as workflow_file:
            json.dump(
                {
                    "Workflow Instance": [{"id": 9, "name": "existing-workflow"}],
                    "Workflow Component": [{"id": 8, "name": "profiles"}],
                },
                workflow_file,
            )

        output = StringIO()
        with redirect_stdout(output), redirect_stderr(StringIO()):
            tau_prof2json.main(["-w", workflow_path, self.profile_dir])

        result = json.loads(output.getvalue())
        workflow = result["Workflow metadata"]
        self.assertEqual(1, len(workflow["Workflow Component"]))
        self.assertEqual("profiles", workflow["Workflow Component"][0]["name"])
        self.assertEqual("existing-workflow", workflow["Workflow Instance"][0]["name"])
        self.assertEqual(1000, workflow["Workflow Instance"][0]["execution_time"])

    def _write_multi(self, root, metrics, events=None):
        """Write MULTI__<metric> directories holding one profile.0.0.0 each."""
        for metric in metrics:
            directory = os.path.join(root, "MULTI__" + metric)
            os.makedirs(directory)
            text = PROFILE.replace("<value>TIME</value>", "<value>%s</value>" % metric)
            text = text.replace("templated_functions_MULTI_TIME",
                                "templated_functions_MULTI_" + metric)
            if events and metric in events:
                text = text.replace('"quoted event" 1 3 3 3 9', events[metric])
            with open(os.path.join(directory, "profile.0.0.0"), "w", encoding="utf-8") as f:
                f.write(text)

    def _convert(self, *arguments):
        output = StringIO()
        with redirect_stdout(output), redirect_stderr(StringIO()):
            tau_prof2json.main(list(arguments))
        return json.loads(output.getvalue())

    def test_multi_metric_directories_are_merged(self):
        root = os.path.join(self.temp_dir.name, "multi")
        os.mkdir(root)
        self._write_multi(root, ["TIME", "PAPI_FP_OPS"])

        result = self._convert("--summary", root)
        application = result[root]
        self.assertEqual(["PAPI_FP_OPS", "TIME"], application["Metrics"])
        self.assertEqual(
            ["MULTI__PAPI_FP_OPS/profile.0.0.0", "MULTI__TIME/profile.0.0.0"],
            [entry["Filename"] for entry in application["Profiles"]],
        )
        self.assertEqual(
            {"PAPI_FP_OPS", "TIME"}, {t["Metric"] for t in application["Timers"]}
        )
        self.assertEqual(4, len(application["Timers"]))
        # User events are identical across metrics, so stored once.
        self.assertEqual(3, len(application["Counters"]))
        self.assertEqual([0, 3], [
            e["Counter Count"] for e in application["Profiles"]
        ])
        self.assertEqual({"TIME"}, {c["Metric"] for c in application["Counters"]})
        self.assertEqual(1, application["Summary"]["threads"])
        self.assertEqual(
            ["PAPI_FP_OPS", "TIME"], list(application["Summary"]["by_metric"])
        )
        workflow = result["Workflow metadata"]
        component = workflow["Workflow Component"][0]
        self.assertEqual(1, component["Threads"])
        self.assertEqual(12, component["aggr_communication_sent_bytes"])
        self.assertEqual(3.5, component["aggr_execution_time"])
        self.assertEqual(2, len(workflow["Metrics"]))

    def test_multi_metric_compact_layout_is_lossless(self):
        root = os.path.join(self.temp_dir.name, "multi")
        os.mkdir(root)
        self._write_multi(root, ["TIME", "PAPI_FP_OPS"])
        full = self._convert(root)
        compact = json.loads(json.dumps(tau_prof2json.compact_output(full)))
        self.assertEqual(full, tau_prof2json.expand_compact(compact))
        if jsonschema is not None:
            for name, document in (
                ("tau_profile_json.schema.json", full),
                ("tau_profile_json_compact.schema.json", compact),
            ):
                with open(os.path.join(TOOLS_SRC, name), encoding="utf-8") as f:
                    jsonschema.validate(document, json.load(f))

    def test_differing_user_events_use_time_metric_with_warning(self):
        root = os.path.join(self.temp_dir.name, "multi")
        os.mkdir(root)
        self._write_multi(
            root, ["TIME", "PAPI_FP_OPS"],
            events={"PAPI_FP_OPS": '"quoted event" 9 3 3 3 9'},
        )
        errors = StringIO()
        output = StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            tau_prof2json.main([root])
        self.assertIn("user events differ", errors.getvalue())
        self.assertEqual(3, len(json.loads(output.getvalue())[root]["Counters"]))

    def test_multi_directories_with_root_profiles_is_an_error(self):
        root = os.path.join(self.temp_dir.name, "mixed")
        os.mkdir(root)
        self._write_multi(root, ["TIME"])
        with open(os.path.join(root, "profile.0.0.0"), "w", encoding="utf-8") as f:
            f.write(PROFILE)
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                tau_prof2json.main([root])

    def test_single_multi_directory_can_be_targeted_directly(self):
        root = os.path.join(self.temp_dir.name, "multi")
        os.mkdir(root)
        self._write_multi(root, ["TIME", "PAPI_FP_OPS"])
        target = os.path.join(root, "MULTI__TIME")
        self.assertEqual(["TIME"], self._convert(target)[target]["Metrics"])

    @unittest.skipIf(jsonschema is None, "jsonschema is not installed")
    def test_output_matches_documented_schema(self):
        output = StringIO()
        with redirect_stdout(output), redirect_stderr(StringIO()):
            tau_prof2json.main([self.profile_dir])
        schema_path = os.path.join(TOOLS_SRC, "tau_profile_json.schema.json")
        with open(schema_path, encoding="utf-8") as schema_file:
            schema = json.load(schema_file)
        jsonschema.validate(json.loads(output.getvalue()), schema)


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--profile-dir",
        help="Also validate an external directory of raw TAU profile.* files "
        "or MULTI__<metric> directories",
    )
    options = argument_parser.parse_args()
    suite_result = unittest.main(argv=[sys.argv[0]], exit=False).result
    if suite_result.wasSuccessful() and options.profile_dir:
        try:
            profile_count, timer_count, aggregate_count, counter_count = (
                validate_external_directory(options.profile_dir)
            )
        except (AssertionError, OSError, ValueError, subprocess.CalledProcessError) as error:
            print("External profile validation failed: {}".format(error), file=sys.stderr)
            sys.exit(1)
        print(
            "Validated {} profiles, {} timers, {} aggregates, and {} user events "
            "from {}".format(
                profile_count,
                timer_count,
                aggregate_count,
                counter_count,
                os.path.abspath(options.profile_dir),
            )
        )
    sys.exit(0 if suite_result.wasSuccessful() else 1)
