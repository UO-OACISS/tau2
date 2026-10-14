#!/usr/bin/env python3
"""Convert TAU text profile directories to JSON."""

import argparse
import html
import json
import math
import os
import re
import sys
from collections import OrderedDict


METADATA_RE = re.compile(r"<metadata>(.*?)</metadata>", re.DOTALL)
ATTRIBUTE_RE = re.compile(
    r"<attribute>\s*<name>(.*?)</name>\s*<value>(.*?)</value>\s*</attribute>",
    re.DOTALL,
)
PROFILE_NAME_RE = re.compile(r"^profile\.(\d+)\.(\d+)\.(\d+)$")
TIME_METRICS = {"TIME", "P_WALL_CLOCK_TIME", "P_VIRTUAL_TIME"}


def parse_tau_xml(instring):
    """Parse the metadata fragment embedded in a TAU profile header."""
    metadata = OrderedDict()
    match = METADATA_RE.search(instring)
    if not match:
        return metadata
    for attribute in ATTRIBUTE_RE.finditer(match.group(1)):
        name = html.unescape(attribute.group(1))
        value = html.unescape(attribute.group(2))
        metadata[name] = value
    return metadata


def _number(value, path, line_number):
    try:
        if re.fullmatch(r"[+-]?\d+", value):
            return int(value)
        number = float(value)
        return number if math.isfinite(number) else None
    except ValueError as error:
        raise ValueError(
            "{}:{}: invalid numeric value {!r}".format(path, line_number, value)
        ) from error


def _quoted_record(line, path, line_number):
    line = line.strip()
    if not line.startswith('"'):
        raise ValueError("{}:{}: expected a quoted record".format(path, line_number))
    end_quote = line.find('"', 1)
    if end_quote < 0:
        raise ValueError("{}:{}: unterminated quoted record".format(path, line_number))
    name = line[1:end_quote]
    remainder = line[end_quote + 1 :].strip()
    group = None
    group_match = re.search(r'\s+GROUP="(.*)"\s*$', remainder)
    if group_match:
        group = group_match.group(1)
        remainder = remainder[: group_match.start()].strip()
    return name, remainder.split(), group


def _parse_timer(line, dimensions, path, line_number):
    name, values, group = _quoted_record(line, path, line_number)
    if len(values) < 4:
        raise ValueError(
            "{}:{}: timer record requires calls, subroutines, exclusive and "
            "inclusive values".format(path, line_number)
        )
    result = OrderedDict(dimensions)
    result["Function"] = name
    labels = ("Calls", "Subroutines", "Exclusive Time", "Inclusive Time")
    for label, value in zip(labels, values):
        result[label] = _number(value, path, line_number)
    if len(values) >= 5:
        result["Profile Calls"] = _number(values[4], path, line_number)
    if group is not None:
        result["Group"] = group
    if len(values) > 5:
        result["Additional Values"] = [
            _number(value, path, line_number) for value in values[5:]
        ]
    return result


def _parse_user_event(line, dimensions, path, line_number):
    name, values, _ = _quoted_record(line, path, line_number)
    if len(values) < 5:
        raise ValueError(
            "{}:{}: user event requires event count, max, min, mean and "
            "sum-of-squares".format(path, line_number)
        )
    result = OrderedDict(dimensions)
    result["Counter"] = name
    labels = ("Num Events", "Max Value", "Min Value", "Mean Value", "SumSqr Value")
    for label, value in zip(labels, values):
        result[label] = _number(value, path, line_number)
    if len(values) > 5:
        result["Additional Values"] = [
            _number(value, path, line_number) for value in values[5:]
        ]
    return result


def _next_nonempty(lines, index):
    while index < len(lines) and not lines[index].strip():
        index += 1
    return index


def _section_count(line, section, path, line_number):
    match = re.fullmatch(r"\s*(\d+)\s+{}".format(re.escape(section)), line.strip())
    if not match:
        raise ValueError(
            "{}:{}: expected '<count> {}' section header".format(
                path, line_number, section
            )
        )
    return int(match.group(1))


def _read_records(lines, index, count, parser, dimensions, path, section):
    records = []
    for _ in range(count):
        index = _next_nonempty(lines, index)
        if index >= len(lines):
            raise ValueError(
                "{}: unexpected end of file in {} section".format(path, section)
            )
        if lines[index].lstrip().startswith("#"):
            index += 1
            index = _next_nonempty(lines, index)
            if index >= len(lines):
                raise ValueError(
                    "{}: unexpected end of file in {} section".format(path, section)
                )
        records.append(parser(lines[index], dimensions, path, index + 1))
        index += 1
    return records, index


def parse_profile(path):
    """Read all timer, aggregate, user-event, and metadata records in a profile."""
    basename = os.path.basename(path)
    name_match = PROFILE_NAME_RE.fullmatch(basename)
    if not name_match:
        raise ValueError(
            "{}: expected profile.<node>.<context>.<thread> filename".format(path)
        )
    node, context, thread = (int(value) for value in name_match.groups())
    dimensions = OrderedDict(
        (
            ("process index", node),
            ("context index", context),
            ("thread index", thread),
        )
    )

    with open(path, encoding="utf-8", errors="replace") as profile:
        lines = profile.readlines()
    if len(lines) < 2:
        raise ValueError("{}: incomplete TAU profile header".format(path))

    first_line = lines[0].split()
    if len(first_line) < 2:
        raise ValueError("{}: invalid function-count header".format(path))
    try:
        timer_count = int(first_line[0])
    except ValueError as error:
        raise ValueError("{}: invalid function count".format(path)) from error
    if timer_count < 0:
        raise ValueError("{}: function count cannot be negative".format(path))
    profile_format = " ".join(first_line[1:])
    metric = "TIME" if profile_format == "templated_functions" else profile_format
    metric_match = re.search(r"templated_functions(?:_MULTI)?_(.+)$", profile_format)
    if metric_match:
        metric = metric_match.group(1)

    header = lines[1]
    metadata = parse_tau_xml(header)
    if metadata.get("Metric Name"):
        metric = metadata["Metric Name"]
    dimensions["Metric"] = metric

    timers, index = _read_records(
        lines, 2, timer_count, _parse_timer, dimensions, path, "timer"
    )

    index = _next_nonempty(lines, index)
    if index >= len(lines):
        raise ValueError("{}: missing aggregates section".format(path))
    aggregate_count = _section_count(lines[index], "aggregates", path, index + 1)
    aggregates, index = _read_records(
        lines, index + 1, aggregate_count, _parse_timer, dimensions, path, "aggregate"
    )

    counters = []
    index = _next_nonempty(lines, index)
    if index < len(lines):
        event_count = _section_count(lines[index], "userevents", path, index + 1)
        counters, index = _read_records(
            lines,
            index + 1,
            event_count,
            _parse_user_event,
            dimensions,
            path,
            "user event",
        )

    return OrderedDict(
        (
            ("process index", node),
            ("context index", context),
            ("thread index", thread),
            ("Metric", metric),
            ("Profile Format", profile_format),
            ("Metadata", metadata),
            ("Timers", timers),
            ("Aggregates", aggregates),
            ("Counters", counters),
            (
                "Additional Profile Data",
                [line.rstrip("\n") for line in lines[index:] if line.strip()],
            ),
        )
    )


def _profile_sort_key(filename):
    match = PROFILE_NAME_RE.fullmatch(filename)
    return tuple(int(value) for value in match.groups()) if match else (sys.maxsize,)


def _load_directory(indir):
    if not os.path.isdir(indir):
        raise ValueError("{}: not a profile directory".format(indir))
    profiles = sorted(
        (
            name
            for name in os.listdir(indir)
            if PROFILE_NAME_RE.fullmatch(name)
            and os.path.isfile(os.path.join(indir, name))
        ),
        key=_profile_sort_key,
    )
    if not profiles:
        raise ValueError("{}: no TAU profile.* files found".format(indir))

    application = OrderedDict()
    application["metadata"] = OrderedDict()
    application["Timers"] = []
    application["Aggregates"] = []
    application["Counters"] = []
    application["Profiles"] = []
    metrics = OrderedDict()
    parsed_profiles = []
    for filename in profiles:
        profile = parse_profile(os.path.join(indir, filename))
        parsed_profiles.append(profile)
        application["Profiles"].append(
            OrderedDict(
                (
                    ("Filename", filename),
                    ("process index", profile["process index"]),
                    ("context index", profile["context index"]),
                    ("thread index", profile["thread index"]),
                    ("Metric", profile["Metric"]),
                    ("Profile Format", profile["Profile Format"]),
                    ("Metadata", profile["Metadata"]),
                    ("Timers", profile["Timers"]),
                    ("Aggregates", profile["Aggregates"]),
                    ("Counters", profile["Counters"]),
                    ("Additional Profile Data", profile["Additional Profile Data"]),
                )
            )
        )
        thread_name = "Rank {}, Thread {}".format(
            profile["process index"], profile["thread index"]
        )
        if profile["context index"]:
            thread_name = "Rank {}, Context {}, Thread {}".format(
                profile["process index"],
                profile["context index"],
                profile["thread index"],
            )
        if thread_name in application["metadata"]:
            thread_name = "{}, Metric {}".format(thread_name, profile["Metric"])
        application["metadata"][thread_name] = profile["Metadata"]
        metrics[profile["Metric"]] = None
        for section in ("Timers", "Aggregates", "Counters"):
            application[section].extend(profile[section])
    application["Metrics"] = list(metrics)
    if len(metrics) == 1:
        application["Metric"] = next(iter(metrics))
    return application, parsed_profiles


def _sum_by_group(profiles):
    group_times = OrderedDict()
    group_calls = OrderedDict()
    for profile in profiles:
        for timer in profile["Timers"]:
            group = timer.get("Group", "")
            key = "ADIOS" if "ADIOS" in timer["Function"] else group
            if timer["Exclusive Time"] is not None:
                group_times[key] = (
                    group_times.get(key, 0) + timer["Exclusive Time"]
                )
            if timer["Calls"] is not None:
                group_calls[key] = group_calls.get(key, 0) + timer["Calls"]
    event_totals = OrderedDict()
    for profile in profiles:
        for event in profile["Counters"]:
            name = event["Counter"]
            if event["Num Events"] is None or event["Mean Value"] is None:
                continue
            average = event["Mean Value"] * event["Num Events"]
            if "Message size for " in name:
                key = "Collective_Bytes"
            elif name == "Message size sent to all nodes":
                key = "Send_Bytes"
            elif name == "Message size received from all nodes":
                key = "Recv_Bytes"
            elif name == "Bytes Read":
                key = "Read_Bytes"
            elif name == "Bytes Written":
                key = "Write_Bytes"
            elif name == "ADIOS data size":
                key = "ADIOS_data_size"
            else:
                continue
            event_totals[key] = event_totals.get(key, 0) + average
    return group_times, group_calls, event_totals


def _workflow_instance(workflow):
    instance = workflow.get("Workflow Instance")
    if isinstance(instance, list):
        if instance:
            return instance[0]
        instance.append(
            OrderedDict(
                (
                    ("id", 1),
                    ("name", "Workflow Instance"),
                    ("location-id", 1),
                    ("version", ""),
                )
            )
        )
        return instance[0]
    if not isinstance(instance, dict):
        instance = OrderedDict(
            (
                ("id", 1),
                ("name", "Workflow Instance"),
                ("location-id", 1),
                ("version", ""),
            )
        )
        workflow["Workflow Instance"] = instance
    return instance


def _application_for(workflow, component_name, app_name):
    applications = workflow.setdefault("Application", [])
    for application in applications:
        if application.get("name") in (component_name, app_name):
            return application["id"]
    app_id = len(applications) + 1
    applications.append(
        OrderedDict(
            (
                ("id", app_id),
                ("location-id", 1),
                ("name", app_name),
                ("version", "0.1"),
                ("uri", ""),
            )
        )
    )
    return app_id


def _build_workflow_component(workflow, component_name, application, profiles):
    metadata = next(
        (
            thread_metadata
            for thread_metadata in application["metadata"].values()
            if thread_metadata.get("Executable")
        ),
        {},
    )
    app_name = metadata.get("Executable", component_name)
    app_id = _application_for(workflow, component_name, app_name)

    timestamps = []
    time_profiles = [
        profile
        for profile in profiles
        if profile["Metric"].upper() in TIME_METRICS
    ]
    max_inclusive = max(
        (
            timer["Inclusive Time"]
            for profile in time_profiles
            for timer in profile["Timers"]
            if timer["Inclusive Time"] is not None
        ),
        default=0,
    )
    for profile in profiles:
        thread_metadata = profile["Metadata"]
        start = thread_metadata.get("Starting Timestamp")
        end = thread_metadata.get("Ending Timestamp")
        if start is not None:
            if end is not None:
                end = int(end)
            elif time_profiles:
                end = int(start) + int(max_inclusive)
            else:
                end = None
            timestamps.append((int(start), end, thread_metadata))

    start_timestamp = min((item[0] for item in timestamps), default=None)
    end_timestamp = max(
        (item[1] for item in timestamps if item[1] is not None), default=None
    )
    local_time = next((item[2].get("Local Time") for item in timestamps if item[2].get("Local Time")), None)
    components = workflow.setdefault("Workflow Component", [])
    component_basename = os.path.basename(os.path.normpath(component_name))
    component = next(
        (
            item
            for item in components
            if item.get("name") in (component_name, component_basename)
        ),
        None,
    )
    if component is None:
        component = OrderedDict(
            (
                ("id", app_id),
                ("name", app_name),
                ("location-id", 1),
                ("application-id", app_id),
            )
        )
        components.append(component)
    if start_timestamp is not None:
        component["start-timestamp"] = str(start_timestamp)
        component["end-timestamp"] = str(end_timestamp)
    if local_time is not None:
        component["Local-Time"] = local_time
    component["Processes"] = len({profile["process index"] for profile in profiles})
    component["Threads"] = len(profiles)

    instances = workflow.setdefault("Application-instance", [])
    instance = OrderedDict(
        (
            ("id", len(instances) + 1),
            ("process-id", metadata.get("pid", "0")),
            ("application-id", app_id),
            ("event-type", ""),
        )
    )
    if start_timestamp is not None:
        instance["start-timestamp"] = str(start_timestamp)
        instance["end-timestamp"] = str(end_timestamp)
    if local_time is not None:
        instance["Local-Time"] = local_time
    instances.append(instance)

    _, calls, event_totals = _sum_by_group(profiles)
    time_group_totals, _, _ = _sum_by_group(time_profiles)
    threads = max(len(profiles), 1)
    comm_calls = sum(value for name, value in calls.items() if "MPI" in name)
    comm_time = sum(
        value for name, value in time_group_totals.items() if "MPI" in name
    )
    io_time = sum(
        value for name, value in time_group_totals.items() if "TAU_IO" in name
    )
    adios_time = sum(
        value for name, value in time_group_totals.items() if "ADIOS" in name
    )
    user_time = sum(
        value
        for name, value in time_group_totals.items()
        if "TAU_USER" in name or "TAU_DEFAULT" in name
    )
    for name, value in (
        ("aggr_communication_calls", comm_calls),
        ("aggr_communication_time", comm_time / threads),
        ("aggr_io_time", io_time / threads),
        ("aggr_adios_time", adios_time / threads),
        ("aggr_execution_time", user_time / threads),
        ("aggr_communication_collective_bytes", event_totals.get("Collective_Bytes", 0)),
        ("aggr_communication_sent_bytes", event_totals.get("Send_Bytes", 0)),
        ("aggr_communication_recv_bytes", event_totals.get("Recv_Bytes", 0)),
        ("aggr_adios_bytes", event_totals.get("ADIOS_data_size", 0)),
        ("aggr_io_read_bytes", event_totals.get("Read_Bytes", 0)),
        ("aggr_io_write_bytes", event_totals.get("Write_Bytes", 0)),
    ):
        if value:
            component[name] = value
    if start_timestamp is not None and end_timestamp is not None:
        component["total_time"] = end_timestamp - start_timestamp
    return start_timestamp, end_timestamp


def _metrics(workflow, metric_names):
    metrics = []
    for metric_id, name in enumerate(metric_names, start=1):
        metrics.append(
            OrderedDict(
                (
                    ("id", metric_id),
                    ("location-id", 1),
                    ("measurement", name),
                    (
                        "units",
                        "microseconds" if name.upper() in TIME_METRICS else "unknown",
                    ),
                    ("description", "TAU profile metric {}".format(name)),
                )
            )
        )
    workflow["Metrics"] = metrics


def parse_args(arguments):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "indir",
        help="One or more directories containing TAU text profiles",
        nargs="+",
    )
    parser.add_argument(
        "-w", "--workflow", help="Workflow metadata input file (JSON)", default=None
    )
    parser.add_argument(
        "-o", "--outfile", help="Output JSON file (default: stdout)", default="-"
    )
    return parser.parse_args(arguments)


def main(arguments=None):
    args = parse_args(arguments)
    if args.workflow:
        with open(args.workflow, encoding="utf-8") as workflow_file:
            workflow = json.load(workflow_file, object_pairs_hook=OrderedDict)
    else:
        workflow = OrderedDict()
        workflow["Workflow Component"] = []
    _workflow_instance(workflow)
    workflow.setdefault("Application", [])
    workflow.setdefault("Application-instance", [])

    output = OrderedDict()
    output["Workflow metadata"] = workflow
    workflow_start = None
    workflow_end = None
    metric_names = OrderedDict()
    for indir in args.indir:
        print("Processing: {}".format(indir), file=sys.stderr)
        application, profiles = _load_directory(indir)
        output[indir] = application
        for metric in application["Metrics"]:
            metric_names[metric] = None
        start, end = _build_workflow_component(
            workflow, indir, application, profiles
        )
        if start is not None:
            workflow_start = start if workflow_start is None else min(workflow_start, start)
            workflow_end = end if workflow_end is None else max(workflow_end, end)
    if workflow_start is not None and workflow_end is not None:
        _workflow_instance(workflow)["execution_time"] = workflow_end - workflow_start
    _metrics(workflow, metric_names)

    json_output = json.dumps(output, indent=2, allow_nan=False)
    if args.outfile == "-":
        sys.stdout.write(json_output)
        sys.stdout.write("\n")
    else:
        with open(args.outfile, "w", encoding="utf-8") as outfile:
            outfile.write(json_output)
            outfile.write("\n")


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print("tau_prof2json.py: {}".format(error), file=sys.stderr)
        sys.exit(2)
