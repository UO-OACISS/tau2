#!/usr/bin/env python3
import os
import sys
import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

class TauProfile:
    """
    Parses and stores the data from a single TAU profile file.
    """
    def __init__(self, filepath):
        self.filepath = filepath
        self.metric_name = "UNKNOWN"
        self.metadata = {}
        self.functions = {}
        self.user_events = {}
        self._parse()

    def _parse(self):
        """Parses the entire profile file."""
        try:
            with open(self.filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except FileNotFoundError:
            print(f"Error: File not found at {self.filepath}", file=sys.stderr)
            sys.exit(1)

        # State machine for parsing different sections
        parsing_user_events = False

        # Regex patterns
        header_re = re.compile(r"^\d+\s+templated_functions_(.*)")
        func_re = re.compile(r'^(".*?")\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+GROUP=')
        user_event_header_re = re.compile(r"^\d+\s+userevents")
        user_event_data_re = re.compile(r'^(".*?")\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)')

        lines = content.splitlines()
        for line in lines:
            # Prioritize checking for metadata on any line, even commented ones.
            if "<metadata>" in line:
                # Isolate just the XML part of the string and parse it.
                try:
                    metadata_blob = line[line.find("<metadata>"):]
                    self._parse_metadata(metadata_blob)
                except Exception as e:
                    print(f"Warning: Failed to process metadata line in {self.filepath}: {e}", file=sys.stderr)
                # We've handled this line, so skip to the next one.
                continue

            line = line.strip()
            # This check is now safe, as the metadata line has already been processed.
            if not line or line.startswith('#'):
                continue

            # This state is now only for lines *after* the "X userevents" header
            if parsing_user_events:
                match = user_event_data_re.match(line)
                if match:
                    name = match.group(1)
                    self.user_events[name] = {
                        "Numevents": float(match.group(2)),
                        "Max": float(match.group(3)),
                        "Min": float(match.group(4)),
                        "Mean": float(match.group(5)),
                        "Sumsqr": float(match.group(6)),
                    }
                continue # Continue to next line after processing a user event

            # Header line (without metadata)
            match = header_re.match(line)
            if match:
                self.metric_name = match.group(1)
                continue

            # User event section header
            if user_event_header_re.match(line):
                parsing_user_events = True
                continue

            # Function data
            match = func_re.match(line)
            if match:
                name = match.group(1)
                self.functions[name] = {
                    "Calls": int(float(match.group(2))),
                    "Subrs": int(float(match.group(3))),
                    "Excl": float(match.group(4)),
                    "Incl": float(match.group(5)),
                }

    def _parse_metadata(self, xml_string):
        """Parses the XML metadata blob."""
        try:
            # Clean up the blob
            xml_string = xml_string[xml_string.find("<metadata>"):]
            root = ET.fromstring(xml_string)
            for attr in root.findall('attribute'):
                name_elem = attr.find('name')
                value_elem = attr.find('value')
                if name_elem is not None and value_elem is not None:
                    self.metadata[name_elem.text] = value_elem.text
        except ET.ParseError:
            print(f"Warning: Could not parse XML metadata in {self.filepath}", file=sys.stderr)


def calculate_diff(v1, v2):
    """Calculates percentage change and handles edge cases."""
    if v1 == v2:
        return 0.0
    if v1 == 0:
        return float('inf')  # Represents "New"
    if v2 == 0:
        return -100.0  # Represents "Removed"
    return ((v2 - v1) / v1) * 100.0

def compare_profiles(p1, p2, threshold, sort_by):
    """Compares two parsed TauProfile objects and returns structured results."""
    results = {
        'metadata': [],
        'functions': [],
        'user_events': [],
        'metric_mismatch': p1.metric_name != p2.metric_name,
    }

    # Compare Metadata
    all_meta_keys = sorted(list(set(p1.metadata.keys()) | set(p2.metadata.keys())))
    for key in all_meta_keys:
        v1 = p1.metadata.get(key)
        v2 = p2.metadata.get(key)
        if v1 != v2:
            results['metadata'].append({'key': key, 'v1': v1, 'v2': v2})

    # Create a map of function name to its original order index in profile 1
    p1_order_map = {name: i for i, name in enumerate(p1.functions.keys())}

    # Compare Functions
    all_func_keys = set(p1.functions.keys()) | set(p2.functions.keys())
    for key in all_func_keys:
        f1 = p1.functions.get(key)
        f2 = p2.functions.get(key)
        
        # Add original_index to the diff dictionary
        original_index = p1_order_map.get(key, float('inf'))
        diff = {'name': key, 'metrics': {}, 'max_abs_change': 0.0, 'sort_key': None, 'original_index': original_index}
        
        is_new = f1 is None
        is_removed = f2 is None
        
        metrics_to_check = f2.keys() if is_new else f1.keys()

        for metric in metrics_to_check:
            v1 = 0 if is_new else f1.get(metric, 0)
            v2 = 0 if is_removed else f2.get(metric, 0)
            p_change = calculate_diff(v1, v2)
            diff['metrics'][metric] = {'v1': v1, 'v2': v2, 'p_change': p_change}
            diff['max_abs_change'] = max(diff['max_abs_change'], abs(p_change))
        
        if diff['max_abs_change'] > threshold:
            # Set the key for metric-based sorting
            if sort_by in diff['metrics']:
                diff['sort_key'] = diff['metrics'][sort_by]['p_change']
            results['functions'].append(diff)
            
    # Compare User Events
    all_event_keys = set(p1.user_events.keys()) | set(p2.user_events.keys())
    for key in all_event_keys:
        e1 = p1.user_events.get(key)
        e2 = p2.user_events.get(key)
        diff = {'name': key, 'metrics': {}, 'max_abs_change': 0.0}

        is_new = e1 is None
        is_removed = e2 is None
        
        metrics_to_check = e2.keys() if is_new else e1.keys()

        for metric in metrics_to_check:
            v1 = 0 if is_new else e1.get(metric, 0)
            v2 = 0 if is_removed else e2.get(metric, 0)
            p_change = calculate_diff(v1, v2)
            diff['metrics'][metric] = {'v1': v1, 'v2': v2, 'p_change': p_change}
            diff['max_abs_change'] = max(diff['max_abs_change'], abs(p_change))

        if diff['max_abs_change'] > threshold:
            results['user_events'].append(diff)

    # Sort results based on the chosen method
    if sort_by == 'alpha':
        results['functions'].sort(key=lambda x: x['name'])
    elif sort_by == 'original':
        results['functions'].sort(key=lambda x: x['original_index'])
    else: # Default numeric sort
        results['functions'].sort(key=lambda x: abs(x.get('sort_key') or 0), reverse=True)

    return results

def format_diff_value(p_change):
    if p_change == float('inf'):
        return "(New)"
    if p_change == -100.0:
        return "(Removed)"
    return f"({p_change:+.1f}%)"

def generate_report_string(p1, p2, results, threshold, sort_by):
    """Formats the final diff report string."""
    report = []
    
    # Header
    report.append("# TAU Profile Difference Report")
    report.append("#")
    report.append(f"# Profile 1: {p1.filepath}")
    report.append(f"# Profile 2: {p2.filepath}")
    report.append("#")
    report.append(f"# Threshold: Showing entries where any metric's change > {threshold}%.")
    
    if sort_by in ['alpha', 'original']:
        report.append(f"# Sorted by: {sort_by} order.")
    else:
        report.append(f"# Sorted by: Largest change in '{sort_by}' column.")
    
    report.append("#" + "-" * 70)
    
    # Metadata
    if results['metadata']:
        report.append("\n# METADATA DIFFERENCES")
        report.append("# Profile 1                               | Profile 2")
        report.append("#" + "-" * 70)
        for item in results['metadata']:
            v1_str = str(item['v1']) if item['v1'] is not None else "(Removed)"
            v2_str = str(item['v2']) if item['v2'] is not None else "(Removed)"
            report.append(f"{item['key']+':':<25} {v1_str:<30}| {v2_str}")
    
    # Metric Mismatch Warning
    if results['metric_mismatch']:
        report.append("\n# METRIC MISMATCH WARNING")
        report.append(f'# Warning: Profile 1 metric "{p1.metric_name}" differs from Profile 2 "{p2.metric_name}". Values may not be comparable.')
        report.append("#" + "-" * 70)
    
    # Functions
    if results['functions']:
        report.append("\n# FUNCTION/EVENT DIFFERENCES")
        report.append("#" + "-" * 70)
        for func in results['functions']:
            is_new = any(m['p_change'] == float('inf') for m in func['metrics'].values())
            is_removed = any(m['p_change'] == -100.0 for m in func['metrics'].values())
            
            title = func['name']
            if is_new: title += " (Unique to Profile 2)"
            if is_removed: title += " (Unique to Profile 1)"
            report.append(f"\n{title}")

            for metric, data in func['metrics'].items():
                v1_str = f"{data['v1']:.1f}" if isinstance(data['v1'], float) else str(data['v1'])
                v2_str = f"{data['v2']:.1f}" if isinstance(data['v2'], float) else str(data['v2'])
                change_str = format_diff_value(data['p_change'])
                report.append(f"    {metric+':':<8} [{v1_str} -> {v2_str}] {change_str}")

    # User Events
    if results['user_events']:
        report.append("\n# USER EVENT DIFFERENCES")
        report.append("#" + "-" * 70)
        for event in results['user_events']:
            report.append(f"\n{event['name']}")
            for metric, data in event['metrics'].items():
                v1_str = f"{data['v1']:.2f}" if isinstance(data['v1'], float) else str(data['v1'])
                v2_str = f"{data['v2']:.2f}" if isinstance(data['v2'], float) else str(data['v2'])
                change_str = format_diff_value(data['p_change'])
                report.append(f"    {metric+':':<10} [{v1_str} -> {v2_str}] {change_str}")

    report.append("\n#" + "-" * 70)
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(
        description="Compares two TAU performance profiles or directories of profiles.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("path1", type=str, help="Path to the first profile or directory.")
    parser.add_argument("path2", type=str, help="Path to the second profile or directory.")
    parser.add_argument(
        "-o", "--output", type=str,
        help="Path for the output file or directory."
    )
    parser.add_argument(
        "-t", "--threshold", type=float, default=0.0,
        help="Report entries where any metric's %% change exceeds this value.\n(Default: 0.0)"
    )
    parser.add_argument(
        "--sort-by", type=str, default="Incl",
        choices=["Calls", "Subrs", "Excl", "Incl", "alpha", "original"],
        help="Sort the report by the change in a metric column, alphabetically,\n"
             "or by the original order in the first profile.\n"
             "(Default: Incl)"
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Check-only mode. Exits with 0 if no differences > threshold,\n1 otherwise. Suppresses all output files."
    )
    args = parser.parse_args()

    if args.check and args.output:
        print("Error: --check and --output are mutually exclusive.", file=sys.stderr)
        sys.exit(1)

    path1 = Path(args.path1)
    path2 = Path(args.path2)

    if not path1.exists() or not path2.exists():
        print(f"Error: A specified path does not exist. Checked: {path1}, {path2}", file=sys.stderr)
        sys.exit(1)

    # --- Check Mode ---
    if args.check:
        p1 = TauProfile(path1)
        p2 = TauProfile(path2)
        results = compare_profiles(p1, p2, args.threshold, args.sort_by)
        if results['functions'] or results['user_events'] or results['metadata'] or results['metric_mismatch']:
            sys.exit(1) # Differences found
        else:
            sys.exit(0) # No differences

    # --- Directory Mode ---
    if path1.is_dir() and path2.is_dir():
        output_dir = Path(args.output or "tau_diff_results")
        output_dir.mkdir(exist_ok=True)
        print(f"Comparing directories, outputting to {output_dir}")

        files1 = {f.name for f in path1.glob("profile.*")}
        files2 = {f.name for f in path2.glob("profile.*")}
        
        all_files = sorted(list(files1 | files2))

        for fname in all_files:
            fpath1 = path1 / fname
            fpath2 = path2 / fname
            out_path = output_dir / (fname + ".diff")

            if fname in files1 and fname in files2:
                p1 = TauProfile(fpath1)
                p2 = TauProfile(fpath2)
                results = compare_profiles(p1, p2, args.threshold, args.sort_by)
                report = generate_report_string(p1, p2, results, args.threshold, args.sort_by)
                with open(out_path, 'w') as f:
                    f.write(report)
            else:
                missing_in = "path2" if fname in files1 else "path1"
                with open(out_path, 'w') as f:
                    f.write(f"# File '{fname}' is missing in {missing_in} ({path2 if missing_in=='path2' else path1}).")
        print("Directory comparison complete.")

    # --- File Mode ---
    elif path1.is_file() and path2.is_file():
        output_file = args.output or "profile.diff"
        p1 = TauProfile(path1)
        p2 = TauProfile(path2)
        results = compare_profiles(p1, p2, args.threshold, args.sort_by)
        report = generate_report_string(p1, p2, results, args.threshold, args.sort_by)
        
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"Comparison complete. Report written to {output_file}")
    
    else:
        print("Error: Both paths must be either files or directories.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
