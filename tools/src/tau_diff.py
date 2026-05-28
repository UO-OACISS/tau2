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

# Pre-compiled patterns for TAU name normalization
_LOCATION_RE = re.compile(r'\s*\[{[^}]*}\s+{[^}]*}(?:-{[^}]*})?\]\s*$')
_LANG_TAG_RE = re.compile(r'\s+[A-Z][a-zA-Z+]*\s*$')
_PARAMS_RE = re.compile(r'\s*\(.*\)\s*$')
_SKIP_PREFIXES = ('.TAU', '[SAMPLE]', '[CONTEXT]', 'parallel ', 'for (', 'barrier ')

def extract_bare_name(raw_name):
    """
    Extracts the canonical function name from a raw TAU profile entry name
    (which includes surrounding quotes), for use in fuzzy cross-profile matching.

    Works for both compiler-instrumented and PDT-instrumented name formats:
      Comp:  '"funcname [{/abs/path/file.c} {line,col}]"'
      PDT:   '"rettype funcname(params) LANG [{file.c} {start,col}-{end,col}]  "'

    Returns the base function name string, or None if the entry cannot be
    meaningfully reduced (OpenMP constructs, TAU internals, sampling/context
    markers, or callpath entries).
    """
    s = raw_name.strip().strip('"').strip()
    # Callpath entries are handled by normalize_entry_key
    if ' => ' in s:
        return None
    # Non-normalizable entry types
    if any(s.startswith(p) for p in _SKIP_PREFIXES):
        return None
    if s in ('taupreload_main',):
        return None
    # Strip file/line location annotation: [{...} {...}] or [{...} {...}-{...}]
    s = _LOCATION_RE.sub('', s).strip()
    # Strip trailing language tag (C, Fortran, C++, etc.)
    s = _LANG_TAG_RE.sub('', s).strip()
    # Strip parameter list
    s = _PARAMS_RE.sub('', s).strip()
    # Remaining is "funcname" or "rettype funcname" (possibly with pointer
    # decorators like "double **funcname"). The function name is the last token.
    parts = s.split()
    if not parts:
        return None
    base = parts[-1].lstrip('*')
    return base or None


def normalize_entry_key(raw_name):
    """
    Normalizes a full TAU profile entry name (including surrounding quotes) to a
    canonical key for cross-instrumentation matching.

    For callpath entries (containing ' => '), each segment is normalized
    independently. Segments that can be reduced to a base function name (user
    functions) use that name; segments that cannot (OpenMP constructs, TAU
    internals, samples) keep their literal text as the key, since they are
    identical across instrumentation styles and act as reliable anchors.

    Returns None only for non-callpath entries that cannot be meaningfully
    reduced (i.e., extract_bare_name returns None for them).
    """
    s = raw_name.strip().strip('"').strip()
    if ' => ' in s:
        segments = s.split(' => ')
        norm_segs = []
        for seg in segments:
            n = extract_bare_name('"' + seg.strip() + '"')
            # Fall back to literal text for non-normalizable segments (OpenMP
            # constructs, SAMPLE/CONTEXT markers, etc.) — they are identical
            # across instrumentation styles and serve as callpath anchors.
            norm_segs.append(n if n is not None else seg.strip())
        return ' => '.join(norm_segs)
    return extract_bare_name(raw_name)


def compare_profiles(p1, p2, threshold, sort_by, normalize=False):
    """Compares two parsed TauProfile objects and returns structured results."""
    results = {
        'metadata': [],
        'functions': [],
        'user_events': [],
        'metric_mismatch': p1.metric_name != p2.metric_name,
        'normalized_matches': [],      # (p1_name, p2_name, norm_key) triples
        'ambiguous_norm_matches': [],  # dicts with norm_key, p1_names, p2_names
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

    # Build normalized-name lookup tables when --normalize is active
    p2_by_norm = {}   # norm_key -> [p2_exact_name, ...]
    p1_by_norm = {}   # norm_key -> [p1_exact_name, ...]
    if normalize:
        for name in p2.functions:
            nk = normalize_entry_key(name)
            if nk:
                p2_by_norm.setdefault(nk, []).append(name)
        for name in p1.functions:
            nk = normalize_entry_key(name)
            if nk:
                p1_by_norm.setdefault(nk, []).append(name)

    # Build comparison pairs: (p1_name_or_None, p2_name_or_None, match_type)
    pairs = []
    p2_accounted = set()
    reported_ambig_keys = set()

    for p1_name in p1.functions:
        if p1_name in p2.functions:
            pairs.append((p1_name, p1_name, 'exact'))
            p2_accounted.add(p1_name)
        elif normalize:
            nk = normalize_entry_key(p1_name)
            if nk and nk in p2_by_norm:
                p2_matches = p2_by_norm[nk]
                p1_matches = p1_by_norm.get(nk, [p1_name])
                if len(p2_matches) == 1 and len(p1_matches) == 1:
                    p2_name = p2_matches[0]
                    if p2_name not in p2_accounted:
                        pairs.append((p1_name, p2_name, 'normalized'))
                        p2_accounted.add(p2_name)
                        results['normalized_matches'].append((p1_name, p2_name, nk))
                        continue
                elif nk not in reported_ambig_keys:
                    # Multiple functions share this base name: cannot auto-resolve.
                    # This is the fundamental overloaded-function limitation of
                    # compiler instrumentation, which loses parameter type info.
                    results['ambiguous_norm_matches'].append({
                        'norm_key': nk,
                        'p1_names': p1_matches,
                        'p2_names': p2_matches,
                    })
                    reported_ambig_keys.add(nk)
            pairs.append((p1_name, None, 'exact'))
        else:
            pairs.append((p1_name, None, 'exact'))

    for p2_name in p2.functions:
        if p2_name not in p2_accounted:
            pairs.append((None, p2_name, 'exact'))

    # Process pairs
    for p1_name, p2_name, match_type in pairs:
        f1 = p1.functions.get(p1_name) if p1_name else None
        f2 = p2.functions.get(p2_name) if p2_name else None
        is_new = f1 is None
        is_removed = f2 is None

        display_name = p1_name if p1_name else p2_name
        original_index = p1_order_map.get(p1_name, float('inf'))
        diff = {
            'name': display_name,
            'metrics': {},
            'max_abs_change': 0.0,
            'sort_key': None,
            'original_index': original_index,
            'match_type': match_type,
        }
        if match_type == 'normalized' and p2_name != p1_name:
            diff['name_p2'] = p2_name

        metrics_to_check = f2.keys() if is_new else f1.keys()
        for metric in metrics_to_check:
            v1 = 0 if is_new else f1.get(metric, 0)
            v2 = 0 if is_removed else f2.get(metric, 0)
            p_change = calculate_diff(v1, v2)
            diff['metrics'][metric] = {'v1': v1, 'v2': v2, 'p_change': p_change}
            diff['max_abs_change'] = max(diff['max_abs_change'], abs(p_change))

        if diff['max_abs_change'] > threshold:
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
    else:  # Default numeric sort
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

    if results.get('normalized_matches'):
        report.append(f"# Note: {len(results['normalized_matches'])} function(s) matched via normalized name (--normalize mode).")

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
            if func.get('match_type') == 'normalized':
                p2_name = func.get('name_p2', func['name'])
                if p2_name != func['name']:
                    title += f"\n#  ~~ normalized match (Profile 2): {p2_name}"
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

    # Ambiguous Normalized Matches
    if results.get('ambiguous_norm_matches'):
        report.append("\n# AMBIGUOUS NORMALIZED MATCHES (skipped)")
        report.append("# The following functions share the same base name in both profiles but")
        report.append("# cannot be unambiguously paired. This is the fundamental limitation of")
        report.append("# compiler instrumentation for overloaded functions: parameter type info")
        report.append("# is lost, making it impossible to reliably match them to their PDT")
        report.append("# counterparts. Use source-level disambiguation or PDT on both sides.")
        report.append("#" + "-" * 70)
        for ambig in results['ambiguous_norm_matches']:
            report.append(f"\n#  Base name: '{ambig['norm_key']}'")
            for n in ambig['p1_names']:
                report.append(f"#    Profile 1: {n}")
            for n in ambig['p2_names']:
                report.append(f"#    Profile 2: {n}")

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
    parser.add_argument(
        "--normalize", action="store_true",
        help="Enable fuzzy name matching between compiler-instrumented and\n"
             "PDT-instrumented profiles. Strips return types, parameter lists,\n"
             "file paths, and line numbers to match functions by base name.\n"
             "Unambiguous 1-to-1 matches are paired; ambiguous cases (e.g.,\n"
             "overloaded functions) are reported but not force-matched, since\n"
             "compiler instrumentation cannot distinguish them."
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
        results = compare_profiles(p1, p2, args.threshold, args.sort_by, normalize=args.normalize)
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
                results = compare_profiles(p1, p2, args.threshold, args.sort_by, normalize=args.normalize)
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
        results = compare_profiles(p1, p2, args.threshold, args.sort_by, normalize=args.normalize)
        report = generate_report_string(p1, p2, results, args.threshold, args.sort_by)
        
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"Comparison complete. Report written to {output_file}")
    
    else:
        print("Error: Both paths must be either files or directories.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
