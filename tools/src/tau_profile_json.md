# TAU profile JSON

[`tau_prof2json.py`](tau_prof2json.py) converts one or more directories of TAU
text profile files (`profile.<rank>.<context>.<thread>`) into JSON. Each
directory becomes an application-profile object keyed by the directory
argument; `Workflow metadata` contains the workflow description and derived
application information.

The JSON contract is described by
[`tau_profile_json.schema.json`](tau_profile_json.schema.json), using JSON
Schema Draft 2020-12. This is the converter's output schema, not a schema for
TAU's source text format or `.ppk` packed files.

Each application-profile object retains flattened `Timers`, `Aggregates`, and
`Counters` arrays; these are the only copy of the records. `Profiles` lists the
source files with each file's metric, metadata, profile-format header, record
counts (`Timer Count`, `Aggregate Count`, `Counter Count`), and any unparsed
trailing profile lines. For each
record, `process index`, `context index`, `thread index`, and `Metric` identify
the source profile and measurement.

The workflow object may include user-supplied properties such as workflow
inputs, outputs, dependency graph, and location. Its shape is therefore
intentionally extensible in the schema.

## Validate a profile directory

Run the focused parser tests and additionally reconcile every source profile,
timer, aggregate, and user event against the JSON produced by the command-line
converter:

```sh
python3 tools/src/tests/test_tau_prof2json.py --profile-dir /path/to/profiles
```

For example, the repository's raw bt.mz profiles can be checked with:

```sh
python3 tools/src/tests/test_tau_prof2json.py --profile-dir data/tau
```

The validator expects raw `profile.*` text files. It does not read `.ppk`
archives or compare against fixed metric values, so the same check can be used
with newly generated or otherwise arbitrary profile directories.

## Versioning and provenance

`Workflow metadata.profile-json-version` identifies this layout (`1.0`, or
`1.0-compact` for `--compact` output); it is the only place the version is recorded, so the top-level keys
remain "Workflow metadata" plus one key per input directory. When an ending
timestamp is missing and is inferred from the longest inclusive time, the
affected workflow component and instance carry `end-timestamp-estimated: true`.
The `aggr_*` values are derived by the converter, not TAU measurements.

## Compact layout (`--compact`)

`--compact` emits a layout for LLM and other token-limited consumers. It is
described by [`tau_profile_json_compact.schema.json`](tau_profile_json_compact.schema.json)
and marked `profile-json-version: "1.0-compact"`. It is lossless:
`tau_prof2json.expand_compact()` restores the full layout. On the bt.mz demo
profiles it is about 80% smaller than the full layout without indentation.
Per application:

- `Timers`, `Aggregates`, `Counters`, and `Profiles` are each a list of
  tables, normally exactly one. A table is
  `{"constants": {...}, "columns": [...], "rows": [[...], ...]}`. Each row
  lists values in `columns` order. A key in `constants` has the same value in
  every record of the table, so read it as an implicit extra column. A new
  table starts only if the record keys change (for example a timer without a
  `Group`).
- Columns named in `indexed` (`Function` and `Group` for timers and
  aggregates, `Counter` for counters) hold integers that index the application's `strings` list:
  the name is `strings[value]`.
- `common_metadata` holds the metadata attributes identical on every thread.
  `thread_metadata` maps each thread name to only its differing attributes;
  a thread's full metadata is `common_metadata` plus its entry. Each
  `Profiles` row has a `Thread` column naming its entry and no inline
  `Metadata`.
- Integer-valued numbers are written without `.0`. `Metrics`, `Metric`, and
  the workflow metadata are otherwise unchanged. Output has no indentation.

Example row, with columns `process index, thread index, Function, Calls, ...`:
`[0,0,5,6432,6432,4737,795480,2]` is process 0, thread 0,
function `strings[5]`, 6432 calls, ..., group `strings[2]`.
