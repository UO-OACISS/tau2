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
`Counters` arrays for convenient analysis. `Profiles` contains the same
records grouped by source file, together with each file's metric, metadata,
profile-format header, and any unparsed trailing profile lines. For each
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
