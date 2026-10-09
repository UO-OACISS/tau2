# Selective instrumentation sanity checks

One set of format cases in `common/` (`NN_name.tau` + `NN_name.expect`) is replayed by every consumer
of the TAU selective-instrumentation (SI) format. Each consumer has a directory with a `run.sh`:

| dir | consumer |
|---|---|
| `pdt/` | `utils/tau_selective.cpp` (tau_instrumentor, tau_selectfile) |
| `dyninst/` | `utils/tau_selective.dyn.cpp` (tau_run / tau_selectfile) |
| `runtime-plugin/` | disable-instrumentation runtime plugin |
| `llvm/` | `plugins/llvm/src/Instrument.cpp` |
| `shell/` | `tau_selectfile.sh`, `tau_rewrite`, `tau_compiler.sh` |
| `salt/` | placeholder (SALT SI is WIP) |

Expect lines: `func: <name> => yes|no`, `file: <path> => yes|no`, `loop: <file:line> => yes|no`
(yes = instrumented). Run all with `./run_all.sh`; exit 77 from a harness means SKIP.

Semantics tested: `#` at line start is a comment; function wildcard `#`, file wildcard `*`, `?` = one char;
names without `/` match the file basename; `"quoted"` function entries; exclude beats include;
unknown lines and `BEGIN_INSTRUMENT_SECTION` are ignored by list parsers; `BEGIN_LOOPS_LIST` is an LLVM extension.
Environment: `PDTDIR` (default regression path) for `pdt/`; LLVM dev files for `llvm/`.
