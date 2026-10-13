# SALT-FM selective instrumentation tests

`run.sh` runs `saltfm` (C/C++ path) on `cases/src.cpp` with each `cases/*.tau` and checks which
timers appear in the output (`cases/*.expect`, lines `timer: <name> => yes|no`). SALT-FM names C/C++
timers with the full signature (`int foo(int)`), so these cases are written against those names
rather than reusing `../common`.

Needs `saltfm` on PATH (`spack load saltfm`); exits 77 (SKIP) otherwise.

`disabled-*` cases describe TAU behavior SALT-FM does not support yet (include list on the C/C++ path,
trailing whitespace, literal regex metacharacters, one-character `?`). They are skipped by default;
run with `SALT_RUN_DISABLED=1` to see them fail, and rename them once SALT-FM is fixed.
The Fortran path is not covered yet. Once the shared-name issue is settled, `../common` can be replayed here.
