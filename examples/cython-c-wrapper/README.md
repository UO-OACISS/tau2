# cython-c-wrapper
[Simple example of wrapping a C library with Cython](https://stavshamir.github.io/python/making-your-c-library-callable-from-python-by-wrapping-it-with-cython/).

This example was adapted to show TAU instrumentation with TAU python measurement.  This example assumes the Cython python module is installed.  If you don't have it, do:

```bash
python3 -m pip install --user cython
```

To run this exmaple, configure TAU with python and binutils support, for example, if python3 is installed in /packages/python/3.6.8:

```bash
./configure -python -bfd=/usr/local/packages/binutils/2.34 -pythoninc=/packages/python/3.6.8/include/python3.6m -pythonlib=/packages/python/3.6.8/lib
```

or

```bash
./configure -python -bfd=download -pythoninc=/packages/python/3.6.8/include/python3.6m -pythonlib=/packages/python/3.6.8/lib
```

...and then in this directory, run `make test`.  After the run, use the `pprof` program to display the output, and you should see the C function `hello_from_c_function` in the output, along with all the events captured by the Python profiling interface:

```bash
[khuck@delphi cython-c-wrapper]$ pprof
Reading Profile files in profile.*

NODE 0;CONTEXT 0;THREAD 0:
---------------------------------------------------------------------------------------
%Time    Exclusive    Inclusive       #Call      #Subrs  Inclusive Name
              msec   total msec                          usec/call
---------------------------------------------------------------------------------------
100.0           92          104           1           5     104781 .TAU application
  5.3        0.011            5           1           1       5558 exec
  5.3        0.024            5           1           2       5547 <module>
  3.8            3            3           1           0       3991 compile
  3.0        0.039            3           1           6       3125 _find_and_load
  3.0        0.394            3           1          21       3093 find_module
  2.8        0.022            2           1           3       2969 _find_and_load_unlocked
  2.5        0.041            2           1           6       2613 _load_unlocked
  2.4        0.018            2           1           3       2496 module_from_spec
  2.3        0.015            2           2           2       1212 _call_with_frames_removed
  2.3        0.011            2           1           2       2408 create_module
  2.3            2            2           1           1       2398 py_hello_from_c
  2.3            2            2           1           1       2386 create_dynamic
  1.6        0.066            1           6           7        288 isfile
  1.6            1            1           8           0        213 stat
  0.7        0.498        0.708           2           2        354 open
  0.4        0.074        0.439           4          10        110 find_spec
  0.3        0.067        0.333           1           9        333 _find_spec
  0.3        0.033        0.271           2           6        136 _get_spec
  0.2        0.073          0.2           1           8        200 search_function
  0.1         0.03        0.122           1           4        122 detect_encoding
  0.1        0.071        0.112           7          23         16 join
  0.1        0.019        0.093           5           5         19 __enter__
  0.1        0.032         0.07           1           8         70 _init_module_attrs
  0.1        0.033         0.06           5           6         12 __exit__
  0.0        0.052        0.052           1           0         52 hello_from_c_function
  0.0        0.036        0.052           1           1         52 read
  0.0         0.03         0.05           1           3         50 _get_module_lock
  0.0        0.002         0.05           2           2         25 _path_stat
  0.0        0.038        0.047           1          12         47 normalize_encoding
  0.0        0.014        0.044           1           2         44 find_cookie
  0.0        0.038        0.042           7           3          6 __init__
  0.0        0.017         0.04           1           1         40 getregentry
  0.0        0.011        0.039           1           1         39 read_or_stop
  0.0        0.007        0.038           1           2         38 exec_module
  0.0         0.02        0.033           1           1         33 __import__
  0.0        0.032        0.032           2           0         16 match
  0.0        0.006        0.031           1           1         31 _path_isfile
  0.0        0.028        0.028           1           0         28 readline
  0.0        0.004        0.025           1           1         25 _path_is_mode_type
  0.0        0.022        0.025           1           3         25 get_suffixes
  0.0        0.008        0.024           1           2         24 _path_join
  0.0         0.02        0.023           1           1         23 __new__
  0.0         0.01        0.023           1           1         23 _path_importer_cache
  0.0        0.013        0.023           1           1         23 cached
  0.0        0.023        0.023           1           0         23 exec_dynamic
  0.0        0.018        0.023           1           2         23 spec_from_file_location
  0.0         0.01        0.021           7           7          3 _get_sep
  0.0         0.02        0.021           1           1         21 acquire
  0.0        0.018        0.018          12           0          2 isinstance
  0.0        0.014        0.017           4           2          4 <listcomp>
  0.0        0.012        0.016           1           1         16 decode
  0.0        0.015        0.015           6           0          2 getattr
  0.0        0.014        0.014           8           0          2 startswith
  0.0        0.008        0.013           1           1         13 _handle_fromlist
  0.0        0.013        0.013           1           0         13 getcwd
  0.0         0.01        0.012           1           1         12 release
  0.0        0.007        0.011           1           4         11 any
  0.0        0.011        0.011           8           0          1 fspath
  0.0        0.007         0.01           1           2         10 _get_cached
  0.0        0.007        0.009           1           1          9 parent
  0.0        0.008        0.008           4           0          2 _verbose_message
  0.0        0.005        0.008           1           3          8 cb
  0.0        0.008        0.008           4           0          2 hasattr
  0.0        0.006        0.006           5           0          1 get
  0.0        0.006        0.006           1           0          6 new_module
  0.0        0.005        0.005           5           0          1 acquire_lock
  0.0        0.005        0.005           2           0          2 is_builtin
  0.0        0.005        0.005           5           0          1 release_lock
  0.0        0.004        0.004           4           0          1 <genexpr>
  0.0        0.004        0.004           4           0          1 endswith
  0.0        0.004        0.004           2           0          2 join
  0.0        0.004        0.004           3           0          1 rpartition
  0.0        0.004        0.004           1           0          4 utf_8_decode
  0.0        0.003        0.003           1           0          3 __new__
  0.0        0.003        0.003           5           0          1 append
  0.0        0.003        0.003           2           0          2 get_ident
  0.0        0.003        0.003           2           0          2 is_frozen
  0.0        0.003        0.003           5           0          1 isalnum
  0.0        0.003        0.003           2           0          2 rstrip
  0.0        0.002        0.002           2           0          1 allocate_lock
  0.0        0.002        0.002           1           0          2 decode
  0.0        0.002        0.002           1           0          2 replace
  0.0        0.001        0.001           1           0          1 S_ISREG
  0.0        0.001        0.001           1           0          1 _relax_case
  0.0            0            0           1           0          0 has_location
```
