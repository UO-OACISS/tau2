# cython-c-wrapper
[Simple example of wrapping a C library with Cython](https://stavshamir.github.io/python/making-your-c-library-callable-from-python-by-wrapping-it-with-cython/).

This example was adapted to show TAU instrumentation with TAU python measurement.  This example assumes the Cython python module is installed.  If you don't have it, do:

```bash
python3 -m pip install --user cython
```

To run this exmaple, configure TAU with pthread, python and binutils support, for example, if python3 is installed in /packages/python/3.6.8:

```bash
./configure -python -pthread -bfd=/usr/local/packages/binutils/2.34 -pythoninc=/packages/python/3.6.8/include/python3.6m -pythonlib=/packages/python/3.6.8/lib
```

or

```bash
./configure -pthread -python -bfd=download -pythoninc=/packages/python/3.6.8/include/python3.6m -pythonlib=/packages/python/3.6.8/lib
```

...and then in this directory, run `make test`.  After the run, use the `pprof` program to display the output, and you should see the C function `hello_from_c_function` in the output, along with all the events captured by the Python profiling interface (this example was executed on a machine with 72 OpenMP threads):

```bash
Reading Profile files in profile.*

FUNCTION SUMMARY (total):
---------------------------------------------------------------------------------------
%Time    Exclusive    Inclusive       #Call      #Subrs  Inclusive Name
              msec   total msec                          usec/call
---------------------------------------------------------------------------------------
100.0          150       18,575          72          76     257993 .TAU application
 98.9       18,147       18,364          71          71     258658 gomp_thread_start [{/packages/gcc/src/build-8.1.0/x86_64-pc-linux-gnu/libgomp/../../../gcc-8.1.0/libgomp/team.c} {69, 0}]
  1.2          218          218          72           0       3037 hello_from_thread [{/storage/users/khuck/src/tau2/examples/cython-c-wrapper/lib/examples.c} {6,0}]
  0.3        0.124           48           1           1      48050 exec
  0.3        0.249           47           1           2      47926 <module> [{main.py}{1}]
  0.2            2           38           1           1      38056 py_hello_from_c
  0.2           18           35           1          72      35324 hello_from_c_function [{/storage/users/khuck/src/tau2/examples/cython-c-wrapper/lib/examples.c} {13,0}]
  0.1           15           15          71           0        212 pthread_create
  0.1         0.59            9           1           6       9621 _find_and_load [{<frozen importlib._bootstrap>}{966}]
  0.0         0.33            8           1           3       8026 _find_and_load_unlocked [{<frozen importlib._bootstrap>}{936}]
  0.0            1            7           1          21       7017 find_module [{imp.py}{255}]
  0.0        0.605            5           1           6       5158 _load_unlocked [{<frozen importlib._bootstrap>}{651}]
  0.0            4            4           1           0       4491 compile
  0.0        0.249            4           1           3       4108 module_from_spec [{<frozen importlib._bootstrap>}{564}]
  0.0        0.238            3           2           2       1688 _call_with_frames_removed [{<frozen importlib._bootstrap>}{211}]
  0.0        0.115            3           1           2       3335 create_module [{<frozen importlib._bootstrap_external>}{919}]
  0.0            3            3           1           1       3108 create_dynamic
  0.0        0.573            2           1           9       2536 _find_spec [{<frozen importlib._bootstrap>}{870}]
  0.0        0.278            2           6           7        372 isfile [{genericpath.py}{27}]
  0.0        0.522            2           2           2       1076 open
  0.0            2            2           8           0        252 stat
  0.0        0.114            1           1           1       1935 find_spec [{<frozen importlib._bootstrap_external>}{1149}]
  0.0        0.227            1           1           4       1821 _get_spec [{<frozen importlib._bootstrap_external>}{1117}]
  0.0        0.616            1           1           8       1515 search_function [{__init__.py}{71}]
  0.0        0.693            1           1           7       1476 find_spec [{<frozen importlib._bootstrap_external>}{1233}]
  0.0        0.217        0.859           1           2        859 __enter__ [{<frozen importlib._bootstrap>}{147}]
  0.0        0.247        0.658           1           4        658 detect_encoding [{tokenize.py}{355}]
  0.0        0.371        0.522           1           8        522 _init_module_attrs [{<frozen importlib._bootstrap>}{504}]
  0.0         0.36        0.515           1           3        515 _get_module_lock [{<frozen importlib._bootstrap>}{157}]
  0.0        0.453         0.51           7          23         73 join [{posixpath.py}{75}]
  0.0        0.398        0.407           1           3        407 get_suffixes [{imp.py}{105}]
  0.0        0.363        0.375           1          12        375 normalize_encoding [{__init__.py}{43}]
  0.0        0.225        0.355           1           2        355 _get_spec [{<frozen importlib._bootstrap_external>}{1228}]
  0.0        0.166        0.302           1           1        302 read
  0.0        0.128        0.264           1           1        264 getregentry [{utf_8.py}{33}]
  0.0        0.137        0.262           1           2        262 __exit__ [{<frozen importlib._bootstrap>}{318}]
  0.0        0.233        0.262           1           2        262 find_cookie [{tokenize.py}{385}]
  0.0        0.128        0.252           1           1        252 __import__
  0.0        0.119        0.234           1           2        234 _path_join [{<frozen importlib._bootstrap_external>}{57}]
  0.0        0.012        0.174           1           2        174 exec_module [{<frozen importlib._bootstrap_external>}{927}]
  0.0         0.15        0.152           1           2        152 __init__ [{<frozen importlib._bootstrap>}{58}]
  0.0        0.115        0.152           1           1        152 _path_isfile [{<frozen importlib._bootstrap_external>}{94}]
  0.0        0.109         0.14           1           1        140 read_or_stop [{tokenize.py}{379}]
  0.0        0.133        0.136           1           1        136 __new__ [{codecs.py}{93}]
  0.0        0.131        0.136           1           1        136 decode [{codecs.py}{318}]
  0.0        0.118        0.131           1           1        131 cached [{<frozen importlib._bootstrap>}{403}]
  0.0        0.114        0.129           1           1        129 __exit__ [{<frozen importlib._bootstrap>}{151}]
  0.0        0.126        0.127           1           1        127 acquire [{<frozen importlib._bootstrap>}{78}]
  0.0        0.119        0.126           1           2        126 spec_from_file_location [{<frozen importlib._bootstrap_external>}{524}]
  0.0        0.116        0.124           1           1        124 _handle_fromlist [{<frozen importlib._bootstrap>}{997}]
  0.0        0.119        0.123           1           4        123 any
  0.0        0.111        0.115           1           1        115 __init__ [{codecs.py}{308}]
  0.0        0.102        0.114           1           1        114 _path_importer_cache [{<frozen importlib._bootstrap_external>}{1080}]
  0.0        0.108        0.113           1           2        113 <listcomp> [{<frozen importlib._bootstrap_external>}{59}]
  0.0        0.006        0.067           2           2         34 _path_stat [{<frozen importlib._bootstrap_external>}{75}]
  0.0        0.005        0.037           1           1         37 _path_is_mode_type [{<frozen importlib._bootstrap_external>}{85}]
  0.0        0.031        0.031           2           0         16 match
  0.0        0.031        0.031           1           0         31 readline
  0.0         0.03         0.03           1           0         30 exec_dynamic
  0.0        0.013        0.027           7           7          4 _get_sep [{posixpath.py}{41}]
  0.0        0.027        0.027          12           0          2 isinstance
  0.0        0.019        0.019           6           0          3 getattr
  0.0        0.017        0.017           8           0          2 startswith
  0.0        0.015        0.015           8           0          2 fspath
  0.0        0.014        0.015           1           1         15 release [{<frozen importlib._bootstrap>}{103}]
  0.0        0.011        0.014           1           1         14 parent [{<frozen importlib._bootstrap>}{416}]
  0.0        0.009        0.013           1           2         13 _get_cached [{<frozen importlib._bootstrap_external>}{361}]
  0.0        0.013        0.013           4           0          3 _verbose_message [{<frozen importlib._bootstrap>}{222}]
  0.0        0.013        0.013           4           0          3 hasattr
  0.0        0.008        0.012           1           3         12 cb [{<frozen importlib._bootstrap>}{176}]
  0.0        0.012        0.012           1           0         12 getcwd
  0.0        0.007         0.01           3           3          3 __exit__ [{<frozen importlib._bootstrap>}{847}]
  0.0        0.005        0.008           3           3          3 __enter__ [{<frozen importlib._bootstrap>}{843}]
  0.0        0.008        0.008           4           0          2 endswith
  0.0        0.008        0.008           1           0          8 new_module [{imp.py}{48}]
  0.0        0.007        0.007           5           0          1 get
  0.0        0.006        0.006           5           0          1 acquire_lock
  0.0        0.003        0.006           1           1          6 find_spec [{<frozen importlib._bootstrap>}{707}]
  0.0        0.006        0.006           2           0          3 is_builtin
  0.0        0.006        0.006           3           0          2 rpartition
  0.0        0.005        0.005           1           0          5 <listcomp> [{imp.py}{107}]
  0.0        0.005        0.005           1           0          5 __init__ [{<frozen importlib._bootstrap>}{369}]
  0.0        0.005        0.005           5           0          1 append
  0.0        0.005        0.005           5           0          1 release_lock
  0.0        0.005        0.005           2           0          2 rstrip
  0.0        0.005        0.005           1           0          5 utf_8_decode
  0.0        0.004        0.004           4           0          1 <genexpr> [{<frozen importlib._bootstrap>}{321}]
  0.0        0.004        0.004           1           0          4 __enter__ [{<frozen importlib._bootstrap>}{311}]
  0.0        0.004        0.004           1           0          4 __init__ [{<frozen importlib._bootstrap>}{143}]
  0.0        0.004        0.004           1           0          4 __init__ [{<frozen importlib._bootstrap>}{307}]
  0.0        0.004        0.004           1           0          4 __init__ [{<frozen importlib._bootstrap_external>}{908}]
  0.0        0.004        0.004           1           0          4 __init__ [{codecs.py}{259}]
  0.0        0.002        0.004           1           1          4 find_spec [{<frozen importlib._bootstrap>}{780}]
  0.0        0.004        0.004           2           0          2 join
  0.0        0.003        0.003           1           0          3 __new__
  0.0        0.003        0.003           1           0          3 decode
  0.0        0.003        0.003           2           0          2 is_frozen
  0.0        0.003        0.003           5           0          1 isalnum
  0.0        0.003        0.003           1           0          3 replace
  0.0        0.002        0.002           1           0          2 <listcomp> [{imp.py}{108}]
  0.0        0.002        0.002           1           0          2 <listcomp> [{imp.py}{109}]
  0.0        0.002        0.002           1           0          2 S_ISREG
  0.0        0.002        0.002           2           0          1 allocate_lock
  0.0        0.002        0.002           2           0          1 get_ident
  0.0        0.001        0.001           1           0          1 _relax_case [{<frozen importlib._bootstrap_external>}{41}]
  0.0        0.001        0.001           1           0          1 has_location [{<frozen importlib._bootstrap>}{424}]

FUNCTION SUMMARY (mean):
---------------------------------------------------------------------------------------
%Time    Exclusive    Inclusive       #Call      #Subrs  Inclusive Name
              msec   total msec                          usec/call
---------------------------------------------------------------------------------------
100.0            2          257           1     1.05556     257993 .TAU application
 98.9          252          255    0.986111    0.986111     258658 gomp_thread_start [{/packages/gcc/src/build-8.1.0/x86_64-pc-linux-gnu/libgomp/../../../gcc-8.1.0/libgomp/team.c} {69, 0}]
  1.2            3            3           1           0       3037 hello_from_thread [{/storage/users/khuck/src/tau2/examples/cython-c-wrapper/lib/examples.c} {6,0}]
  0.3      0.00172        0.667   0.0138889   0.0138889      48050 exec
  0.3      0.00346        0.666   0.0138889   0.0277778      47926 <module> [{main.py}{1}]
  0.2       0.0379        0.529   0.0138889   0.0138889      38056 py_hello_from_c
  0.2        0.264        0.491   0.0138889           1      35324 hello_from_c_function [{/storage/users/khuck/src/tau2/examples/cython-c-wrapper/lib/examples.c} {13,0}]
  0.1        0.209        0.209    0.986111           0        212 pthread_create
  0.1      0.00819        0.134   0.0138889   0.0833333       9621 _find_and_load [{<frozen importlib._bootstrap>}{966}]
  0.0      0.00458        0.111   0.0138889   0.0416667       8026 _find_and_load_unlocked [{<frozen importlib._bootstrap>}{936}]
  0.0       0.0145       0.0975   0.0138889    0.291667       7017 find_module [{imp.py}{255}]
  0.0       0.0084       0.0716   0.0138889   0.0833333       5158 _load_unlocked [{<frozen importlib._bootstrap>}{651}]
  0.0       0.0624       0.0624   0.0138889           0       4491 compile
  0.0      0.00346       0.0571   0.0138889   0.0416667       4108 module_from_spec [{<frozen importlib._bootstrap>}{564}]
  0.0      0.00331       0.0469   0.0277778   0.0277778       1688 _call_with_frames_removed [{<frozen importlib._bootstrap>}{211}]
  0.0       0.0016       0.0463   0.0138889   0.0277778       3335 create_module [{<frozen importlib._bootstrap_external>}{919}]
  0.0        0.043       0.0432   0.0138889   0.0138889       3108 create_dynamic
  0.0      0.00796       0.0352   0.0138889       0.125       2536 _find_spec [{<frozen importlib._bootstrap>}{870}]
  0.0      0.00386        0.031   0.0833333   0.0972222        373 isfile [{genericpath.py}{27}]
  0.0      0.00725       0.0299   0.0277778   0.0277778       1076 open
  0.0        0.028        0.028    0.111111           0        252 stat
  0.0      0.00158       0.0269   0.0138889   0.0138889       1935 find_spec [{<frozen importlib._bootstrap_external>}{1149}]
  0.0      0.00315       0.0253   0.0138889   0.0555556       1821 _get_spec [{<frozen importlib._bootstrap_external>}{1117}]
  0.0      0.00856        0.021   0.0138889    0.111111       1515 search_function [{__init__.py}{71}]
  0.0      0.00962       0.0205   0.0138889   0.0972222       1476 find_spec [{<frozen importlib._bootstrap_external>}{1233}]
  0.0      0.00301       0.0119   0.0138889   0.0277778        859 __enter__ [{<frozen importlib._bootstrap>}{147}]
  0.0      0.00343      0.00914   0.0138889   0.0555556        658 detect_encoding [{tokenize.py}{355}]
  0.0      0.00515      0.00725   0.0138889    0.111111        522 _init_module_attrs [{<frozen importlib._bootstrap>}{504}]
  0.0        0.005      0.00715   0.0138889   0.0416667        515 _get_module_lock [{<frozen importlib._bootstrap>}{157}]
  0.0      0.00629      0.00708   0.0972222    0.319444         73 join [{posixpath.py}{75}]
  0.0      0.00553      0.00565   0.0138889   0.0416667        407 get_suffixes [{imp.py}{105}]
  0.0      0.00504      0.00521   0.0138889    0.166667        375 normalize_encoding [{__init__.py}{43}]
  0.0      0.00313      0.00493   0.0138889   0.0277778        355 _get_spec [{<frozen importlib._bootstrap_external>}{1228}]
  0.0      0.00231      0.00419   0.0138889   0.0138889        302 read
  0.0      0.00178      0.00367   0.0138889   0.0138889        264 getregentry [{utf_8.py}{33}]
  0.0       0.0019      0.00364   0.0138889   0.0277778        262 __exit__ [{<frozen importlib._bootstrap>}{318}]
  0.0      0.00324      0.00364   0.0138889   0.0277778        262 find_cookie [{tokenize.py}{385}]
  0.0      0.00178       0.0035   0.0138889   0.0138889        252 __import__
  0.0      0.00165      0.00325   0.0138889   0.0277778        234 _path_join [{<frozen importlib._bootstrap_external>}{57}]
  0.0     0.000167      0.00242   0.0138889   0.0277778        174 exec_module [{<frozen importlib._bootstrap_external>}{927}]
  0.0      0.00208      0.00211   0.0138889   0.0277778        152 __init__ [{<frozen importlib._bootstrap>}{58}]
  0.0       0.0016      0.00211   0.0138889   0.0138889        152 _path_isfile [{<frozen importlib._bootstrap_external>}{94}]
  0.0      0.00151      0.00194   0.0138889   0.0138889        140 read_or_stop [{tokenize.py}{379}]
  0.0      0.00185      0.00189   0.0138889   0.0138889        136 __new__ [{codecs.py}{93}]
  0.0      0.00182      0.00189   0.0138889   0.0138889        136 decode [{codecs.py}{318}]
  0.0      0.00164      0.00182   0.0138889   0.0138889        131 cached [{<frozen importlib._bootstrap>}{403}]
  0.0      0.00158      0.00179   0.0138889   0.0138889        129 __exit__ [{<frozen importlib._bootstrap>}{151}]
  0.0      0.00175      0.00176   0.0138889   0.0138889        127 acquire [{<frozen importlib._bootstrap>}{78}]
  0.0      0.00165      0.00175   0.0138889   0.0277778        126 spec_from_file_location [{<frozen importlib._bootstrap_external>}{524}]
  0.0      0.00161      0.00172   0.0138889   0.0138889        124 _handle_fromlist [{<frozen importlib._bootstrap>}{997}]
  0.0      0.00165      0.00171   0.0138889   0.0555556        123 any
  0.0      0.00154       0.0016   0.0138889   0.0138889        115 __init__ [{codecs.py}{308}]
  0.0      0.00142      0.00158   0.0138889   0.0138889        114 _path_importer_cache [{<frozen importlib._bootstrap_external>}{1080}]
  0.0       0.0015      0.00157   0.0138889   0.0277778        113 <listcomp> [{<frozen importlib._bootstrap_external>}{59}]
  0.0     8.33E-05     0.000931   0.0277778   0.0277778         34 _path_stat [{<frozen importlib._bootstrap_external>}{75}]
  0.0     6.94E-05     0.000514   0.0138889   0.0138889         37 _path_is_mode_type [{<frozen importlib._bootstrap_external>}{85}]
  0.0     0.000431     0.000431   0.0277778           0         16 match
  0.0     0.000431     0.000431   0.0138889           0         31 readline
  0.0     0.000417     0.000417   0.0138889           0         30 exec_dynamic
  0.0     0.000181     0.000375   0.0972222   0.0972222          4 _get_sep [{posixpath.py}{41}]
  0.0     0.000375     0.000375    0.166667           0          2 isinstance
  0.0     0.000264     0.000264   0.0833333           0          3 getattr
  0.0     0.000236     0.000236    0.111111           0          2 startswith
  0.0     0.000208     0.000208    0.111111           0          2 fspath
  0.0     0.000194     0.000208   0.0138889   0.0138889         15 release [{<frozen importlib._bootstrap>}{103}]
  0.0     0.000153     0.000194   0.0138889   0.0138889         14 parent [{<frozen importlib._bootstrap>}{416}]
  0.0     0.000125     0.000181   0.0138889   0.0277778         13 _get_cached [{<frozen importlib._bootstrap_external>}{361}]
  0.0     0.000181     0.000181   0.0555556           0          3 _verbose_message [{<frozen importlib._bootstrap>}{222}]
  0.0     0.000181     0.000181   0.0555556           0          3 hasattr
  0.0     0.000111     0.000167   0.0138889   0.0416667         12 cb [{<frozen importlib._bootstrap>}{176}]
  0.0     0.000167     0.000167   0.0138889           0         12 getcwd
  0.0     9.72E-05     0.000139   0.0416667   0.0416667          3 __exit__ [{<frozen importlib._bootstrap>}{847}]
  0.0     6.94E-05     0.000111   0.0416667   0.0416667          3 __enter__ [{<frozen importlib._bootstrap>}{843}]
  0.0     0.000111     0.000111   0.0555556           0          2 endswith
  0.0     0.000111     0.000111   0.0138889           0          8 new_module [{imp.py}{48}]
  0.0     9.72E-05     9.72E-05   0.0694444           0          1 get
  0.0     8.33E-05     8.33E-05   0.0694444           0          1 acquire_lock
  0.0     4.17E-05     8.33E-05   0.0138889   0.0138889          6 find_spec [{<frozen importlib._bootstrap>}{707}]
  0.0     8.33E-05     8.33E-05   0.0277778           0          3 is_builtin
  0.0     8.33E-05     8.33E-05   0.0416667           0          2 rpartition
  0.0     6.94E-05     6.94E-05   0.0138889           0          5 <listcomp> [{imp.py}{107}]
  0.0     6.94E-05     6.94E-05   0.0138889           0          5 __init__ [{<frozen importlib._bootstrap>}{369}]
  0.0     6.94E-05     6.94E-05   0.0694444           0          1 append
  0.0     6.94E-05     6.94E-05   0.0694444           0          1 release_lock
  0.0     6.94E-05     6.94E-05   0.0277778           0          3 rstrip
  0.0     6.94E-05     6.94E-05   0.0138889           0          5 utf_8_decode
  0.0     5.56E-05     5.56E-05   0.0555556           0          1 <genexpr> [{<frozen importlib._bootstrap>}{321}]
  0.0     5.56E-05     5.56E-05   0.0138889           0          4 __enter__ [{<frozen importlib._bootstrap>}{311}]
  0.0     5.56E-05     5.56E-05   0.0138889           0          4 __init__ [{<frozen importlib._bootstrap>}{143}]
  0.0     5.56E-05     5.56E-05   0.0138889           0          4 __init__ [{<frozen importlib._bootstrap>}{307}]
  0.0     5.56E-05     5.56E-05   0.0138889           0          4 __init__ [{<frozen importlib._bootstrap_external>}{908}]
  0.0     5.56E-05     5.56E-05   0.0138889           0          4 __init__ [{codecs.py}{259}]
  0.0     2.78E-05     5.56E-05   0.0138889   0.0138889          4 find_spec [{<frozen importlib._bootstrap>}{780}]
  0.0     5.56E-05     5.56E-05   0.0277778           0          2 join
  0.0     4.17E-05     4.17E-05   0.0138889           0          3 __new__
  0.0     4.17E-05     4.17E-05   0.0138889           0          3 decode
  0.0     4.17E-05     4.17E-05   0.0277778           0          2 is_frozen
  0.0     4.17E-05     4.17E-05   0.0694444           0          1 isalnum
  0.0     4.17E-05     4.17E-05   0.0138889           0          3 replace
  0.0     2.78E-05     2.78E-05   0.0138889           0          2 <listcomp> [{imp.py}{108}]
  0.0     2.78E-05     2.78E-05   0.0138889           0          2 <listcomp> [{imp.py}{109}]
  0.0     2.78E-05     2.78E-05   0.0138889           0          2 S_ISREG
  0.0     2.78E-05     2.78E-05   0.0277778           0          1 allocate_lock
  0.0     2.78E-05     2.78E-05   0.0277778           0          1 get_ident
  0.0     1.39E-05     1.39E-05   0.0138889           0          1 _relax_case [{<frozen importlib._bootstrap_external>}{41}]
  0.0     1.39E-05     1.39E-05   0.0138889           0          1 has_location [{<frozen importlib._bootstrap>}{424}]
```
