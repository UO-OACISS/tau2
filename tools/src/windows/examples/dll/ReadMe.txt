This Project contains an example of using the TAU api using 
the shared object (dll).  This example builds a dll that uses tau,
and another instrumented source file that uses TAU and that dll.

This shows that there is a single runtime of TAU in the TAU dll.

use 'nmake' to build

You will have to fix the lib directory to the proper subdirectory 
based on your compiler (vc6, vc7, vc8)


