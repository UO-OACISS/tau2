This directory contains an example of the use of the TAU java dlls

To use:

make sure you PATH contains $TAU_ROOT$\JavaDLL
(e.g. PATH=%PATH%;c:\tau-2.14.2.1\JavaDLL)

Then run the following:

java -XrunTau-profile Pi

then run pprof to examine the output (in the case of profiles)
or analyze the trace output.
