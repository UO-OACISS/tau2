*****************************************************************************
**                      Tuning and Analysis Utilities                      **
**			http://www.cs.uoregon.edu/research/paracomp/tau    **
*****************************************************************************
**    Copyright 1997-2010                                                  **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
**    Research Center Juelich, ZAM Germany                                 **
*****************************************************************************

To use TAU with JAVA, you'll need Java 2 (jdk1.2+).

1) Configuration
For profiling
% ./configure -c++=g++ -jdk=/usr/local/packages/jdk1.2 
For tracing
% ./configure -c++=g++ -jdk=/usr/local/packages/jdk1.2  -TRACE
For both profiling and tracing
% ./configure -c++=g++ -jdk=/usr/local/packages/jdk1.4.x  -TRACE -PROFILE

2) Installing
% make install

3) Running an example
% set path=($path <taudir>/<tauarch>/bin)
% setenv LD_LIBRARY_PATH $LD_LIBRARY_PATH:<taudir>/<tauarch>/lib
% cd examples/java/pi
% java -XrunTAU Pi 200000
(java -version shows the version of JVM).

4) Selective instrumentation
To exclude certain classes, you may use the -XrunTAU:exclude=<classes> flag.
e.g., to exclude java/*, sun/*, com/* methods, you may use:
% java -XrunTAU:exclude=java,sun,com Pi 200000

To exclude all methods (and *not* incur the JVMPI method invocation overhead), you may use the -XrunTAU:nomethods flag as:
% java -XrunTAU:nomethods Pi 20000
This allows you to use source level timers (see examples/java/api for an example) and generate accurate multi-threaded performance data without the JVMPI overhead.

5) Displaying profiles/traces
For Profiling:
% pprof
% paraprof 
pprof is for text profile display and paraprof is the GUI.

For Tracing:
First merge the traces using tau_merge
% tau_merge tautrace*.trc Pi.trc
Then convert to Vampir trace format
% tau_convert -vampir Pi.trc tau.edf Pi.pv

Download and install Vampir (free evaluation license typically lasts for a 
month) from http://www.pallas.de.
% vampir Pi.pv 
Then assign colors to the groups by choosing the menu option 
Preferences -> Colors -> Activities.
See TAU user's Guide for further details on using racy and Vampir. 

For any comments/suggestions/questions on TAU, please send mail to 
tau-bugs@cs.uoregon.edu


