*****************************************************************************
**                      Tuning and Analysis Utilities                      **
**                      http://www.acl.lanl.gov/tau                        **
*****************************************************************************
**    Copyright 1997-99                                                    **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
**    Research Center Juelich, ZAM Germany                                 **
*****************************************************************************

To use TAU with JAVA, you'll need Java 2 (jdk1.2).

1) Configuration
For profiling
% ./configure -c++=g++ -jdk=/usr/local/packages/jdk1.2 
For tracing
% ./configure -c++=g++ -jdk=/usr/local/packages/jdk1.2  -TRACE
For both profiling and tracing
% ./configure -c++=g++ -jdk=/usr/local/packages/jdk1.2  -TRACE -PROFILE

2) Installing
% make install

3) Running an example
% set path=($path <taudir>/<tauarch>/bin)
% setenv LD_LIBRARY_PATH $LD_LIBRARY_PATH:<taudir>/<tauarch>/lib
% cd examples/java/pi
% java -XrunTAU Pi 200000
On Solaris, you may need to disable the JIT compiler (it works on Linux) while
using TAU. 
% java -XrunTAU -Djava.compiler=  Pi 200000

4) Displaying profiles/traces
For Profiling:
% pprof
% racy 
pprof is for text profile display and racy is the GUI.
When the project management window comes up, type in pi.pmf and click on Create.

For Tracing:
First merge the traces using tau_merge
% tau_merge tautrace*.trc Pi.trc
Then convert to Vampir trace format
% tau_convert -vampir Pi.trc tau.edf Pi.pv

Download and install Vampir (free evaluation license typically lasts for a 
month) from http://www.pallas.de.
% vampir Pi.pv 
Then assign colors to the groups by (Preferences -> Colors -> Activities).
See TAU user's Guide for further details on using racy and Vampir. 

For any comments/suggestions/questions on TAU, please send mail to 
sameer@cs.uoregon.edu


