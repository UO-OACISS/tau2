*****************************************************************************
**                      Tuning and Analysis Utilities                      **
**			http://www.cs.uoregon.edu/research/paracomp/tau    **
*****************************************************************************
**    Copyright 1997-2011                                                  **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
**    Research Center Juelich, ZAM Germany                                 **
*****************************************************************************

To use TAU with JAVA, you'll need Java 5+ (jdk1.5+).

1) Configuration
% ./configure -jdk=/usr/local/packages/jdk1.6 

2) Installing
% make install

3) Running an example
% set path=($path <taudir>/<tauarch>/bin)
% setenv LD_LIBRARY_PATH $LD_LIBRARY_PATH:<taudir>/<tauarch>/lib
% cd examples/java/pi
% tau_java Pi 200000
(java -version shows the version of JVM).

% tau_java

    -tau:agentlib=<agentlib> By default tau_java uses the most recently configured jdk, you can specify a different one here.
    -tau:java=<javapath>     Path to a java binary, by default uses the one corresponding to the most recently configured jdk.
    -tau:bootclasspath=<bootclasspath>    To modify the bootclasspath to point to a different jar, not usually necessary.
    -tau:include=<item>      Only these classes/methods
    -tau:exclude=<item>      Exclude these classes/methodsn
    item    Qualified class and/or method names
       e.g. (*.<init>;Foobar.method;sun.*)

5) Displaying profiles/traces
For Profiling:
% pprof
% paraprof 
pprof is for text profile display and paraprof is the GUI.

For Tracing:
% setenv TAU_TRACE 1
First merge the traces using tau_merge
% tau_treemerge.pl
Then convert to Vampir trace format
% tau2otf tau.trc tau.edf Pi.otf

% vampir Pi.otf 
See TAU user's Guide for further details on using paraprof and Vampir. 

For any comments/suggestions/questions on TAU, please send mail to 
tau-bugs@cs.uoregon.edu


