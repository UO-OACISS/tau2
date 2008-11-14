
TAUPIN: 
------
 
This is a TAU tool to enable profiling applications directly from the command line. Right now it works for 
Windows on Intel IA-32 architectures as of now. It instruments the application binary with PIN 
apis. Besides the application binary it also needs the PDB file corresponding to the EXE. Invoking this tool without
PDB will give no profile data.



Usage:
------

tau_pin [-n proc_num] [-r rules_1 -r rule_2 ...] -- myapp myargs 

-n proc_num: This argument enables multple instances of MPI applications launched with MPIEXEC. 
             proc_num is the parameter indicating number of MPI process instances to be launched. 
	     This argument is optional and one can profile MPI application even with single process 
             instance without this argument. 

 
-r rule    : This argument is specification rule for profiling the application. It allows selective profiling
	     by specifying the "rule". The rule is a wildcard expression token which will indicate the area of 
             profiling. It can be only the routine specification like "*" which indicates it'll instrument all the 
	     routines in the EXE or MPI routines. One can further specify the routines on a particular dll by 
             the rule "somedll.dll!*". The dll name can also be in regular expression. We treat the application exe
	     and MPI routines as special cases and specifying only the routines is allowed.     


myapp      : It's the application exe. This application can be Windows or console application. Profiling large 
             Windows applications might suffer from degraded performance and interactability. Specifying a limited 
	     number of interesting routines can help.    


myargs     : It's the command line arguments of the application.  
	 
	
Wild Cards Supported:
---------------------

1) * - for anything , for example *MPI* means any string having MPI in between

2) ? - It's a placeholder wild card ?MPI* means any character followed by MPI and followed by any string  
	Example: ??Try could be __Try or MyTry or MeTry etc.  
     
Examples:
--------                  

 
-Profiling routines in mytest.exe with prefix "myf"
	tauprofile -r myf.*  -- mytest.exe
 
-Profiling all routines in mpitest.exe ( no need to specify any rule for all ) 
	tauprofile  mpitest.exe

-Profiling only MPI routines in mpitest.exe by launching two instances

	tauprofile -n 2 -r _MPI_.* -- mpitest.exe

 

 
