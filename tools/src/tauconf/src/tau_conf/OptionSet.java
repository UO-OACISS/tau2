/*
 * OptionSet.java
 *
 * Created on August 2, 2007, 4:44 PM
 *
 * To change this template, choose Tools | Template Manager
 * and open the template in the editor.
 */

package tau_conf;

import javax.swing.JButton;
import javax.swing.JCheckBox;

/**
 *
 * @author wspear
 */
public class OptionSet {
    
    /** Creates a new instance of OptionSet */
    private OptionSet() {
    }
    
    protected static final TAUOption[] options={
        //COMPILERS
        new TAUOption("-c++","<C++ compiler>","Specifies the name of the C++ compiler. Supported  C++ compilers include KCC (from KAI/Intel), CC,  g++ and power64-linux-g++ (from GNU), FCC (from Fujitsu), xlC(from IBM), guidec++ (from KAI/Intel), aCC (from HP), c++ (from Apple), and pgCC (from PGI)."),
        new TAUOption("-cc","<C Compiler>","Specifies the name of the C compiler. Supported C compilers include cc, gcc and powerpc64-linux-gcc (from GNU), pgcc (from PGI), fcc (from Fujitsu), xlc (from IBM), and KCC (from KAI/Intel)."),
        new TAUOption("-fortran","<Fortran Compiler>","Specifies the name of the Fortran90 compiler. Valid options are: gnu, sgi, ibm, ibm64, hp, cray, pgi, absoft, fujitsu, sun, compaq, nec, hitachi, kai, absoft, lahey, nagware, and intel."),
        new TAUOption("-pdt","<directory>","Specifies the location of the installed PDT (Program Database Toolkit) root directory. PDT is used to build tau_instrumentor, a C++, C and F90 instrumentation program that automatically inserts TAU annotations in the source code. If PDT is configured with a subdirectory option (-compdir=<opt>) then TAU can be configured with the same option by specifying -pdt=<dir> -pdtcompdir=<opt>. \n[http://www.cs.uoregon.edu/research/paracomp/pdtoolkit/]"),
        new TAUOption("-pdt_c++","<C++ Compiler>","Specifies a different C++ compiler for PDT (tau_instrumentor). This is typically used when the library is compiled with a C++ compiler (specified with -c++) and the tau_instrumentor is compiled with a different <pdt_c++> compiler. For e.g., -c++=pgCC -cc=pgcc -pdt_c++=KCC -openmp ... uses PGI's OpenMP compilers for TAU's library and KCC for tau_instrumentor."),
        new TAUOption("-papi","<directory>","Specifies the location of the installed PAPI (Performance API) root directory. PAPI specifies a standard application programming interface (API) for accessing hardware performance counters available on most modern microprocessors similar. To measure floating point instructions, set the environment variable PAPI_EVENT to PAPI_FP_INS (for example). Refer to the TAU User's Guide or PAPI Documentation for other event names.\n[Ref : http://icl.cs.utk.edu/projects/papi/api/]"),
        new TAUOption("-PAPIWALLCLOCK","","Uses PAPI (must specify -papi=<dir> also) to access high resolution CPU timers for wallclock time. The default case uses gettimeofday() which has a higher overhead than this."),
        new TAUOption("-PAPIVIRTUAL","","Uses PAPI (must specify -papi=<dir> also) to access process virtual time. This represents the user time for measurements."),
        new TAUOption("-MULTIPLECOUNTERS","","Allows TAU to track more than one quantity (multiple hardware counters, CPU time, wallclock time, etc.) Configure with other options such as -papi=<dir>, -pcl=<dir>, -LINUXTIMERS, -SGITIMERS, -CRAYTIMERS, -CPUTIME, -PAPIVIRTUAL, etc. See examples/multicounters/README file for detailed instructions on setting the environment variables for this option. If -MULTIPLECOUNTERS is used with the -TRACE option, tracing employs the COUNTER1 variable for wallclock time."),
        //MESSAGE PASSING
        new TAUOption("-mpi","","Specifies use of the TAU MPI wrapper library."),
        new TAUOption("-mpiinc","<dir>","Specifies the directory  where mpi header files reside (such as mpi.h and mpif.h). This option also generates the TAU MPI wrapper library that instruments MPI routines using the MPI Profiling Interface. See the examples/NPB2.3/config/make.def file for its usage with Fortran and MPI programs and examples/pi/Makefile for a C++ example that uses MPI."),
        new TAUOption("-mpilib","<dir>","Specifies the directory where mpi library files reside. This option should be used in conjunction with the -mpiinc=<dir> option to generate the TAU MPI wrapper library."),
        new TAUOption("-mpilibrary","<lib>","Specifies the directory where mpi library files reside. This option should be used in conjunction with the -mpiinc=<dir> option to generate the TAU MPI wrapper library."),
        new TAUOption("-tag","<Unique Name>","Specifies a tag in the name of the stub Makefile and TAU makefiles to uniquely identify the installation. This is useful when more than one MPI library may be used with different versions of compilers.  e.g., % configure -c++=icpc -cc=icc -tag=intel71-vmi -mpiinc=/vmi2/mpich/include"),
        new TAUOption("-nocomm","","Allows the user to turn off tracking of messages (synchronous/asynchronous) in TAU's MPI wrapper interposition library. Entry and exit events for MPI routines are still tracked. Affects both profiling and tracing."),
        new TAUOption("-MPITRACE","","Specifies the tracing option and generates event traces for MPI calls and routines that are ancestors of MPI calls in the callstack. This option is useful for generating traces that are converted to the EPILOG trace format.  KOJAK's Expert automatic diagnosis tool needs traces with events that call MPI routines. Do not use this option with the -TRACE option."),
        new TAUOption("-shmem","","Specifies use of the TAU SHMEM wrapper library."),
        new TAUOption("-shmeminc","<dir>","Specifies the directory where shmem.h resides. Specifies the use of the TAU SHMEM interface."),
        new TAUOption("-shmemlib","<lib>","Specifies the directory where libsma.a resides. Specifies the use of the TAU SHMEM interface."),
        new TAUOption("-shmemlibrary","<lib>","By default, TAU uses -lsma as the shmem/pshmem library. This option allows the user to specify a different shmem library."),
        //TRACING/PROFILING
        new TAUOption("-PROFILE","","This is the default option; it specifies summary profile files to be generated at the end of execution. Profiling generates aggregate statistics (such as the total time spent in routines and statements), and can be used in conjunction with the profile browser paraprof to analyse the performance.  Wallclock time is used for profiling  program entities."),
        new TAUOption("-COMPENSATE","","Specifies online compensation of performance perturbation. When this option is used, TAU computes its overhead and subtracts it from the profiles. It can be only used when profiling is chosen. This option works with MULTIPLECOUNTERS as well, but while it is relevant for removing perturbation with wallclock time, it cannot accurately account for perturbation with hardware performance counts (e.g., L1 Data cache misses).  See TAU Publication [Europar04] for further information on this option."),
        new TAUOption("-PROFILECALLPATH","","This option generates call path profiles which shows the time spent in a routine when it is called by another routine in the calling path. \"a => b\" stands for the time spent in routine \"b\" when it is invoked by routine \"a\".  This option is an extension of -PROFILE, the default profiling option.  Specifying TAU_CALLPATH_DEPTH environment variable, the user can vary the depth of the callpath. See examples/calltree for further information."),
        new TAUOption("-PROFILEHEADROOM","","Specifies tracking memory available in the heap (as opposed to memory utilization tracking in -PROFILEMEMORY). When any function entry takes place, a sample of the memory available (headroom to grow) is taken. This data is stored as user-defined event data in profiles/traces. Please refer to the examples/headroom/README file for a full explanation of these headroom options and the C++/C/F90 API for evaluating the headroom."),
        new TAUOption("-PROFILEMEMORY","","Specifies tracking heap memory utilitization for each instrumented function.  When any function entry takes place, a sample of the heap memory used is taken. This data is stored as user-defined event data in profiles/traces."),
        new TAUOption("-TRACE","","Generates event-trace logs, rather than summary profiles. Traces show when and where an event occurred, in terms of the location in the source code and the process that executed it. Traces can be merged and converted using tau_merge and tau_convert utilities respectively, and  visualized using Vampir, a commercial trace visualization tool. [ Ref http://www.pallas.de ]"),
        new TAUOption("-epilog","<dir>","Specifies the directory where the EPILOG tracing package [FZJ] is installed.  This option should be used in conjunction with the -TRACE option to generate binary EPILOG traces (instead of binary TAU traces). EPILOG traces can then be used with other tools such as EXPERT. EPILOG comes with its own implementation of the MPI wrapper library and the POMP library used with Opari. Using option overrides TAU's libraries for MPI, and OpenMP."),
        new TAUOption("-epilogbin","<dir>","Specifies the absolute location of the epilog bin directory."),
        new TAUOption("-epiloginc","<dir>","Specifies the absolute location of the epilog include directory."),
        new TAUOption("-epiloglib","<dir>","Specifies the absolute location of the epilog lib directory."),
        new TAUOption("-slog2","","Specifies the use of the SLOG2 trace generation package and the Jumpshot trace visualizer that is bundled with TAU. Jumpshot v4 and SLOG2 v1.2.5delta are included in the TAU distribution. When the -slog2 flag is specified, tau2slog2 and jumpshot tools are copied to the <tau>/<arch>/<bin> directory.  It is important to have a working javac and java (preferably v1.4+) in your path. On linux systems, where /usr/bin/java may be a place holder, you'll need to modify your path accordingly."),
        new TAUOption("-slog2=","<dir>","Specifies the location of the SLOG2 SDK trace generation package. TAU's binary traces can be converted to the SLOG2 format using tau2slog2, a tool that uses the SLOG2 SDK. The SLOG2 format is read by the Jumpshot4 trace visualization software, a freely available trace visualizer from Argonne National Laboratories.\n[Ref: http://www-unix.mcs.anl.gov/perfvis/download/index.htm#slog2sdk]"),
        new TAUOption("-vtf","<dir>","Specifies the location of the VTF3 trace generation package. TAU's binary traces can be converted to the VTF3 format using tau2vtf, a tool that links with the VTF3 library. The VTF3 format is read by Intel trace analyzer, formerly known as vampir, a commercial trace visualization tool developed by TU. Dresden, Germany."),
        new TAUOption("-otf","<dir>","Specifies the location of the Open Trace Format (OTF) package."),
        new TAUOption("-vampirtrace","<dir>","Specifies the location of the VampirTrace package.  This allows TAU to output .otf trace files directly.  These can be read in vampir without additional merging or conversion."),
        new TAUOption("-PROFILEPHASE","","This option generates phase based profiles. It requires special instrumentation to mark phases in an application (I/O, computation, etc.).  Phases can be static or dynamic (different phases for each loop iteration, for instance). See examples/phase/README for further information."),
        new TAUOption("-DEPTHLIMIT","","Allows users to enable instrumentation at runtime based on the depth of a calling routine on a callstack. The depth is specified using the environment variable TAU_DEPTH_LIMIT. When its value is 1, instrumentation in the top-level routine such as main (in C/C++) or program (in F90) is activated.  When it is 2, only routine invoked directly by main and main are recorded.  When a routine appears at a depth of 2 and at 10 and we set the limit at 5, then the routine is recorded when its depth is 2, and ignored when its depth is 10 on the calling stack. This can be used with -PROFILECALLPATH to generate a tree of height <h> from the main routine by setting TAU_CALLPATH_DEPTH and TAU_DEPTH_LIMIT variables to <h>."),
        new TAUOption("-perfinc","<dir>","Specifies the directory  where Perflib header files reside.  This option also generates the TAU Perflib library that generates Perflib performance data."),
        new TAUOption("-perflib","<dir>","Specifies the directory where Perflib library files reside. This option should be used in conjunction with the -perfinc=<dir> option to generate the TAU Perflib library."),
        new TAUOption("-perflibrary","<lib>","Specifies the directory where perf library files reside. This option should be used in conjunction with the -perfinc=<dir> option to generate the TAU Perflib library."),
        //THREADS
        new TAUOption("-pthread","","Specifies pthread as the thread package to be used. In the default mode, no thread package is used."),
        new TAUOption("-openmp","","Specifies OpenMP as the threads package to be used. [Ref: http://www.openmp.org]"),
        new TAUOption("-opari","<dir>","Specifies the location of the Opari OpenMP directive rewriting tool.  The use of Opari source-to-source instrumentor in conjunction with TAU exposes OpenMP events for instrumentation. See examples/opari directory. [ Ref: http://www.fz-juelich.de/zam/kojak/opari/ ] the newer KOJAK - kojak-<ver>.tar.gz opari/ directory. Please upgrade to the KOJAK version (especially if you're using IBM xlf90) and specify -opari=<kojak-dir>/opari while configuring TAU."),
        new TAUOption("-opari_region","","Report performance data for only OpenMP regions and not constructs.  By default, both regions and constructs are profiled with Opari."),
        new TAUOption("-opari_construct","", "Report performance data for only OpenMP constructs and not regions.  By default, both regions and constructs are profiled with Opari."),
        new TAUOption("-charm","<dir>","Specifies charm++ (converse) threads as the thread package to be used."),
        new TAUOption("-sproc","","Use the SGI sproc thread package."),
        new TAUOption("-tulipthread","<dir>","Specifies Tulip threads (HPC++) as the threads package to be used as well [ Ref: http://www.acl.lanl.gov/tulip ]"),
        new TAUOption("-smarts","","Specifies  SMARTS (Shared Memory Asynchronous Runtime System) as the threads package to be used. <directory> gives the location of the SMARTS root directory. [ Ref: http://www.acl.lanl.gov/smarts ]"),
        //DATA TOOLS
        new TAUOption("-pcl","<dir>","Specifies the location of the installed PCL (Performance Counter Library) root directory. PCL provides a common interface to access hardware performance counters on modern microprocessors. The library supports Sun UltraSparc I/II, PowerPC 604e under AIX, MIPS R10000/12000 under IRIX, HP/Compaq Alpha 21164, 21264 under Tru64 Unix and Cray Unicos (T3E) and the Intel Pentium family of microprocessors under Linux. This option specifies the use of hardware performance counters for profiling (instead of time).  To measure floating point instructions, set the environment variable PCL Documentation (pcl.h) for other event names. [ Ref : http://www.fz-juelich.de/zam/PCL ]"),
        new TAUOption("-dyninst","<dir>","Specifies the location of the DynInst (dynamic instrumentation) package.  See README.DYNINST for instructions on using TAU with DynInstAPI for binary runtime instrumentation (instead of manual instrumentation) or prior to execution by rewriting it. [Ref: http://www.cs.umd.edu/projects/dyninstAPI/]"),
        new TAUOption("-muse","","Specifies the use of MAGNET/MUSE to extract low-level information from the kernel. To use this configuration, Linux kernel has to be patched with MAGNET and MUSE has to be install on the executing machine.  Also, magnetd has to be running with the appropriate handlers and filters installed. User can specify package by setting the environment variable TAU_MUSE_PACKAGE.  By default it uses the \"count\". Please refer to README.MUSE for more information."),
        new TAUOption("-muse_event","","Specifies use of the MUSE/MAGNET library with non-monotonically increasing values."),
        new TAUOption("-muse_multiple","","Specifies use of the MUSE/MAGNET library with monotonically increasing values."),
        new TAUOption("-CPUTIME","","Uses usertime + system time instead of wallclock time. It gives the CPU time spent in the routines.  This currently works only on LINUX systems for multi-threaded programs and on all systems for single-threaded programs."),
        new TAUOption("-CRAYTIMERS","","Specifies use of the free running nanosecond resolution on-chip timer on the CRAY X1 cpu (accessed by the rtc() syscall). This timer has a significantly lower overhead than the default timer on the X1, and is recommended for profiling. Since this timer is not synchronized across different cpus, this option should not be used with the -TRACE option for tracing a multi-cpu application, where a globally synchronized realtime clock is required."),
        new TAUOption("-LINUXTIMERS","","Specifies the use of the free running nanosecond resolution time stamp counter (TSC) on Pentium III+ and Itanium family of processors under Linux.  This timer has a lower overhead than the default time and is recommended."),
        new TAUOption("-SGITIMERS","","Specifies use of the free running nanosecond resolution on-chip timer on the MIPS R10000. This timer has a lower overhead than the default timer on SGI, and is recommended for SGIs."),
        //MISC
        new TAUOption("-pythoninc","<dir>","Specifies the location of the Python include directory. This is the directory where Python.h header file is located. This option enables python bindings to be generated. The user should set the environment variable PYTHONPATH to <TAUROOT>/<ARCH>/lib/bindings-<options> to use a specific version of the TAU Python bindings. By importing package pytau, a user can manually instrument the source code and use the TAU API. On the other hand, by importing tau and using tau.run('<func>'), TAU can automatically generate instrumentation. See examples/python directory for further information."),
        new TAUOption("-pythonlib","<dir>","Specifies the location of the Python lib directory. This is the directory where *.py and *.pyc files (and config directory) are located. This option is mandatory for IBM when Python bindings are used. For other systems, this option may not be specified (but -pythoninc=<dir> needs to be specified)."),
        new TAUOption("-jdk","<dir>","Specifies the location of the Java 2 development kit (jdk1.2+). See README.JAVA on instructions on using TAU with Java 2 applications.  This option should only be used for configuring TAU to use JVMPI for profiling and tracing of Java applications. It should not be used for configuring paraprof, which uses java from the user's path."),
        new TAUOption("-JAVACPUTIME","","Use JVMPI thread specific cpu time."),
        new TAUOption("-prefix","<dir>","Specifies the destination directory where the header, library and binary files are copied. By default, these are copied to subdirectories <arch>/bin and <arch>/lib in the TAU root directory."),
        new TAUOption("-exec-prefix","<dir>","Specifies an alternate architecture directory name to be generated in the TAU root directory. By default, the <arch> directory is named in accordance with the detected or specified architecture.  This function allows multiple alternate configurations to be generated on the same architecture."),
        new TAUOption("-arch","<architecture>","Specifies the architecture. If the user does not specify this option, configure determines the architecture. For SGI, the user can specify either of sgi32, sgin32 or sgi64 for 32, n32 or 64 bit compilation modes respectively. The files are installed in the <architecture>/bin and <architecture>/lib directories.<architecture>/lib directories."),
        new TAUOption("-useropt","<options-list>","Specifies additional user options such as -g or -I.  For multiple options, the options list should be enclosed in a single quote."),
        new TAUOption("-noex","","Specifies that no exceptions be used while compiling the library. This is relevant for C++."),
    };
    
    protected static void showHelp(Object button)
    {
        for(int i=0;i<OptionSet.options.length;i++)
        {
            if(OptionSet.options[i].helpButton!=null && OptionSet.options[i].helpButton.equals(button))
            {
                OptionSet.options[i].QButton();
                break;
            }
        }
    }
    
    /*protected static void optCheck(Object checkbox)
    {
        for(int i=0;i<OptionSet.options.length;i++)
        {
            if(OptionSet.options[i].optToggle!=null && OptionSet.options[i].optToggle.equals(checkbox))
            {
                OptionSet.options[i].CheckChecked();
                break;
            }
        }        
    }*/
    
    protected static JButton getHelpButton(String commandFlag)
    {
        for(int i=0;i<OptionSet.options.length;i++)
        {
            if(OptionSet.options[i].helpButton!=null && OptionSet.options[i].optFlag.equals(commandFlag))
            {
                //return OptionSet.options[i].helpButton=new JButton();
                return OptionSet.options[i].helpButton;
            }
        }
        return null;
    }
    
    protected static JCheckBox getCheckBox(String commandFlag)
    {
        for(int i=0;i<OptionSet.options.length;i++)
        {
            if(OptionSet.options[i].optToggle!=null && OptionSet.options[i].optFlag.equals(commandFlag))
            {
                //return OptionSet.options[i].helpButton=new JButton();
                return OptionSet.options[i].optToggle;
            }
        }
        return null;
    }
}
