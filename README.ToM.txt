TAUoverMRNet 2.0-alpha build instructions:
==========================================
By: Chee Wai Lee (cheelee@cs.uoregon.edu)

Introduction:
=============
     This documents the steps required to build the alpha re-engineered 
     implementation of TAUoverMRNet (ToM) with TAU and a beta (unreleased)
     copy of MRNet 3.0.

Software Requirements:
======================
     1. A copy of TAU pre 2.19.2 release (or higher number depending on
        actual release which will officially support ToM).
     2. A copy of MRNet 3.0 (or beta). The current code base supports
        the build dated 2010-04-26 (yyyy-mm-dd).

Other Requirements:
===================
     1. TAU must be built with MPI.
     2. ToM will only work with MPI codes.
     3. ToM current relies on TAU experimental feature TAU_EXP_UNIFY.

Building ToM:
=============
     1. Configure TAU:
          cd <tau root directory>;
          ./configure -mpi 
                      -mpiinc=<mpi include dir, if needed>
                      -mpilib=<mpi lib dir, if needed>
                      -pdt=<pdt support if needed>
                      -mrnet=<mrnet source root>
                      -mrnetlib=<dir where mrnet libraries are installed>

     2. Build TAU:
          make install

     3. Edit ToM Front-End parameters:
          cd tools/src/ToM;
          edit Makefile - set INSTALL_ROOT, CXX and CXXFLAGS

     4. Build ToM Front-End:
          make install

     5. Locate appropriate start script in the scripts directory. These
          will generally be named startToM_<platform>.sh. The selected
	  script should be copied to the directory where the experiments
	  with ToM are to be run.

     6. Edit ToM supporting tools:
          cd probeHosts;
          edit Makefile - set INSTALL_DIR and settings required for MPI.

     7. Build supporting tools:
          make install

     8. Make sure INSTALL_ROOT/bin is in the PATH environment variable; and
        INSTALL_ROOT/lib is in the LD_LIBRARY_PATH environment variable.

     9. Build your application with the ToM-supported build of TAU. Make
        sure the environment variable TAU_MAKEFILE is correctly set.

Running ToM-supported instrumented application:
===============================================

This will be somewhat platform specific. In general, 2 or more steps 
are followed:

1. Start the front-end using the start script:

      ./startToM_<platform>.sh <num total cores> ToM_FE <num app cores> <mrnet fanout> <mrnet depth>

   This also uses MRNet's default topology-file constructor. If so desired,
   the script can be modified so a custom topology-file generation process
   is deployed. The script puts the front-end process into the background
   so you can proceed with the next step.

2. Start the back-end processes (your instrumented application) using
   <num app cores>.

Example:

      ./startToM_craycnl.sh 24 ToM_FE 8 2 2
      mpirun -n 8 ./hello-PDT 3

Platform-specific Issues:
=========================

1. On Linux Rocks clusters, depending on cluster setup, you may need
   to use ssh-agent to allow MRNet processes to communicate via ssh
   without password requirements. An example to do so 
   (from the node you are first allocated interactively):

   ssh-agent bash;
   ssh-add

   You will be prompted for your password. This also implies you cannot
   use such a system non-interactively.

2. On the Cray XT series, MRNet tree processes cannot share nodes with
   application processes. aprun is apparently only node-aware when 
   putting stuff into background execution. Therefore, it will wait
   for sufficient nodes to be available for executing the application
   process if the MRNet tree occupies more nodes.

Known ToM Issues:
=================

1. ToM is still currently in alpha development. There are many esoteric
restrictions and design decisions that need to be ironed-out for
production use. This document will continue to be updated with lists
as development proceeds towards a proper release.

2. Please contact cheelee@cs.uoregon.edu for any questions and comments.
Thank you.