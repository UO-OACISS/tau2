Summary: TAU Portable Profiling and Tracing Package
Name: TAU
Version: 2.6
Release: 0
Group: Development/Tools
Copyright: Copyright (c) 1999 University of Oregon, Los Alamos National Laboratory
Packager: Sameer Shende <sameer@cs.uoregon.edu>
Prefix:/usr/local/packages
Requires: /bin/sh

%description
TAU (Tuning and Analysis Utilities) provides a framework for integrating
program and performance analysis tools and components. A core tool
component for parallel performance evaluation is a profile measurement and
analysis package. The TAU portable profiling and tracing package was developed jointly by the University of Oregon and LANL for profiling parallel, multi-threaded C++ programs. The package implements a instrumentation library, profile analysis procedures, and a visualization tool. 

It runs on SGI Origin 2000s, Intel PCs running Linux, Sun, Compaq Alpha Tru64,
Compaq Alpha Linux clusters, HP workstations and Cray T3E. The current release 
(v2.6) supports C++, C and Fortran90. It works  with KAI KCC, PGI, g++, egcs, 
Fujitsu and vendor supplied C++ compilers. See INSTALL and README files 
included with the distribution. Documentation can be found at 
http://www.cs.uoregon.edu/research/tau

- Sameer Shende (University of Oregon)
Report bugs to tau-bugs@cs.uoregon.edu

%prep

%build

%install
%post
echo "Installed TAU 2.6 See INSTALL file. Use configure and make install"

%files
%attr(-,root,root) /home/users/sameer/tau-2.6
