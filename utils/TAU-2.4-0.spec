Summary: TAU Portable Profiling and Tracing Package
Name: TAU
Version: 2.4
Release: 0
Group: Development/Tools
Copyright: Copyright (c) 1998 University of Oregon, Los Alamos National Laboratory
Packager: Sameer Shende <sameer@cs.uoregon.edu>
Prefix:/usr/local/packages
Requires: /bin/sh

%description
TAU (Tuning and Analysis Utilities) provides a framework for integrating
program and performance analysis tools and components. A core tool
component for parallel performance evaluation is a profile measurement and
analysis package. The TAU portable profiling and tracing package was developed jointly by the University of Oregon and LANL for profiling parallel, multi-threaded C++ programs. The package implements a instrumentation library, profile analysis procedures, and a visualization tool. 

It runs on SGI Origin 2000s, PCs running Linux, Sun, DEC, HP workstations
and Cray T3E. The current release (v2.4) supports C++ as well as C. It
works  with KAI KCC, g++, egcs and vendor supplied C++ compilers. See INSTALL 
and README files included with the distribution. Documentation can be
found at http://www.acl.lanl.gov/tau
- Sameer Shende (ACL, Los Alamos National Laboratory, University of Oregon)
Report bugs to taubugs@cs.uoregon.edu

%prep

%build

%install
%post
echo "Installed TAU 2.4 See INSTALL file and  %./configure and % make install"

%files
%attr(-,root,root) /users/sameer/rs/tau-2.4

