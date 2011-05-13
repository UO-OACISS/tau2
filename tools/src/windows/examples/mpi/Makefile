
all: mpi

mpi: stdafx.cpp stdafx.h mpi.cpp
	cl /MT -GX -DPROFILING_ON -o mpi stdafx.cpp mpi.cpp /I "C:\Program Files\Microsoft Compute Cluster Pack\Include" /I "C:\Program Files\Microsoft HPC Pack 2008 SDK\Include" -I..\..\..\..\..\include /link ..\..\..\..\..\win32\lib\tau-profile-static-mt.lib /LIBPATH:"C:\Program Files\Microsoft Compute Cluster Pack\Lib\i386" /LIBPATH:"C:\Program Files\Microsoft HPC Pack 2008 SDK\Lib\i386" /LIBPATH:"..\..\..\..\..\win32\lib" TauMpi.lib msmpi.lib DelayImp.lib


clean: 
	del *.exe *.ncb *.obj *.bak profile.* *.trc *.edf *.gz *.vpt *.slog2
