
all: mpi

mpi: stdafx.cpp stdafx.h mpi.cpp
	cl /MD -GX -DPROFILING_ON -o mpi stdafx.cpp mpi.cpp /I "C:\Program Files\Microsoft Compute Cluster Pack\Include" -I..\..\include /link ..\..\lib\vc9\tau-profile-static-mt.lib /LIBPATH:"C:\Program Files\Microsoft Compute Cluster Pack\Lib\i386" /LIBPATH:"..\..\lib\vc9" TauMpi.lib msmpi.lib DelayImp.lib


clean: 
	del *.exe *.ncb *.obj *.bak profile.* *.trc *.edf *.gz *.vpt *.slog2
