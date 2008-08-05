
all: profile callpath trace

profile: stdafx.cpp stdafx.h testTau.cpp
	cl /MD -GX -DPROFILING_ON -o profile stdafx.cpp testTau.cpp -I..\..\include /link ..\..\lib\vc9\tau-profile-static-mt.lib

callpath: stdafx.cpp stdafx.h testTau.cpp
	cl /MD -GX -DPROFILING_ON -DTAU_CALLPATH -o callpath stdafx.cpp testTau.cpp -I..\..\include /link ..\..\lib\vc9\tau-callpath-static-mt.lib

trace: stdafx.cpp stdafx.h testTau.cpp
	cl /MD -GX -DTRACING_ON -o trace stdafx.cpp testTau.cpp -I..\..\include /link ..\..\lib\vc9\tau-trace-static-mt.lib

clean: 
	del *.exe *.ncb *.obj *.bak profile.* *.trc *.edf *.gz *.vpt *.slog2
