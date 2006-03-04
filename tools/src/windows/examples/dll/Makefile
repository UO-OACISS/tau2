

INC=-I..\..\include
PROFILELIB=..\..\lib\tau-profile.lib
PROFILEDLL=..\..\lib\tau-profile.dll
CALLPATHLIB=..\..\lib\tau-callpath.lib
CALLPATHDLL=..\..\lib\tau-callpath.dll
TRACELIB=..\..\lib\tau-trace.lib
TRACEDLL=..\..\lib\tau-trace.dll

all: profile callpath trace

profile: main.cpp library.cpp
	cl /MD /GX -DPROFILING_ON -DTAU_USE_C_API -c library.cpp $(INC)
	link /dll /out:library-profile.dll library.obj $(PROFILELIB)
	cl -o profile /MD /GX -DPROFILING_ON -DTAU_USE_C_API main.cpp $(INC) $(PROFILELIB) library-profile.lib
	cp $(PROFILEDLL) .

callpath: main.cpp library.cpp
	cl /MD /GX -DPROFILING_ON -DTAU_CALLPATH -DTAU_USE_C_API -c library.cpp $(INC)
	link /dll /out:library-callpath.dll library.obj $(CALLPATHLIB)
	cl -o callpath /MD /GX -DPROFILING_ON -DTAU_CALLPATH -DTAU_USE_C_API main.cpp $(INC) $(CALLPATHLIB) library-callpath.lib
	cp $(CALLPATHDLL) .

trace: main.cpp library.cpp
	cl /MD /GX -DTRACING_ON -DTAU_USE_C_API -c library.cpp $(INC)
	link /dll /out:library-trace.dll library.obj $(TRACELIB)
	cl -o trace /MD /GX -DTRACING_ON -DTAU_USE_C_API main.cpp $(INC) $(TRACELIB) library-trace.lib
	cp $(TRACEDLL) .


clean: 
	del *.exe *.ncb *.obj *.bak profile.* *.trc *.edf *.gz *.vpt *.slog2 *.dll *.lib *.exp
