

INC=-I..\..\include
PROFILELIB=..\..\lib\tau-profile.lib
PROFILEDLL=..\..\lib\tau-profile.dll

all: profile

profile: main.cpp library.cpp
	cl /MD /GX -DPROFILING_ON -DTAU_USE_C_API -c library.cpp $(INC)
	link /dll /out:library-profile.dll library.obj $(PROFILELIB)
	cl -o profile /MD /GX -DPROFILING_ON -DTAU_USE_C_API main.cpp $(INC) $(PROFILELIB) library-profile.lib
	cp $(PROFILEDLL) .



clean: 
	del *.exe *.ncb *.obj *.bak profile.* *.trc *.edf *.gz *.vpt *.slog2 *.dll *.lib *.exp
