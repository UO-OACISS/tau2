
call "C:\Program Files\Microsoft Visual Studio 9.0\Common7\Tools\vsvars32.bat"

set ROOT=C:\tau
cd %ROOT%\tau2

nmake /f Makefile.win32 /k
nmake /f Makefile.win32 JDK=%ROOT%\j2sdk1.4.2_13 java
REM nmake /f Makefile.win32 PDT=%ROOT%\pdtoolkit tau_instrumentor
nmake /f Makefile.win32 VTF=%ROOT%\vtf3\binaries tau2vtf
cd %ROOT%\tau2\utils\taupin
nmake /f Makefile.win32 clean
nmake /f Makefile.win32
 
