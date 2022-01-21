
REM call "C:\Program Files\Microsoft Visual Studio 9.0\Common7\Tools\vsvars32.bat"
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\vsDevCmd.bat"
set ROOT=C:\tau
cd %ROOT%\tau2
mkdir win32\lib
bash configure -cc=x86_64-w64-mingw32-gcc -arch=x86_64
nmake /f Makefile.win32
REM nmake /f Makefile.win32 JDK=%ROOT%\j2sdk1.4.2_13 java
REM nmake /f Makefile.win32 PDT=%ROOT%\pdtoolkit tau_instrumentor
REM nmake /f Makefile.win32 VTF=%ROOT%\vtf3\binaries tau2vtf
REM cd %ROOT%\tau2\utils\taupin
REM nmake /f Makefile.win32 clean
REM nmake /f Makefile.win32
 
