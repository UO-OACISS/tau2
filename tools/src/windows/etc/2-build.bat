
REM call "C:\Program Files\Microsoft Visual Studio 9.0\Common7\Tools\vsvars32.bat"
REM call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\vsDevCmd.bat"
rem call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"

if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\vsDevCmd.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\vsDevCmd.bat"
) else if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" (
    call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
) else (
    echo Error: Could not locate vsDevCmd.bat in any standard location.
    pause
    exit /b 1
)

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
 
