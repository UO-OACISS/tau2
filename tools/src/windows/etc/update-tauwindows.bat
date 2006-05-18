
c:\cygwin\bin\bash --login "/c/update-tau.sh"

copy c:\tau2\tools\src\windows\trace_impl.h c:\tau2\utils\slogconverter

cd \
cd tau2
nmake /f Makefile.win32 /k
nmake /f Makefile.win32 JDK=c:\j2sdk1.4.2_07 java
nmake /f Makefile.win32 PDT=c:\pdtoolkit tau_instrumentor
nmake /f Makefile.win32 VTF=c:\vtf3\binaries tau2vtf
nmake /f makefile.win32 JDK=c:\j2sdk1.4.2_07 SLOG2=c:\slog2sdk-1.2.5beta tau2slog2
del c:\tau2\win32\bin\TraceInput.exp
del c:\tau2\win32\bin\TraceInput.lib


copy tools\src\windows\bin\* c:\tau-windows\bin
xcopy /e /y tools\src\windows\examples c:\tau-windows\examples
md c:\tau-windows\tools\src\perfdmf\etc
xcopy /e /y tools\src\perfdmf\etc c:\tau-windows\tools\src\perfdmf\etc
copy win32\bin\*.* c:\tau-windows\bin
copy tools\src\contrib\*.jar c:\tau-windows\bin
copy tools\src\perfdmf\bin\perfdmf.jar c:\tau-windows\bin
copy tools\src\common\bin\tau-common.jar c:\tau-windows\bin
copy tools\src\paraprof\bin\*.jar c:\tau-windows\bin
copy tools\src\vis\bin\*.jar c:\tau-windows\bin
copy tools\src\contrib\windows\jogl.dll c:\tau-windows\bin
copy tools\src\contrib\slog2sdk\lib\jumpshot.jar c:\tau-windows\bin
copy win32\lib\*.* c:\tau-windows\lib
copy win32\java\*.* c:\tau-windows\javadll
copy LICENSE c:\tau-windows
copy tools\src\windows\etc\Readme.txt c:\tau-windows


cd \
