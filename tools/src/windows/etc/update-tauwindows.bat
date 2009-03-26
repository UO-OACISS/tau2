
set ROOT=C:\tau
c:\cygwin\bin\bash --login "/c/tau/update-tau.sh"

copy %ROOT%\tau2\tools\src\windows\trace_impl.h %ROOT%\tau2\utils\slogconverter
copy %ROOT%\tau2\tools\src\windows\etc\tau_config.h %ROOT%\tau2\include
copy %ROOT%\tau2\tools\src\windows\etc\tauarch.h %ROOT%\tau2\include

cd %ROOT%\tau2
nmake /f Makefile.win32 /k
nmake /f Makefile.win32 JDK=%ROOT%\j2sdk1.4.2_13 java
nmake /f Makefile.win32 PDT=%ROOT%\pdtoolkit tau_instrumentor
nmake /f Makefile.win32 VTF=%ROOT%\vtf3\binaries tau2vtf
cd %ROOT%\tau2\utils\taupin
nmake /f Makefile.win32 clean
nmake /f Makefile.win32

cd %ROOT%\tau2
del %ROOT%\tau2\win32\bin\TraceInput.exp
del %ROOT%\tau2\win32\bin\TraceInput.lib

copy utils\taupin\tau_pin.exe %ROOT%\tau-windows\bin
copy utils\taupin\*.dll %ROOT%\tau-windows\bin
copy utils\taupin\README-PIN.txt %ROOT%\tau-windows
copy %ROOT%\pin\*.* %ROOT%\tau-windows\bin

copy tools\src\windows\bin\* %ROOT%\tau-windows\bin
copy src\Profiles\TAU.jar %ROOT%\tau-windows\bin
xcopy /e /y tools\src\windows\examples %ROOT%\tau-windows\examples
md %ROOT%\tau-windows\examples\pin
copy %ROOT%\pin\examples\*.* %ROOT%\tau-windows\examples\pin
md %ROOT%\tau-windows\tools\src\perfdmf\etc
md %ROOT%\tau-windows\etc
md %ROOT%\tau-windows\contrib
copy tools\src\perfexplorer\etc\* %ROOT%\tau-windows\etc
copy %ROOT%\weka.jar %ROOT%\tau-windows\bin
xcopy /e /y tools\src\perfdmf\etc %ROOT%\tau-windows\tools\src\perfdmf\etc
copy win32\bin\*.* %ROOT%\tau-windows\bin
copy tools\src\contrib\*.jar %ROOT%\tau-windows\bin
copy tools\src\perfdmf\bin\perfdmf.jar %ROOT%\tau-windows\bin
copy tools\src\perfexplorer\perfexplorer.jar %ROOT%\tau-windows\bin
copy tools\src\common\bin\tau-common.jar %ROOT%\tau-windows\bin
copy tools\src\paraprof\bin\*.jar %ROOT%\tau-windows\bin
copy tools\src\vis\bin\*.jar %ROOT%\tau-windows\bin
copy tools\src\contrib\jogl\jogl.jar %ROOT%\tau-windows\bin
copy tools\src\contrib\jogl\windows\jogl.dll %ROOT%\tau-windows\bin
copy tools\src\contrib\jogl\windows\jogl_awt.dll %ROOT%\tau-windows\bin
copy tools\src\contrib\jogl\windows\jogl_cg.dll %ROOT%\tau-windows\bin
copy tools\src\contrib\slog2sdk\lib\jumpshot.jar %ROOT%\tau-windows\bin
copy win32\lib\*.* %ROOT%\tau-windows\lib
copy win32\java\*.* %ROOT%\tau-windows\javadll
copy LICENSE %ROOT%\tau-windows
copy tools\src\windows\etc\Readme.txt %ROOT%\tau-windows
copy tools\src\contrib\LICENSE-* %ROOT%\tau-windows\contrib
copy tools\src\contrib\jogl\LICENSE-* %ROOT%\tau-windows\contrib
copy tools\src\contrib\jogl\COPYRIGHT-* %ROOT%\tau-windows\contrib

cd %ROOT%
