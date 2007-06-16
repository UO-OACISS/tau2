@echo off

set ROOT=C:\tau

if "%1" == "" goto error

cd %ROOT%

rd /s /q tau-windows

md %ROOT%\tau-windows
md %ROOT%\tau-windows\bin
md %ROOT%\tau-windows\lib
md %ROOT%\tau-windows\examples
md %ROOT%\tau-windows\JavaDLL
md %ROOT%\tau-windows\include
md %ROOT%\tau-windows\lib\VC7
md %ROOT%\tau-windows\lib\VC8

call vc7
call update-tauwindows.bat
move %ROOT%\tau-windows\lib\tau-*.* %ROOT%\tau-windows\lib\VC7

call vc8
call update-tauwindows.bat
move %ROOT%\tau-windows\lib\tau-*.* %ROOT%\tau-windows\lib\VC8


rmdir /q /s zip
mkdir %ROOT%\zip
mkdir %ROOT%\zip\tau-%1
xcopy /e %ROOT%\tau-windows %ROOT%\zip\tau-%1
cd %ROOT%\zip
zip -r tau-%1.zip tau-%1

echo -------------
echo Finished
echo File is %ROOT%\zip\tau-%1.zip
echo -------------

cd %ROOT%

PATH=%PATH%;"c:\Program Files\NSIS"

makensis /DVERSION=%1 /DOUTFILE=%ROOT%\zip\tau-%1.exe %ROOT%\tau2\tools\src\windows\tau.nsi
goto done

:error
echo -------------
echo Usage: build-windows-release version (e.g. 2.14.5)
echo -------------

: done
