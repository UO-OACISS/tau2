@echo off

if "%1" == "" goto error


rd /s /q tau-windows

md c:\tau-windows
md c:\tau-windows\bin
md c:\tau-windows\lib
md c:\tau-windows\examples
md c:\tau-windows\JavaDLL
md c:\tau-windows\include
md c:\tau-windows\lib\VC6
md c:\tau-windows\lib\VC7
md c:\tau-windows\lib\VC8

call vc6
call update-tauwindows.bat
move c:\tau-windows\lib\tau-*.* c:\tau-windows\lib\VC6

call vc7
call update-tauwindows.bat
move c:\tau-windows\lib\tau-*.* c:\tau-windows\lib\VC7

call vc8
call update-tauwindows.bat
move c:\tau-windows\lib\tau-*.* c:\tau-windows\lib\VC8


rmdir /q /s zip
mkdir zip
mkdir zip\tau-%1
xcopy /e tau-windows zip\tau-%1
cd zip
zip -r tau-%1.zip tau-%1

echo -------------
echo Finished
echo File is c:\zip\tau-%1.zip
echo -------------

cd \

PATH=%PATH%;"c:\Program Files\NSIS"

makensis /DVERSION=%1 /DOUTFILE=C:\zip\tau-%1 C:\tau2\tools\src\windows\tau.nsi
goto done

:error
echo -------------
echo Usage: build-windows-release version (e.g. 2.14.5)
echo -------------

: done
