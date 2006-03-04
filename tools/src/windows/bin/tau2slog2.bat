
::
:: set TAU_ROOT below and make sure java is in your path
::
set TAU_ROOT=c:\tau-windows
set SLOG_ROOT=c:\slog2sdk-1.2.5beta
java -Xmx500m -Djava.library.path=%TAU_ROOT%\bin -jar %SLOG_ROOT%\lib\traceTOslog2.jar %1:%2 %3 %4 %5
