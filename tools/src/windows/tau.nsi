
!ifndef VERSION
  !define VERSION '2.X'
!endif

!define NAME "Tau ${VERSION}"


!macro SED_REPLACE FILE
ClearErrors
FileOpen $0 "${FILE}" "r"                     ; open target file for reading
GetTempFileName $R0                            ; get new temp file name
FileOpen $1 $R0 "w"                            ; open temp file for writing
loop-${FILE}:
   FileRead $0 $2                              ; read line from target file
   IfErrors done-${FILE}                               ; check if end of file reached
   StrCmp $2 "set TAU_ROOT=..$\r$\n" 0 +2      ; compare line with search string with CR/LF
      StrCpy $2 'set TAU_ROOT="$INSTDIR"$\r$\n'    ; change line
   StrCmp $2 "set TAU_ROOT=.." 0 +2            ; compare line with search string without CR/LF (at the end of the file)
      StrCpy $2 'set TAU_ROOT="$INSTDIR"'          ; change line
   FileWrite $1 $2                             ; write changed or unchanged line to temp file
   Goto loop-${FILE}
 
done-${FILE}:
   FileClose $0                                ; close target file
   FileClose $1                                ; close temp file
   Delete "${FILE}"                           ; delete target file
   CopyFiles /SILENT $R0 "${FILE}"            ; copy temp file to target file
   Delete $R0                                  ; delete temp file 
!macroend


!macro ASSOCIATE EXT TYPE DESC ICON EXECUTE
  ; back up old value of extension
  ReadRegStr $1 HKCR "${EXT}" ""
  StrCmp $1 "" "${EXT}-NoBackup"
  StrCmp $1 "${TYPE}" "${EXT}-NoBackup"
  WriteRegStr HKCR "${EXT}" "backup_val" $1
  
  "${EXT}-NoBackup:"
  WriteRegStr HKCR "${EXT}" "" "${TYPE}"
  ReadRegStr $0 HKCR "${TYPE}" ""
  StrCmp $0 "" 0 "${EXT}-Skip"
  WriteRegStr HKCR "${TYPE}" "" "${DESC}"
  WriteRegStr HKCR "${TYPE}\shell" "" "open"
  WriteRegStr HKCR "${TYPE}\DefaultIcon" "" ${ICON}
  
  "${EXT}-Skip:"
  WriteRegStr HKCR "${TYPE}\shell\open\command" "" '${EXECUTE} "%1"'
  System::Call 'Shell32::SHChangeNotify(i 0x8000000, i 0, i 0, i 0)'
!macroend


!macro CLEANASSOC EXT TYPE
  !define Index "Line${__LINE__}"
  ReadRegStr $1 HKCR "${EXT}" ""
  StrCmp $1 "${TYPE}" 0 "${Index}-NoOwn" ; only do this if we own it
  ReadRegStr $1 HKCR "${EXT}" "backup_val"
  StrCmp $1 "" 0 "${Index}-Restore" ; if backup="" then delete the whole key
  DeleteRegKey HKCR "${EXT}"
  Goto "${Index}-NoOwn"
  "${Index}-Restore:"
  WriteRegStr HKCR "${EXT}" "" $1
  DeleteRegValue HKCR "${EXT}" "backup_val"
  DeleteRegKey HKCR "${TYPE}" ;Delete key with association settings
  System::Call 'Shell32::SHChangeNotify(i 0x8000000, i 0, i 0, i 0)'
  "${Index}-NoOwn:"
  !undef Index
!macroend


;--------------------------------

; The name of the installer
Name "Tau"
!ifdef OUTFILE
  OutFile "${OUTFILE}"
!else
  OutFile "C:\tau\zip\${NAME}.exe"
!endif

; The default installation directory
InstallDir "$PROGRAMFILES\${NAME}"

;--------------------------------

; Pages

Page directory
Page instfiles
UninstPage uninstConfirm
UninstPage instfiles

;--------------------------------

; The stuff to install
Section "" ;No components page, name is not important

  ; Set output path to the installation directory.
  SetOutPath $INSTDIR
  
  ; Put files there
  File /r c:\tau\tau-windows\*.*
  
  !insertmacro SED_REPLACE $INSTDIR\bin\paraprof.bat
  !insertmacro SED_REPLACE $INSTDIR\bin\jumpshot.bat
  !insertmacro SED_REPLACE $INSTDIR\bin\perfdmf_configure.bat
  !insertmacro SED_REPLACE $INSTDIR\bin\perfexplorer_configure.bat
  !insertmacro SED_REPLACE $INSTDIR\bin\perfexplorer.bat
  !insertmacro SED_REPLACE $INSTDIR\bin\tau2slog2.bat
  
  ; Write the uninstall keys for Windows
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${NAME}" "DisplayName" "${NAME}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${NAME}" "UninstallString" '"$INSTDIR\uninstall.exe"'
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${NAME}" "NoModify" 1
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${NAME}" "NoRepair" 1
  WriteUninstaller "uninstall.exe"

  ; Create Shortcuts  
  CreateDirectory "$SMPROGRAMS\${NAME}"
  CreateShortCut "$SMPROGRAMS\${NAME}\ParaProf.lnk" $INSTDIR\bin\paraprof.bat "" "$INSTDIR\bin\tau.ico"
  CreateShortCut "$SMPROGRAMS\${NAME}\Readme.lnk" $INSTDIR\Readme.txt
  CreateShortCut "$SMPROGRAMS\${NAME}\README-PIN.lnk" $INSTDIR\README-PIN.txt
  CreateShortCut "$SMPROGRAMS\${NAME}\JumpShot.lnk" $INSTDIR\bin\jumpshot.bat
  CreateShortCut "$SMPROGRAMS\${NAME}\PerfDMF_Configure.lnk" $INSTDIR\bin\perfdmf_configure.bat "" "$INSTDIR\bin\tau.ico"
  CreateShortCut "$SMPROGRAMS\${NAME}\PerfExplorer_Configure.lnk" $INSTDIR\bin\perfexplorer_configure.bat "" "$INSTDIR\bin\tau.ico"
  CreateShortCut "$SMPROGRAMS\${NAME}\PerfExplorer.lnk" $INSTDIR\bin\perfexplorer.bat "" "$INSTDIR\bin\tau.ico"
  CreateShortCut "$SMPROGRAMS\${NAME}\Uninstall.lnk" "$INSTDIR\uninstall.exe"
  
  
  
  !insertmacro ASSOCIATE .ppk Tau.PPK "Packed Profile" "$INSTDIR\bin\tau.ico" "$INSTDIR\bin\paraprof.bat"
  !insertmacro ASSOCIATE .slog2 Tau.SLOG2 "SLOG2 Trace" "$INSTDIR\bin\tau.ico" "$INSTDIR\bin\jumpshot.bat"
  
  
  
  
SectionEnd ; end the section


; The uninstall section
Section "Uninstall"

  !insertmacro CLEANASSOC .ppk Tau.PPK
  !insertmacro CLEANASSOC .slog2 Tau.SLOG2

  ; Remove registry keys
  DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${NAME}"
  RMDir /r "$SMPROGRAMS\${NAME}"
  RMDir /r "$INSTDIR"

SectionEnd
