#*********************************************************************#
#*                TAU/pC++/Sage++  Copyright (C) 1994                *#
#*                        Jerry Manic Saftware                       *#
#*  Indiana University  University of Oregon  University of Rennes   *#
#*********************************************************************#
displayMessage hourglass "Loading $myself..."

source "$TAUDIR/inc/cosyutil.tcl"
source "$TAUDIR/inc/depfile.tcl"
source "$TAUDIR/inc/fileio.tcl"
source "$TAUDIR/inc/help.tcl"
source "$TAUDIR/inc/selectfile.tcl"
source "$TAUDIR/inc/stack.tcl"
source "$TAUDIR/inc/tauutil.tcl"

set BUILDDIR "$TAUROOT/build"

set cosy_var(make) "norm";       # program "type": prf-, trc-, brk-
set cosy_var(old_make) "norm";
set cosy_var(makefile) "";       # name of the makefile
set cosy_var(warn) 1;            # show warnings?
set cosy_var(com) "";            # holds current command to execute
set cosy_var(reset) 1;           # reset text output window?
set cosy_var(deftarget) "";      # default make target
set cosy_var(filtarget) 0;       # filter out target which contain "."
set cosy_var(num_errs) 0;        # number of errors found
set cosy_old_tval(adjust) " -a";
set cosy_old_tval(format) "dump"

#  DEBUG - For debuggin messages, change DEBUG_SET to 1 and wrap your
#    scaf in 'if [DEBUG] { }'
set DEBUG_SET 0
proc DEBUG {} {
    global DEBUG_SET
    return $DEBUG_SET
}

#
#  ALONE - for standalone cosy
#
set ALONE_SET 0
proc ALONE {} {
    global ALONE_SET
    return $ALONE_SET
}

#
#  doExit
#  Added 6/26/96 SAS
#
#  This procedure sets up for exiting from cosy, by removing cosy from
#  the list of working tools, then exiting.
rename exit exit.old
proc exit {{status 0}} {
    global myself

    PM_RemTool $myself
    exit.old $status
}

#
# machineMenu: generate menu item which represent POSSIBLE executables
#              for the current machine
#
#        menu: pathname of menu widget
#         com: command to execute for each item
#

proc machineMenu {menu com} {
  global cosy_var

  $menu delete 0 last
  if { $cosy_var(make) == "norm" } {
    foreach m $cosy_var(machines) {
      $menu add command -label $m -underline 0 -command "$com $m"
    }
  } else {
    foreach m $cosy_var(machines) {
      $menu add command -label $cosy_var(make)$m -underline 0 -command "$com $m"
    }
  }
}

#
# getExecutables: generate list of all AVAILABLE executables
#

proc getExecutables {} {
  global depfile
  global cosy_var

  set projname [file rootname $depfile(project)]
  set files [FileIO_ls $depfile(host) $depfile(dir)]
  set result ""
  foreach mach $cosy_var(machines) {
    foreach type {{} prf- trc- brk-} {
      set exec $projname-$type$mach
      set idx [lsearch -exact $files $exec]
      if { $idx != -1 } {
        lappend result $projname-$type$mach
      }
    }
  }
  return $result
}

#
#   setMachines: determine available executables for current machines
# appendMachine: auxillary function for setting machine related variables
#

proc appendMachine {mach} {
  global cosy_var

  lappend cosy_var(machines) $mach
  set cosy_var(defmach) $mach
}

proc setMachines {} {
  global depfile
  global cosy_var

  # -- uniprocessor mode always possible
  set cosy_var(machines) uniproc
  set cosy_var(defmach)  uniproc

  # -- check if running on multiprocessor
  switch -exact $depfile(arch) {
    ptx           -
    symmetry      { appendMachine symmetry }
    cm5           { appendMachine cm5 }
    t3e           { appendMachine t3e }
    paragon       { appendMachine sunmos
                    appendMachine paragon }
    ksr1          { appendMachine ksr-ms
                    appendMachine ksr }
  }

  # -- check for cross-development environments
  switch -exact $depfile(arch) {
    rs6000     -
    sgi4k      -
    sgi8k      -
    solaris2   -
    c90        -
    hp9000s700 -
    hp9000s800 -
    sun4      {
      if { $depfile(host) == "localhost" } {
        set xdev [FileIO_exec $depfile(host) "$depfile(root)/utils/archfind -x"]
      } else {
        set xdev [FileIO_exec $depfile(host) "cd $depfile(dir);$depfile(root)/utils/archfind -x"]
      }
      switch -exact $xdev {
        sp1     { appendMachine sp2 }
        cs2     { appendMachine cs2.nx }
        sgimp   { appendMachine sgimp-ms
                  appendMachine sgimp }
        cm5     { appendMachine cm5 }
        paragon { appendMachine paragon }
        t3d     { appendMachine t3d }
        cnxspp  { appendMachine cnxspp-ms }
      }
    }
  }

  # -- check for additional development environments
  if { $depfile(host) != "localhost" } {
    set com "cd $depfile(dir); "
  }
  append com "$depfile(root)/utils/checkExtraRTS $depfile(root) $depfile(arch)"
  set extra [FileIO_exec $depfile(host) $com]
  if { $extra != "NOT_OK" } {
    foreach e $extra {
      lappend cosy_var(machines) $e
    }
  }

  # -- get local hostname if necessary
  set cosy_var(host) ""
  if { $depfile(host) == "localhost" } {
    if [catch {exec uname -n} cosy_var(host)] {
      [catch {exec hostname} cosy_var(host)]
    }
  } else {
    set cosy_var(host) $depfile(host)
  }
}


#
# createWindow: create main application window
#

proc createWindow {} {
  global TAUDIR
  global myself
  global cosy_var

  toplevel .$myself
  wm title .$myself "COSY"
  wm minsize .$myself 50 50
  wm iconbitmap .$myself @$TAUDIR/xbm/$myself.xbm

  # The error scrolling globals, Alt-n and Alt-p 
  # 4/15 SAS
if {![ALONE]} {
  bind .$myself <Alt-n> errScrollForward
  bind .$myself <Alt-p> errScrollBackward
}

  # -- menu bar
  frame .$myself.mbar -relief raised -borderwidth 2

if {[ALONE]} {
  menubutton .$myself.mbar.b1 -text Project -menu .$myself.mbar.b1.m1 -underline 0
  menu .$myself.mbar.b1.m1 
  .$myself.mbar.b1.m1 add command -label "Load new project" -underline 5 \
	         -command "tau_OpenProject"
  .$myself.mbar.b1.m1 add command -label "Copy project (Backup)" -underline 0 \
                 -command "tau_CopyProject"
  .$myself.mbar.b1.m1 add command -label "View/Change project options" -underline 0 \
                 -command "tau_ProjectOptions"
  .$myself.mbar.b1.m1 add separator
  .$myself.mbar.b1.m1 add command -label "Exit" -underline 0 \
                 -command "exit"
}

  menubutton .$myself.mbar.b2 -text File -menu .$myself.mbar.b2.m1 -underline 0
  menu .$myself.mbar.b2.m1
if {[ALONE]} {
  .$myself.mbar.b2.m1 add command -label "Add source file" -underline 0 \
	         -command "tau_AddFile"
  .$myself.mbar.b2.m1 add command -label "Delete selected source file" -underline 0 \
	         -command "tau_DeleteFile"
}
  .$myself.mbar.b2.m1 add command -label "List directory" -underline 0 \
                 -command "doExec {ls -l}"
if {![ALONE]} {
  .$myself.mbar.b2.m1 add separator
  .$myself.mbar.b2.m1 add command -label "Exit" -underline 0 \
                 -command "exit"
}

  menubutton .$myself.mbar.b3 -text View -menu .$myself.mbar.b3.m1 -underline 0
  menu .$myself.mbar.b3.m1 
  .$myself.mbar.b3.m1 add checkbutton -label "Show warnings" -underline 0 \
                -variable cosy_var(warn) -onvalue 1 -offvalue 0
  .$myself.mbar.b3.m1 add checkbutton -label "Reset output" -underline 0 \
                -variable cosy_var(reset) -onvalue 1 -offvalue 0
  .$myself.mbar.b3.m1 add checkbutton -label "Filter targets" -underline 0 \
                -variable cosy_var(filtarget) -onvalue 1 -offvalue 0
  .$myself.mbar.b3.m1 add separator
  .$myself.mbar.b3.m1 add command -label "Clear window" -underline 0 \
                 -command "untagErrors; .$myself.bot.right.txt delete 1.0 end; set cosy_var(com) {}"

  menubutton .$myself.mbar.b4 -text Make -menu .$myself.mbar.b4.m1 -underline 0
  menu .$myself.mbar.b4.m1 
  .$myself.mbar.b4.m1 add cascade -label "Build makefile" -underline 0 \
                    -menu .$myself.mbar.b4.m1.1
  menu .$myself.mbar.b4.m1.1 \
	     -postcommand "machineMenu .$myself.mbar.b4.m1.1 doBuild"
  .$myself.mbar.b4.m1 add command -label "Make" -underline 0 \
	         -command "doMake"
  .$myself.mbar.b4.m1 add command -label "Rescan makefile" -underline 0 \
                 -command "scanMakefile"

  menubutton .$myself.mbar.b5 -text Run -menu .$myself.mbar.b5.m1 -underline 0
  menu .$myself.mbar.b5.m1 
  .$myself.mbar.b5.m1 add cascade -label "Run" -underline 0 \
                    -menu .$myself.mbar.b5.m1.1
  menu .$myself.mbar.b5.m1.1 \
	     -postcommand "machineMenu .$myself.mbar.b5.m1.1 doRun"
  .$myself.mbar.b5.m1 add command -label "Trace processing" -underline 0 \
                 -command "traceProcessing"
  .$myself.mbar.b5.m1 add command -label "List queues" -underline 5 \
                 -command "doQueue"

  menubutton .$myself.mbar.b6 -text Options -menu .$myself.mbar.b6.m1 -underline 0
  menu .$myself.mbar.b6.m1
  .$myself.mbar.b6.m1 add command -label "Set compiler parameters ..." \
    -underline 4 -command "setCompileParameter"
  .$myself.mbar.b6.m1 add command -label "Set linker parameters ..." \
    -underline 4 -command "setBuildParameter"
  .$myself.mbar.b6.m1 add command -label "Set run parameters ..." \
    -underline 4 -command "setRunParameter"


if {![ALONE]} {
  menubutton .$myself.mbar.b7 -text Errors -menu .$myself.mbar.b7.m1 -underline 0
  menu .$myself.mbar.b7.m1
  set sastemp .$myself.mbar.b7.m1
  $sastemp add command -label "Next error" \
      -command "errScrollForward"  -accelerator "Alt-n"
  $sastemp add command -label "Previous error" \
      -command "errScrollBackward" -accelerator "Alt-p"

  createToolMenu .$myself.mbar.b8
}

  menubutton .$myself.mbar.b9 -text Help -menu .$myself.mbar.b9.m1 -underline 0
  menu .$myself.mbar.b9.m1
if [ALONE] {
  .$myself.mbar.b9.m1 add command -label "on $myself" -underline 3 \
                 -command "showHelp $myself 1-$myself 1"
  .$myself.mbar.b9.m1 add separator
  .$myself.mbar.b9.m1 add command -label "on Project menu" -underline 3 \
                 -command "showHelp $myself 1.1-project 1"
  .$myself.mbar.b9.m1 add command -label "on File menu" -underline 3 \
                 -command "showHelp $myself 1.2-file 1"
  .$myself.mbar.b9.m1 add command -label "on View menu" -underline 3 \
                 -command "showHelp $myself 1.3-view 1"
  .$myself.mbar.b9.m1 add command -label "on Make menu" -underline 3 \
                 -command "showHelp $myself 1.4-make 1"
  .$myself.mbar.b9.m1 add command -label "on Run menu" -underline 3 \
                 -command "showHelp $myself 1.5-run 1"
  .$myself.mbar.b9.m1 add command -label "on Options menu" -underline 3 \
                 -command "showHelp $myself 1.6-options 1"
  .$myself.mbar.b9.m1 add separator
  .$myself.mbar.b9.m1 add command -label "on buttons" -underline 3 \
                 -command "showHelp $myself 1.8-buttons 1"
  .$myself.mbar.b9.m1 add command -label "on file display area" -underline 3 \
                 -command "showHelp $myself 1.9-display 1"
  .$myself.mbar.b9.m1 add separator
  .$myself.mbar.b9.m1 add command -label "on using Help" -underline 3 \
                 -command "showHelp general 1-help 1 "
} else {
  .$myself.mbar.b9.m1 add command -label "on $myself" -underline 3 \
                 -command "xsend tau {showHelp $myself 1-$myself 1}"
  .$myself.mbar.b9.m1 add separator
  .$myself.mbar.b9.m1 add command -label "on File menu" -underline 3 \
                 -command "xsend tau {showHelp $myself 1.2-file 1}"
  .$myself.mbar.b9.m1 add command -label "on View menu" -underline 3 \
                 -command "xsend tau {showHelp $myself 1.3-view 1}"
  .$myself.mbar.b9.m1 add command -label "on Make menu" -underline 3 \
                 -command "xsend tau {showHelp $myself 1.4-make 1}"
  .$myself.mbar.b9.m1 add command -label "on Run menu" -underline 3 \
                 -command "xsend tau {showHelp $myself 1.5-run 1}"
  .$myself.mbar.b9.m1 add command -label "on Options menu" -underline 3 \
                 -command "xsend tau {showHelp $myself 1.6-options 1}"
  .$myself.mbar.b9.m1 add command -label "on Errors menu" -underline 3 \
                 -command "xsend tau {showHelp $myself 1.7-errors 1}"
  .$myself.mbar.b9.m1 add separator
  .$myself.mbar.b9.m1 add command -label "on buttons" -underline 3 \
                 -command "xsend tau {showHelp $myself 1.8-buttons 1}"
  .$myself.mbar.b9.m1 add separator
  .$myself.mbar.b9.m1 add command -label "on using Help" -underline 3 \
                 -command "xsend tau {showHelp general 1-help 1}"
}

  if [ALONE] {
    pack  .$myself.mbar.b1 .$myself.mbar.b2 .$myself.mbar.b3 .$myself.mbar.b4 \
          .$myself.mbar.b5 .$myself.mbar.b6 -side left -padx 5
    pack  .$myself.mbar.b9 -side right -padx 5
  } else {
    pack  .$myself.mbar.b2 .$myself.mbar.b3 .$myself.mbar.b4 \
          .$myself.mbar.b5 .$myself.mbar.b6 .$myself.mbar.b7 -side left -padx 5
    pack  .$myself.mbar.b9 .$myself.mbar.b8 -side right -padx 5
  }
  pack  .$myself.mbar -side top -fill x

  # -- display area
if {[ALONE]} {
  frame .$myself.info1
  frame .$myself.info1.left
  label .$myself.info1.l1 -text "host:"
  label .$myself.info1.l2 -text "dir:"
  label .$myself.info1.l3 -text "project:"
  pack  .$myself.info1.l1 .$myself.info1.l2 .$myself.info1.l3 \
        -side top -anchor e -in .$myself.info1.left
  pack  .$myself.info1.left -side left -padx 10
  frame .$myself.info1.right
  label .$myself.info1.r1 -relief sunken -anchor w -textvariable depfile(host)
  label .$myself.info1.r2 -relief sunken -anchor w -textvariable depfile(dir)
  label .$myself.info1.r3 -relief sunken -anchor w -textvariable depfile(project)
  pack  .$myself.info1.r1 .$myself.info1.r2 .$myself.info1.r3 \
        -side top -anchor w -in .$myself.info1.right -fill x -expand 1 -ipadx 10
  pack  .$myself.info1.right -side right -padx 10 -fill x -expand 1

  frame .$myself.info2
  set   temp [frame .$myself.info2.ext]
  frame $temp.int
  frame $temp.int.top
  label $temp.int.top.filelabel -text "File"
  label $temp.int.top.langlabel -text "Language"
  pack  $temp.int.top.filelabel -side left -padx 10 -anchor nw 
  pack  $temp.int.top.langlabel -side right -padx 25 -anchor ne
  pack  $temp.int.top -side top -fill x
  frame $temp.int.bot
  listbox $temp.int.bot.list  -width 80 -height 10 \
        -background white -foreground black \
	-font -*-courier-bold-r-*-*-12-*-*-*-*-*-*-* \
	-yscrollcommand "$temp.int.sb set"
  pack  $temp.int.bot.list  -side left -anchor nw -fill y -expand 1
  pack  $temp.int.bot -side left -expand 1 -fill y -anchor n
  scrollbar $temp.int.sb -command "$temp.int.bot.list yview" 
  pack  $temp.int.sb -side left -fill y -expand 1
  pack  $temp.int -side top -anchor n -expand 1 -fill y
  pack  $temp -side top -expand 1 -anchor n -fill y

  pack  .$myself.info1 -side top -padx 10 -pady 10 -fill x
  pack  .$myself.info2 -side top -padx 10 -pady 10 -fill both -expand 1

  frame .$myself.line0 -borderwidth 1 -relief sunken
  pack  .$myself.line0  -side top -fill x -ipady 1
} ;# end if {![ALONE]}

  # -- compile and execution mode bar
  frame .$myself.mid1
  label .$myself.mid1.b0 -text "mode:"
  radiobutton .$myself.mid1.b1 -text "normal" -relief flat \
          -variable cosy_var(make) -value "norm" -command {
	  if {$cosy_var(make) != $cosy_var(old_make)} { \
	      set cosy_var(old_make) $cosy_var(make); \
	      resetMakefile; \
          }}
  radiobutton .$myself.mid1.b2 -text "profiling" -relief flat \
          -variable cosy_var(make) -value "prf-" -command {
	  if {$cosy_var(make) != $cosy_var(old_make)} { \
	      set cosy_var(old_make) $cosy_var(make); \
	      resetMakefile; \
          }}
  radiobutton .$myself.mid1.b3 -text "tracing" -relief flat \
          -variable cosy_var(make) -value "trc-" -command {
	  if {$cosy_var(make) != $cosy_var(old_make)} { \
	      set cosy_var(old_make) $cosy_var(make); \
	      resetMakefile; \
          }}
  radiobutton .$myself.mid1.b4 -text "breakpointing" -relief flat \
          -variable cosy_var(make) -value "brk-" -command {
	  if {$cosy_var(make) != $cosy_var(old_make)} { \
	      set cosy_var(old_make) $cosy_var(make); \
	      resetMakefile; \
          }}
  pack .$myself.mid1.b1 .$myself.mid1.b2 .$myself.mid1.b3 .$myself.mid1.b4 \
          -side left  -padx 5 -pady 10 -ipadx 5
  frame .$myself.line1 -borderwidth 1 -relief sunken

  # -- command button bar
  frame  .$myself.mid2
  button .$myself.mid2.build -text "Build" \
         -command {doBuild $cosy_var(make)$cosy_var(defmach) 1}
  button .$myself.mid2.make -text "Make" \
         -command {doMake $cosy_var(deftarget)}
  button .$myself.mid2.run  -text "Run"  \
         -command {doRun $cosy_var(make)$cosy_var(defmach) 1}
  button .$myself.mid2.stop -text "Stop" -command "doStop 1"
  button .$myself.mid2.exit -text "Exit" -command "exit"
  pack   .$myself.mid2.build .$myself.mid2.make  .$myself.mid2.run \
         .$myself.mid2.stop .$myself.mid2.exit \
	 -side left -padx 10 -pady 10 -ipadx 5
  frame  .$myself.line2 -borderwidth 1 -relief sunken

  pack  .$myself.mid1 -side top -anchor w
  pack  .$myself.line1 -side top -fill x -ipady 1
  pack  .$myself.mid2 -side top -anchor w
  pack  .$myself.line2 -side top -fill x -ipady 1

  # -- command line display
  frame .$myself.top
  frame .$myself.left
  label .$myself.l1 -text "executing:"
  pack  .$myself.l1 -side top -anchor e -in .$myself.left
  frame .$myself.right
  label .$myself.r1 -textvariable cosy_var(com)
  pack  .$myself.r1 -side top -anchor w -in .$myself.right
  pack  .$myself.left .$myself.right -side left -in .$myself.top
  pack  .$myself.top -side top -anchor w -padx 5 -pady 5

  # -- command output display
  frame .$myself.bot
  frame .$myself.bot.left
  frame .$myself.bot.right
  label .$myself.bot.left.lab -text "Make Targets"
  pack  .$myself.bot.left.lab -side top -fill x
  frame .$myself.bot.left.butfr
  button .$myself.bot.left.butfr.setp -text "Set Options" \
         -command {
	     if {[.$myself.bot.left.tgtlist curselection] != ""} {
		 set pf [target2progfile [.$myself.bot.left.tgtlist get active]]
		 if {$pf != "T2P_FAILED"} {
		     setLangParameter $pf [PM_SetLangOption $pf]
		 } else {
		     setBuildParameter
		 }
	     }
	 }
  button .$myself.bot.left.butfr.make -text "Make Target" \
	  -command doMake
  pack  .$myself.bot.left.butfr.setp .$myself.bot.left.butfr.make \
	  -side top -fill x
  pack  .$myself.bot.left.butfr -side top -fill x
  listbox .$myself.bot.left.tgtlist -width 16 -height 10 -background white \
	  -foreground black
  scrollbar .$myself.bot.left.s1 -orient vert -relief sunken \
	  -command ".$myself.bot.left.tgtlist yview"
  .$myself.bot.left.tgtlist configure -yscrollcommand \
	  ".$myself.bot.left.s1 set"
  label .$myself.bot.right.lab -text "Command Output"
  pack  .$myself.bot.right.lab -side top -fill x
  text  .$myself.bot.right.txt -width 70 -height 12 -background white \
	  -foreground black 
  scrollbar .$myself.bot.right.s2 -orient vert -relief sunken \
                        -command ".$myself.bot.right.txt yview"
  .$myself.bot.right.txt configure -yscrollcommand ".$myself.bot.right.s2 set"

  pack  .$myself.bot -side top -padx 10 -pady 10 -expand yes -fill both
  pack  .$myself.bot.left .$myself.bot.right -side left -expand yes -fill both \
	  -padx 5
  pack  .$myself.bot.left.s1 -side right -fill y
  pack  .$myself.bot.left.tgtlist -side right -expand yes -fill both
  pack  .$myself.bot.right.s2 -side right -fill y
  pack  .$myself.bot.right.txt -side right -expand yes -fill both

  bind .$myself.bot.left.tgtlist <Double-ButtonPress-1> {doMake}

  update idletasks
  scanMakefile
}

#
#  doBuild: execute pC++ build command
#
#     what: executable to build
#  hastype: what is fullname of executable including the type
#

proc doBuild {what {hastype 0}} {
  global cosy_var myself
  global depfile
  global BUILDDIR

  set projname [file rootname $depfile(project)]

  # -- determine type of executable
  if { $hastype } {
    set type "norm"
    regexp {(^...-|^norm)(.*)$} $what dummy type what
  } else {
    set type $cosy_var(make)
  }

  # -- execute right build command
  if { $type == "norm" } {
    doExec "$BUILDDIR/build $projname ${what}"
  } else {
    doExec "$BUILDDIR/build $projname $type${what}"
  }

  # -- set default machine executable
  set cosy_var(defmach) $what

  scanMakefile

  .$myself.mbar.b4.m1 entryconfigure 2 -state normal
  .$myself.mbar.b4.m1 entryconfigure 3 -state normal
  .$myself.mid2.make configure -state normal
  .$myself.bot.left.butfr.setp configure -state normal
  .$myself.bot.left.butfr.make configure -state normal
}

#
# doMake: execute make
#

proc doMake {{target {}}} {
  global cosy_var depfile myself
  global env

  if { $target == {} } {
    set target [string trim [.$myself.bot.left.tgtlist get active]]
  }
  
  if [info exists cosy_var(makefile)] {
    set com "make -f $cosy_var(makefile)"
  } else {
    set com "make"
  }

  # -- create timestamp file
  # -- the file should be unique at least to the user
  FileIO_exec $depfile(host) "touch /tmp/cosy-timestamp"
  after 1000  ;# Be sure the the timestamps differ by at least a sec.

  # -- execute make and highlight errors
  doExec "$com $target"
  tagErrors

  # -- check for new depfiles
  foreach progfile [PM_GetFiles] {
      if [DEBUG] {puts "COSY - CHECKING $progfile";}
      set depname ""
      append depname [file rootname $progfile] ".dep"
      set newer [FileIO_exec $depfile(host) \
	      "find $depfile(dir) -name $depname -newer /tmp/cosy-timestamp"]
      if {$newer != ""} {
	  if [DEBUG] {puts "COSY ---> NEW DEPFILE: $depname";}
	  lappend notify_for $progfile
      }
  }
  if [info exists notify_for] { PM_RecompileNotify $notify_for; }
  FileIO_exec $depfile(host) "rm /tmp/cosy-timestamp"
}


#
#   doRun: run compiled executable
#
#    what: binary to run
# hastype: what is fullname of executable including the type
#

proc doRun {what {hastype 0}} {
  global cosy_var cosy_old_val
  global depfile
  global BINDIR

  set projname [file rootname $depfile(project)]

  # Prepare language-specific parameters
  set run_params ""
  foreach lang [PM_GetProjectLangs] {
      if {[info command ${lang}_GetRunParams] != ""} {
	  append run_params [${lang}_GetRunParams $what $hastype]
      }
  }

  # -- check for binary
  set execs [getExecutables]
  if { [lsearch -exact $execs [lindex $run_params 0]] == -1 } {
    askForAction [lindex $run_params 0] $execs
    return
  }

  # -- does user-defined run command exist?
  set hostexecfunc "exec_$cosy_var(host)"
  set archexecfunc "exec_$depfile(arch)"
  if { [ info procs $hostexecfunc ] == "$hostexecfunc" } {
    doExec [$hostexecfunc [lindex $run_params 0] $numproc \
	    [lrange $run_params 1 end]]
  } elseif { [ info procs $archexecfunc ] == "$archexecfunc" } {
    doExec [$archexecfunc [lindex $run_params 0] $numproc \
	    [lrange $run_params 1 end]]
  } elseif { [ info procs exec_default ] == "exec_default" } {
    doExec [exec_default [lindex $run_params 0] $numproc \
	    [lrange $run_params 1 end]]
  } else {
    doExec "$run_params"
  }
}

#
# askForAction: ask for user input, if specified executable cannot be found
#
#         prog: executable user wanted to run
#        execs: list if available executables
#

proc askForAction {prog execs} {
  global myself
  
  toplevel .$myself.act
  wm title .$myself.act "ERROR"

  message .$myself.act.m -text "$prog does not exist! What now?" \
          -background white -foreground red -width 250 -relief sunken
  frame .$myself.act.f
  button .$myself.act.f.b1 -text " make it" -anchor w \
         -command "destroy .$myself.act; doMake $prog"
  pack .$myself.act.f.b1 -side top -fill x -ipadx 5 -pady 5
  foreach e $execs {
    button .$myself.act.f.$e -text " run $e instead" -anchor w \
           -command "destroy .$myself.act; doRun $e 1"
    pack .$myself.act.f.$e -side top -fill x -ipadx 5 -pady 5
  }
  button .$myself.act.f.b2 -text " forget it" -anchor w \
         -command "destroy .$myself.act"
  pack .$myself.act.f.b2 -side top -fill x -ipadx 5 -pady 5

  pack .$myself.act.m -side top -padx 5 -pady 5
  pack .$myself.act.f -side top -padx 10 -pady 10
}

#
# doQueue: list run queues if host has batch queue system
#

proc doQueue {} {
  global cosy_var

  # -- does user-defined queue command exist?
  set execfunc "queue_$cosy_var(host)"
  if { [ info procs $execfunc ] == "$execfunc" } {
    doExec [$execfunc]
  } elseif { [ info procs queue_default ] == "queue_default" } {
    doExec [queue_default]
  } else {
    showError "no queue command defined for this host."
  }
}

#
# setCompileParameter: user dialog window to allow compile parameter settings
#

proc setCompileParameter {} {
    global myself
    
    if {[.$myself.bot.left.tgtlist curselection] != ""} {
	set pf [target2progfile [.$myself.bot.left.tgtlist get active]]
	if {$pf != "T2P_FAILED"} {
	    setLangParameter $pf [PM_SetLangOption $pf]
	}
    } elseif {[ALONE]} {
	if {[.$myself.info2.ext.int.bot.list curselection] != ""} {
	    set pf [target2progfile [.$myself.info2.ext.int.bot.list get active]]
	    if {$pf != "T2P_FAILED"} {
		setLangParameter $pf [PM_SetLangOption $pf]
	    }
	}
    } 
}

#
# setRunParameter: user dialog window to allow run parameter settings
#

proc setRunParameter {} {
    global cosy_var
    global cosy_old_val

    set maintag [Bdb_GetMaintag]
    if {[llength $maintag] != 2} {
	showError "The project must be compiled before setting run parameters."
	return
    }
    
    set mainfile [lindex $maintag 0]
    set mainlang [PM_SetLangOption $mainfile]

    ${mainlang}_setRunParam
}


#
# setBuildParameter: user dialog window to allow build parameter settings
#

proc setBuildParameter {} {
  global cosy_var
  global cosy_old_bval

  set cosy_var(oldfocus) [focus]

  if [winfo exists .bparam] {
    .bparam.b2 invoke
  }

  toplevel .bparam
  wm title .bparam "Set Linker Parameters"
  
  frame .bparam.top
  frame .bparam.tleft
  label .bparam.tl1 -text "extra loader switches:"
  label .bparam.tl2 -text "extra object files:"
  pack .bparam.tl1 .bparam.tl2 \
	  -side top -anchor e -in .bparam.tleft -padx 5
  
  frame .bparam.tright
  entry .bparam.tr1 -textvariable cosy_old_bval(LDSWITCHES) -relief sunken \
	  -width 50
  entry .bparam.tr2 -textvariable cosy_old_bval(USEROBJS) -relief sunken \
	  -width 50
  pack .bparam.tr1 .bparam.tr2 \
	  -side top -anchor w -in .bparam.tright
  
  pack .bparam.tleft .bparam.tright -side left -in .bparam.top
  pack .bparam.top -side top -padx 5 -pady 5
  
  frame .bparam.bottom -relief groove -bd 2
  frame .bparam.frame -relief sunken -bd 1
  button .bparam.b1 -text "set" -command {
      foreach n [array names cosy_old_bval] {
	  set cosy_var($n) $cosy_old_bval($n)
      }
      focus $cosy_var(oldfocus)
      PM_SetProjectOption \
	      "LDSWITCHES=$cosy_var(LDSWITCHES)@USEROBJS=$cosy_var(USEROBJS)@"
      destroy .bparam
      askforReBuild
  }
  button .bparam.b2 -text "cancel" -command {
      foreach n [array names cosy_old_bval] {
	  if [info exists cosy_var($n)] {set cosy_old_bval($n) $cosy_var($n)} }
	  focus $cosy_var(oldfocus)
	  destroy .bparam
      }
  pack .bparam.b1 -side top -ipadx 5 -padx 5 -pady 5 -in .bparam.frame
  pack .bparam.frame -side left -padx 15 -pady 10 -in .bparam.bottom
  pack .bparam.b2 -side right -ipadx 5 -padx 15 -in .bparam.bottom
  pack .bparam.bottom -side bottom -fill x -expand yes
      
  bind .bparam.tr1 <Return> {.bparam.b1 invoke}
  bind .bparam.tr2 <Return> {.bparam.b1 invoke}
}


set cosy_var(param_file) ""
set cosy_var(param_lang) ""

#
# setLangParameter: user dialog window to allow language-specific
#                   build parameter settings
#

proc setLangParameter {progfile lang} {
  global cosy_var
  global cosy_old_bval

  set cosy_var(oldfocus) [focus]
  set cosy_var(param_file) $progfile
  set cosy_var(param_lang) $lang

  if [winfo exists .bparam] {
    .bparam.b2 invoke
  }

  # Get the parameters from the PM
  foreach opt [Lang_GetCompileOpts $lang] {
      set cosy_old_bval($opt) ""
  }
  set fileopts [split [string trim [PM_SetCompileOption $progfile] " @"] "@"]
  foreach opt $fileopts {
      set macro [string range $opt 0 [expr [string first "=" $opt] - 1]]
      set value [string range $opt [expr [string first "=" $opt] + 1] end]
      set cosy_old_bval($macro) $value
  }

  toplevel .bparam
  wm title .bparam "Set $lang Compilation Parameters"
  
  frame .bparam.top
  label .bparam.top.lab -text "$lang compile parameters for $progfile"
  pack .bparam.top.lab -side top -pady 10
  frame .bparam.tleft
  frame .bparam.tright

  set idx 0
  foreach opt [Lang_GetCompileOpts $lang] {
      label .bparam.tl${idx} \
	      -text [lindex [Lang_GetCompileOptsDesc $lang] $idx]
      pack .bparam.tl${idx} -side top -anchor e -in .bparam.tleft -padx 5
      entry .bparam.tr${idx} -textvariable cosy_old_bval($opt) -relief sunken \
	      -width 50
      pack .bparam.tr${idx} -side top -anchor w -in .bparam.tright
      bind .bparam.tr${idx} <Return> ".bparam.b${idx} invoke"
      incr idx
  }

  pack .bparam.tleft .bparam.tright -side left -in .bparam.top
  pack .bparam.top -side top -padx 5 -pady 5

  frame .bparam.bottom -relief groove -bd 2
  frame .bparam.frame -relief sunken -bd 1
  button .bparam.b1 -text "set" -command {
      foreach n [array names cosy_old_bval] {
	  set cosy_var($n) $cosy_old_bval($n)
      }
      focus $cosy_var(oldfocus)
      set optstr ""
      foreach opt [Lang_GetCompileOpts $cosy_var(param_lang)] {
	  append optstr "$opt=$cosy_var($opt)@"
      }
      PM_SetCompileOption $cosy_var(param_file) $optstr
      destroy .bparam
      askforReBuild
    }
    button .bparam.b2 -text "cancel" -command {
      foreach n [array names cosy_old_bval] {
        if [info exists cosy_var($n)] {set cosy_old_bval($n) $cosy_var($n)} }
      focus $cosy_var(oldfocus)
      destroy .bparam
    }
    pack .bparam.b1 -side top -ipadx 5 -padx 5 -pady 5 -in .bparam.frame
    pack .bparam.frame -side left -padx 15 -pady 10 -in .bparam.bottom
    pack .bparam.b2 -side right -ipadx 5 -padx 15 -in .bparam.bottom
    pack .bparam.bottom -side bottom -fill x -expand yes
}



#
# setTrcStuff: auxilary procedure for setTrcProcParameter
#

set cosy_exts(alog) "log"
set cosy_exts(SDDF) "trf"
set cosy_exts(dump) "txt"
set cosy_exts(pv)   "pv"

proc setTrcStuff {} {
  global cosy_old_tval cosy_exts

  if { $cosy_old_tval(format) == "pv" } {
    .tparam.row5.b1 configure -state normal
    .tparam.row5.b2 configure -state normal
  } else {
    .tparam.row5.b1 configure -state disabled
    .tparam.row5.b2 configure -state disabled
  }
  set cosy_old_tval(otrace) \
    "[file rootname $cosy_old_tval(otrace)].$cosy_exts($cosy_old_tval(format))"
}


proc traceProcessing {} {
  global cosy_var
  global cosy_old_tval
  global depfile cosy_trfdef
  global REMSH

  # -- create / raise window
  set cosy_var(oldfocus) [focus]
  if [winfo exists .tparam] {
    raise .tparam
  } else {
    toplevel .tparam
    wm title .tparam "Trace Postprocessing"
    wm minsize .tparam 500 300

    frame .tparam.top
    frame .tparam.row1
    label .tparam.row1.la -text "input traces:" -width 15 -anchor e
    listbox .tparam.row1.li -width 40 -height 8 -relief sunken \
            -yscrollcommand ".tparam.row1.sb set"
    scrollbar .tparam.row1.sb -orient vert -relief sunken \
              -command ".tparam.row1.li yview"
    frame .tparam.row1.f
    button .tparam.row1.f.b1 -text "reset" -width 7 -command {
      .tparam.row1.li delete 0 end
      eval .tparam.row1.li insert end $cosy_var(tfiles)
    }
    button .tparam.row1.f.b2 -text "delete" -width 7 -command {
      set idx [.tparam.row1.li curselecton]
      set ridx ""
      foreach i $idx { set ridx "$i $ridx" }
      foreach i $ridx { .tparam.row1.li delete $i }
    }
    pack .tparam.row1.f.b1 .tparam.row1.f.b2 -side top
    pack .tparam.row1.la -side left -padx 5
    pack .tparam.row1.li -side left -padx 5 -fill both -expand 1
    pack .tparam.row1.li .tparam.row1.sb -side left -padx 5 -fill y -expand 1
    pack .tparam.row1.f -side left -padx 5 -anchor n

    frame .tparam.row2
    label .tparam.row2.la -text "merge:" -width 15 -anchor e
    checkbutton .tparam.row2.cb -text "adjust timestamps" \
                -relief flat -variable cosy_old_tval(adjust) \
                -onvalue " -a" -offvalue ""
    pack .tparam.row2.la .tparam.row2.cb -side left -padx 5

    frame .tparam.row3
    label .tparam.row3.la -text "global trace:" -width 15 -anchor e
    entry .tparam.row3.ey -width 40 -textvariable cosy_old_tval(gtrace) \
          -relief sunken
    pack .tparam.row3.la .tparam.row3.ey -side left -padx 5
    
    frame .tparam.row4
    label .tparam.row4.la -text "convert:" -width 15 -anchor e
    radiobutton .tparam.row4.b1 -text "alog" -relief flat \
                -variable cosy_old_tval(format) -value "alog" \
                -command setTrcStuff
    radiobutton .tparam.row4.b2 -text "SDDF" -relief flat \
                -variable cosy_old_tval(format) -value "SDDF" \
                -command setTrcStuff
    radiobutton .tparam.row4.b3 -text "dump" -relief flat \
                -variable cosy_old_tval(format) -value "dump" \
                -command setTrcStuff
    radiobutton .tparam.row4.b4 -text "pv" -relief flat \
                -variable cosy_old_tval(format) -value "pv" \
                -command setTrcStuff
    pack .tparam.row4.la .tparam.row4.b1 .tparam.row4.b2 \
         .tparam.row4.b3 .tparam.row4.b4  -side left -padx 5

    frame .tparam.row5
    label .tparam.row5.la -text "" -width 15 -anchor e
    checkbutton .tparam.row5.b1 -text "compact" -relief flat \
                -variable cosy_old_tval(pvcompact) \
                -onvalue " -compact" -offvalue ""
    checkbutton .tparam.row5.b2 -text "no data access" -relief flat \
                -variable cosy_old_tval(pvnocomm) \
                -onvalue " -nocomm" -offvalue ""
    pack .tparam.row5.la .tparam.row5.b1 .tparam.row5.b2 -side left -padx 5

    frame .tparam.row6
    label .tparam.row6.la -text "output trace:" -width 15 -anchor e
    entry .tparam.row6.ey -width 40 -textvariable cosy_old_tval(otrace) \
          -relief sunken
    pack .tparam.row6.la .tparam.row6.ey -side left -padx 5
    
    pack .tparam.row1 -side top -pady 5 -anchor w -fill both -expand 1
    pack .tparam.row2 .tparam.row3 .tparam.row4 \
         .tparam.row5 .tparam.row6 -side top -pady 5 -anchor w
 
    pack .tparam.top -side top -padx 5 -pady 5

    frame .tparam.bottom -relief groove -bd 2
    frame .tparam.frame -relief sunken -bd 1
    button .tparam.b1 -text "do it" -command {
      foreach n [array names cosy_old_tval] {
        set cosy_var($n) $cosy_old_tval($n)
      }
      focus $cosy_var(oldfocus)
      doTraceProcessing
      destroy .tparam
    }
    button .tparam.b2 -text "cancel" -command {
      foreach n [array names cosy_old_tval] {
        if [info exists cosy_var($n)] {set cosy_old_tval($n) $cosy_var($n)} }
      focus $cosy_var(oldfocus)
      destroy .tparam
    }
    pack .tparam.b1 -side top -ipadx 5 -padx 5 -pady 5 -in .tparam.frame
    pack .tparam.frame -side left -padx 15 -pady 10 -in .tparam.bottom
    pack .tparam.b2 -side right -ipadx 5 -padx 15 -in .tparam.bottom
    pack .tparam.bottom -side bottom -fill x 
  }

  # -- set trace related information
  if { [info exists cosy_var(tracefile)] && $cosy_var(tracefile) != "" } {
    regsub "####" $cosy_var(tracefile) "*" trcpat
  } else {
    set trcpat $cosy_trfdef($cosy_var(defmach))
  }
  .tparam.row1.li delete 0 end
  if { $depfile(host) == "localhost" } {
    set cosy_var(tfiles) [glob -nocomplain $trcpat]
  } else {
    set cosy_var(tfiles) [split [exec $REMSH $depfile(host) \
                                 -n "cd $depfile(dir); ls $trcpat"] "\n"]
  }
  eval .tparam.row1.li insert end $cosy_var(tfiles)

  if { ![info exists cosy_old_tval(gtrace)] || $cosy_old_tval(gtrace) == "" } {
    set cosy_old_tval(gtrace) "[file rootname $depfile(project)].trc"
  }
  if { ![info exists cosy_old_tval(otrace)] || $cosy_old_tval(otrace) == "" } {
    set cosy_old_tval(otrace) $cosy_old_tval(gtrace)
  }
  setTrcStuff
}

#
# doTraceProcessing: run pcxx_merge and pcxx_convert according
#                    to settings in traceProcessing
#

proc doTraceProcessing {} {
  global cosy_var
  global depfile myself

  set n [.tparam.row1.li size]
  if { $n == 0 } return

  set r $cosy_var(reset)
  if { $r != 0 } {
    set cosy_var(reset) 0
    .$myself.bot.right.txt delete 1.0 end
    set cosy_var(com) {}
  }

  for {set i 0} {$i<$n} {incr i} { lappend tfiles [.tparam.row1.li get $i] }
  set com "$depfile(root)/bin/$depfile(arch)/pcxx_merge"
  append com "$cosy_var(adjust) $tfiles $cosy_var(gtrace)"
  doExec $com

  set com "$depfile(root)/bin/$depfile(arch)/pcxx_convert"
  append com " -$cosy_var(format)"
  if { $cosy_var(format) == "pv" } {
    append com "$cosy_var(pvcompact)$cosy_var(pvnocomm)"
  }
  append com " $cosy_var(gtrace) [file rootname $depfile(project)].edf"
  append com " $cosy_var(otrace)"
  doExec $com

  set cosy_var(reset) $r
}


#
# initialize: initialize application
#
#        dir: next working directory
#       root: TAUROOT of current machine
#       arch: architecture of current machine
#

proc initialize {} {
  global depfile
  global myself
  global cosy_var cosy_old_tval cosy_old_bval

  set cosy_old_tval(gtrace) "[file rootname $depfile(project)].trc"
  set cosy_old_tval(otrace) ""

  # -- if not remote, set working directory to new application directory
  if { $depfile(host) == "localhost" } { cd $depfile(dir) }

  # -- reset command output display
  if [winfo exists .$myself.bot.right.txt] {
    if { $cosy_var(reset) } {
      untagErrors
      .$myself.bot.right.txt delete 1.0 end
    } else {
      .$myself.bot.right.txt insert end "===> switched to $depfile(host):$depfile(dir)/$depfile(project)\n"
      .$myself.bot.right.txt yview -pickplace end
    }
  }
  set cosy_var(com) ""

  # -- determine available executables of current machine
  setMachines

  #Set build and run options from PM
  set projopts [split [string trim [PM_SetProjectOption] " @"] "@"]
  foreach opt $projopts {
      set macro [string range $opt 0 [expr [string first "=" $opt] - 1]]
      set value [string range $opt [expr [string first "=" $opt] + 1] end]
      set cosy_old_bval($macro) $value
      set cosy_var($macro) $value
  }

  buildRootFilesTable
} 


proc askforReBuild {} {
    global cosy_var

    set result [tk_dialog .rebuild "COSY: Rebuild the Makefile?" \
	    "The Makefile must be rebuilt before changes take effect.  \
	    Rebuild now?" \
	    question 0 "Yes" "No"]
    if {$result == 0} {
	doBuild $cosy_var(make)$cosy_var(defmach) 1
    }
}


proc resetMakefile {} {
    global myself cosy_var cosy_targets
    
    .$myself.bot.left.tgtlist delete 0 end
    .$myself.bot.left.tgtlist insert end "<NO MAKEFILE>"
    .$myself.mbar.b4.m1 entryconfigure 2 -state disabled
    .$myself.mbar.b4.m1 entryconfigure 3 -state disabled
    .$myself.mid2.make configure -state disabled
    .$myself.bot.left.butfr.setp configure -state disabled
    .$myself.bot.left.butfr.make configure -state disabled

    set cosy_var(makefile) ""
    set cosy_targets ""
}

    
proc scanMakefile {} {
  global cosy_var cosy_targets
  global myself depfile

  .$myself.bot.left.tgtlist delete 0 end

  if {$cosy_var(makefile) == ""} {
      set dirlist [FileIO_ls $depfile(host) $depfile(dir)]
      set mfiles [lpick $dirlist {[mM]akefile*}]
      set mnum [llength $mfiles]
      if { $mnum == 0 } {
	  .$myself.bot.left.tgtlist insert end "<NO MAKEFILE>"
	  .$myself.mbar.b4.m1 entryconfigure 2 -state disabled
	  .$myself.mbar.b4.m1 entryconfigure 3 -state disabled
	  .$myself.mid2.make configure -state disabled
	  .$myself.bot.left.butfr.setp configure -state disabled
	  .$myself.bot.left.butfr.make configure -state disabled
	  return
      } elseif { $mnum == 1 } {
	  set mfile "$depfile(dir)/$mfiles"
	  set cosy_var(makefile) $mfile
      } else {
	  toplevel .$myself.mchoice
	  wm title .$myself.mchoice "Makefile Selector"
	  message .$myself.mchoice.txt -text "Multiple makefiles found!\nSelect one:" \
		  -relief raised -width 250
	  pack .$myself.mchoice.txt -fill x
	  set n 0
	  foreach m $mfiles {
	      radiobutton .$myself.mchoice.$n -text $m -variable cosy_var(makefile) \
		  -value $m -anchor w \
		  -command "grab release .$myself.mchoice; destroy .$myself.mchoice" 
	      pack .$myself.mchoice.$n -side top -padx 5 -pady 5 -fill x
	      incr n
	  }
	  if [lsearch -exact $mfiles "Makefile"] {
	      set cosy_var(makefile) "Makefile"
	  } elseif [lsearch -exact $mfiles "makefile"] {
	      set cosy_var(makefile) "makefile"
	  } else {
	      set cosy_var(makefile) [lindex $mfiles 0]
	  }
	  raise .$myself.mchoice .$myself
	  grab .$myself.mchoice
	  tkwait variable cosy_var(makefile)

	  set mfile $depfile(dir)/$cosy_var(makefile)
	  set cosy_var(makefile) $mfile
      }
  } else {
      set mfile $cosy_var(makefile)
  }

  set cosy_targets ""

  if { $mfile != "" } {
    set in [FileIO_file_open $depfile(host) $mfile r]
    while {[gets $in line] >= 0} {
      if { ([string index $line 0] != "\t") &&
           ([string index [string trimleft $line] 0] != "#")} {
        # -- no comment or commando line
        if [regexp {([^#]*):} $line dummy target] {
          # -- target definition
          if { !$cosy_var(filtarget) || ([string first "." $target] == -1) } {
	    set tmptgt [string trim $target]
	    if {[string first "." $target] >= 0} {
		lappend cosy_targets "    $tmptgt"
	    } else {
		lappend cosy_targets $tmptgt
	    }
          }
        }
      }
      set line ""
    }
    if [catch {FileIO_file_close $depfile(host) $in} errmsg] {
      showError "reading $mfile: `$errmsg'."
      exit
    }
  }

  # -- initialize make target list

  foreach t $cosy_targets {
    .$myself.bot.left.tgtlist insert end $t
  }
  if {[llength $cosy_targets] > 0} {
      set cosy_var(deftarget) [lindex $cosy_targets 0]
      .$myself.bot.left.tgtlist selection set 0
  }
}



###########################################################################
# AnsiC Specific Code
#
proc ansic_GetRunParams {what {hastype 0}} {
  global cosy_var depfile

  set projname [file rootname $depfile(project)]
  set params ""

  if [info exists cosy_var(ansic_run_params)] {
      append params $cosy_var(ansic_run_params)
  }

  return $params
}


proc ansic_setRunParam {} {
  global cosy_var
  global cosy_old_val

  set cosy_var(oldfocus) [focus]

  if [winfo exists .acrparam] {
    raise .acrparam
  } else {
    toplevel .acrparam
    wm title .acrparam "Set Run Parameters"

    frame .acrparam.top
    frame .acrparam.tleft
    label .acrparam.tl1 -text "Command line arguments:"
    pack .acrparam.tl1 -side top -anchor e -in .acrparam.tleft -padx 5

    frame .acrparam.tright
    entry .acrparam.tr1 -textvariable cosy_old_val(ansic_run_params) \
	    -relief sunken
    pack .acrparam.tr1 -side top -anchor w -in .acrparam.tright -fill x \
	    -expand yes

    pack .acrparam.tleft .acrparam.tright -side left -in .acrparam.top
    pack .acrparam.top -side top -padx 5 -pady 5 -fill x -expand yes

    frame .acrparam.bottom -relief groove -bd 2
    frame .acrparam.frame -relief sunken -bd 1
    button .acrparam.b1 -text "set" -command {
      foreach n [array names cosy_old_val] {
        set cosy_var($n) $cosy_old_val($n)
      }
      focus $cosy_var(oldfocus)
      destroy .acrparam
    }
    button .acrparam.b2 -text "cancel" -command {
      foreach n [array names cosy_old_val] {
        if [info exists cosy_var($n)] {set cosy_old_val($n) $cosy_var($n)} }
      focus $cosy_var(oldfocus)
      destroy .acrparam
    }
    pack .acrparam.b1 -side top -ipadx 5 -padx 5 -pady 5 -in .acrparam.frame
    pack .acrparam.frame -side left -padx 15 -pady 10 -in .acrparam.bottom
    pack .acrparam.b2 -side right -ipadx 5 -padx 15 -in .acrparam.bottom
    pack .acrparam.bottom -side bottom -fill x -expand yes

    bind .acrparam.tr1 <Return> {.acrparam.b1 invoke}
  }
}


#
# end of AnsiC Specific Code
###########################################################################

###########################################################################
# Pc++ Specific Code
#

#
# -- initialize 'number of processors' and 'trace filename'  parameters
#

if [info exists env(pcxx_NUMPROC)] {
  set cosy_old_val(numproc) $env(pcxx_NUMPROC);
  set cosy_var(numproc) $env(pcxx_NUMPROC);
} else {
  set cosy_old_val(numproc) 16;  # default parameter for number of nodes
  set cosy_var(numproc) 16;      # previous value of this parameter
}

if [info exists env(pcxx_TRACEFILE)] {
  set cosy_old_val(tracefile) $env(pcxx_TRACEFILE);
  set cosy_var(tracefile) $env(pcxx_TRACEFILE);
}

if [info exists env(pcxx_EVENTCLASS)] {
  set classes [split $env(pcxx_EVENTCLASS) "+"]
  foreach c $classes {
    switch -glob $c {
      B*   -
      b*   { set cosy_old_val(p_basic) "B" }
      C*   -
      c*   { set cosy_old_val(p_collacs) "C" }
      K*   -
      k*   { set cosy_old_val(p_kernel) "K" }
      R*   -
      r*   { set cosy_old_val(p_runtime) "R" }
      TI*  -
      Ti*  -
      tI*  -
      ti*  { set cosy_old_val(p_timers) "Ti" }
      TR*  -
      Tr*  -
      tR*  -
      tr*  { set cosy_old_val(p_trace) "Tr" }
      1    { set cosy_old_val(p_user1) "1" }
      2    { set cosy_old_val(p_user2) "2" }
      3    { set cosy_old_val(p_user3) "3" }
      4    { set cosy_old_val(p_user4) "4" }
      U*   -
      u*   { set cosy_old_val(p_user1) "1"
             set cosy_old_val(p_user2) "2"
             set cosy_old_val(p_user3) "3"
             set cosy_old_val(p_user4) "4" }
      P*   -
      p*   { set cosy_old_val(p_profile) "P" }
      A*   -
      a*   { set cosy_old_val(p_user1) "1"
             set cosy_old_val(p_user2) "2"
             set cosy_old_val(p_user3) "3"
             set cosy_old_val(p_user4) "4"
             set cosy_old_val(p_basic) "B"
             set cosy_old_val(p_collacs) "C"
             set cosy_old_val(p_kernel) "K"
             set cosy_old_val(p_runtime) "R"
             set cosy_old_val(p_timers) "Ti"
             set cosy_old_val(p_trace) "Tr"
             set cosy_old_val(p_profile) "P" }
    }
  }
  foreach n [array names cosy_old_val] {
    if { [regexp ^p_ $n] } {
      set cosy_var($n) $cosy_old_val($n)
    }
  }
}

#
# traceProcessing: user dialog window to allow trace postprocessing
#

set cosy_trfdef(uniproc)    "uniproc.trc"
set cosy_trfdef(symmetry)   "symmetry.*.trc"
set cosy_trfdef(ksr)        "ksr1.*.trc"
set cosy_trfdef(paragon)    "paragon.*.trc"
set cosy_trfdef(sp2)        "sp1.*.trc"
set cosy_trfdef(cm5)        "cm5.*.trc"
set cosy_trfdef(pvmexe)     "pvm.*.trc"
set cosy_trfdef(pvmNEW)     "pvm.*.trc"
set cosy_trfdef(mpi)        "mpi.*.trc"
set cosy_trfdef(mpiNEW)     "mpi.*.trc"
set cosy_trfdef(sgimp)      "sgi.*.trc"
set cosy_trfdef(ksr-ms)     "ms.trc"
set cosy_trfdef(sgimp-ms)   "ms.trc"
set cosy_trfdef(task-ms)    "ms.trc"
set cosy_trfdef(awe-ms)     "ms.trc"
set cosy_trfdef(lwp-ms)     "ms.trc"
set cosy_trfdef(pthread-ms) "ms.trc"


proc pc++_GetRunParams {what {hastype 0}} {
  global cosy_var cosy_old_val
  global depfile
  global BINDIR

  set projname [file rootname $depfile(project)]

  # -- determine type of executable
  if { $hastype } {
    set type "norm"
    regexp {(^...-|^norm)(.*)$} $what dummy type what
  } else {
    set type $cosy_var(make)
  }

  # -- set program and number of nodes to use
  if { $what == "uniproc" } {
      set numproc 1
      set param ""
  } else {
      set numproc $cosy_var(numproc)
      set param " -pcxx_NUMPROC $cosy_var(numproc)"
  }

  if { $type == "norm" } {
    set prog "$projname-$what"
  } else {
    set prog "$projname-$type$what"
  }

  switch $type {
    trc-   { # -- TF is user-specified tracefile, if specified
             set TF ""
             if { [info exists cosy_var(tracefile)] &&
                  $cosy_var(tracefile) != "" } {
               set TF " -pcxx_TRACEFILE '$cosy_var(tracefile)'"
             }

             # -- EC is user-specified event classes
             set EC ""
             foreach n [array names cosy_old_val] {
	       if [DEBUG] {puts "EventClass: $n"}
               if { [regexp ^p_ $n] && $cosy_var($n) != "" } {
                 if { $EC != "" } {
                   append EC "+$cosy_var($n)"
                 } else {
                   set EC "$cosy_var($n)"
                 }
	       if [DEBUG] {puts "EC: $EC"}
               }
             }
             if { $EC != "" } {set EC " -pcxx_EVENTCLASS $EC"}
             append param "$TF$EC"
           }
    brk-   { # for breakpointing start breezy user interface first
             exec $BINDIR/breezy &

             # then start program, with hostname if necessary
             if { $depfile(host) != "localhost" } {
               append param " -pcxx_BRKHOST $cosy_var(host)"
             }
           }
    default {}
  }

  # -- add possible user-defined run parameters?
  if { [info exists cosy_var(userparam)] && $cosy_var(userparam) != "" } {
    append param " $cosy_var(userparam)"
  }

  return "$prog$param"
}


proc pc++_setRunParam {} {
  global cosy_var
  global cosy_old_val

  set cosy_var(oldfocus) [focus]

  if [winfo exists .rparam] {
    raise .rparam
  } else {
    toplevel .rparam
    wm title .rparam "Set Run Parameters"

    frame .rparam.top
    frame .rparam.tleft
    label .rparam.tl1 -text "NUMPROC:"
    label .rparam.tl2 -text "TRACEFILE:"
    label .rparam.tl3 -text "USERPARAM:"
    pack .rparam.tl1 .rparam.tl2 .rparam.tl3 \
         -side top -anchor e -in .rparam.tleft -padx 5

    frame .rparam.tright
    entry .rparam.tr1 -textvariable cosy_old_val(numproc) -relief sunken
    entry .rparam.tr2 -textvariable cosy_old_val(tracefile) -relief sunken
    entry .rparam.tr3 -textvariable cosy_old_val(userparam) -relief sunken
    pack .rparam.tr1 .rparam.tr2 .rparam.tr3 \
         -side top -anchor w -in .rparam.tright -fill x -expand yes

    pack .rparam.tleft .rparam.tright -side left -in .rparam.top
    pack .rparam.top -side top -padx 5 -pady 5 -fill x -expand yes

    frame .rparam.mid
    label .rparam.midl1 -text "EVENTCLASS:"
    pack .rparam.midl1 -side left  -in .rparam.mid -padx 5

    frame .rparam.midleft
    checkbutton .rparam.midleft.l1 -text "basic" \
                                   -variable cosy_old_val(p_basic) \
                                   -onvalue "B" -offvalue "" -relief flat
    checkbutton .rparam.midleft.l2 -text "kernel" \
                                   -variable cosy_old_val(p_kernel) \
                                   -onvalue "K" -offvalue "" -relief flat
    checkbutton .rparam.midleft.l3 -text "runtime" \
                                   -variable cosy_old_val(p_runtime) \
                                   -onvalue "R" -offvalue "" -relief flat
    checkbutton .rparam.midleft.l4 -text "timers" \
                                   -variable cosy_old_val(p_timers) \
                                   -onvalue "Ti" -offvalue "" -relief flat
    pack .rparam.midleft.l1 .rparam.midleft.l2 .rparam.midleft.l3 \
         .rparam.midleft.l4 -side top -anchor w

    frame .rparam.midmid
    checkbutton .rparam.midmid.m1 -text "trace" \
                                  -variable cosy_old_val(p_trace) \
                                  -onvalue "Tr" -offvalue "" -relief flat
    checkbutton .rparam.midmid.m2 -text "collacs" \
                                  -variable cosy_old_val(p_collacs) \
                                  -onvalue "C" -offvalue "" -relief flat
    checkbutton .rparam.midmid.m3 -text "profile" \
                                  -variable cosy_old_val(p_profile) \
                                  -onvalue "P" -offvalue "" -relief flat
    pack .rparam.midmid.m1 .rparam.midmid.m2 .rparam.midmid.m3 \
         -side top -anchor w

    frame .rparam.midright
    checkbutton .rparam.midright.r1 -text "user1" \
                                   -variable cosy_old_val(p_user1) \
                                   -onvalue "1" -offvalue "" -relief flat
    checkbutton .rparam.midright.r2 -text "user2" \
                                   -variable cosy_old_val(p_user2) \
                                   -onvalue "2" -offvalue "" -relief flat
    checkbutton .rparam.midright.r3 -text "user3" \
                                   -variable cosy_old_val(p_user3) \
                                   -onvalue "3" -offvalue "" -relief flat
    checkbutton .rparam.midright.r4 -text "user4" \
                                   -variable cosy_old_val(p_user4) \
                                   -onvalue "4" -offvalue "" -relief flat
    pack .rparam.midright.r1 .rparam.midright.r2 .rparam.midright.r3\
         .rparam.midright.r4 -side top -anchor w

    pack .rparam.midleft .rparam.midmid .rparam.midright \
         -side left -padx 10 -pady 10 -in .rparam.mid
    pack .rparam.mid -side top

    frame .rparam.bottom -relief groove -bd 2
    frame .rparam.frame -relief sunken -bd 1
    button .rparam.b1 -text "set" -command {
      foreach n [array names cosy_old_val] {
        set cosy_var($n) $cosy_old_val($n)
      }
      focus $cosy_var(oldfocus)
      destroy .rparam
    }
    button .rparam.b2 -text "cancel" -command {
      foreach n [array names cosy_old_val] {
        if [info exists cosy_var($n)] {set cosy_old_val($n) $cosy_var($n)} }
      focus $cosy_var(oldfocus)
      destroy .rparam
    }
    pack .rparam.b1 -side top -ipadx 5 -padx 5 -pady 5 -in .rparam.frame
    pack .rparam.frame -side left -padx 15 -pady 10 -in .rparam.bottom
    pack .rparam.b2 -side right -ipadx 5 -padx 15 -in .rparam.bottom
    pack .rparam.bottom -side bottom -fill x -expand yes

    bind .rparam.tr1 <Return> {.rparam.b1 invoke}
    bind .rparam.tr2 <Return> {.rparam.b1 invoke}
    bind .rparam.tr3 <Return> {.rparam.b1 invoke}
  }
}


#
# end of pC++ Specific Code
###########################################################################


###########################################################################
# HPC++ Specific Code
#
proc hpc++_GetRunParams {what {hastype 0}} {
  global cosy_var depfile

  set projname [file rootname $depfile(project)]
  set params ""

  # -- determine type of executable
  if { $hastype } {
    set type "norm"
    regexp {(^...-|^norm)(.*)$} $what dummy type what
  } else {
    set type $cosy_var(make)
  }

  if { $type == "norm" } {
    set prog "$projname-$what"
  } else {
    set prog "$projname-$type$what"
  }

  if [info exists cosy_var(hpc++_run_params)] {
      append params $cosy_var(hpc++_run_params)
  }

  return "$prog $params"
}


proc hpc++_setRunParam {} {
  global cosy_var
  global cosy_old_val

  set cosy_var(oldfocus) [focus]

  if [winfo exists .hpcxxrparam] {
    raise .hpcxxrparam
  } else {
    toplevel .hpcxxrparam
    wm title .hpcxxrparam "Set Run Parameters"

    frame .hpcxxrparam.top
    frame .hpcxxrparam.tleft
    label .hpcxxrparam.tl1 -text "Command line arguments:"
    pack .hpcxxrparam.tl1 -side top -anchor e -in .hpcxxrparam.tleft -padx 5

    frame .hpcxxrparam.tright
    entry .hpcxxrparam.tr1 -textvariable cosy_old_val(hpc++_run_params) \
	    -relief sunken
    pack .hpcxxrparam.tr1 -side top -anchor w -in .hpcxxrparam.tright -fill x \
	    -expand yes

    pack .hpcxxrparam.tleft .hpcxxrparam.tright -side left -in .hpcxxrparam.top
    pack .hpcxxrparam.top -side top -padx 5 -pady 5 -fill x -expand yes

    frame .hpcxxrparam.bottom -relief groove -bd 2
    frame .hpcxxrparam.frame -relief sunken -bd 1
    button .hpcxxrparam.b1 -text "set" -command {
      foreach n [array names cosy_old_val] {
        set cosy_var($n) $cosy_old_val($n)
      }
      focus $cosy_var(oldfocus)
      destroy .hpcxxrparam
    }
    button .hpcxxrparam.b2 -text "cancel" -command {
      foreach n [array names cosy_old_val] {
        if [info exists cosy_var($n)] {set cosy_old_val($n) $cosy_var($n)} }
      focus $cosy_var(oldfocus)
      destroy .hpcxxrparam
    }
    pack .hpcxxrparam.b1 -side top -ipadx 5 -padx 5 -pady 5 -in .hpcxxrparam.frame
    pack .hpcxxrparam.frame -side left -padx 15 -pady 10 -in .hpcxxrparam.bottom
    pack .hpcxxrparam.b2 -side right -ipadx 5 -padx 15 -in .hpcxxrparam.bottom
    pack .hpcxxrparam.bottom -side bottom -fill x -expand yes

    bind .hpcxxrparam.tr1 <Return> {.hpcxxrparam.b1 invoke}
  }
}


#
# end of HPC++ Specific Code
###########################################################################



proc Tool_AcceptChanges {progfiles flag} {
    global myself depfile

    switch $flag {

        d {
            foreach pf $progfiles { Cgm_RemoveDep $pf;}
            # Delete a file
	    if [ALONE] SetFileDisplay
	    askforReBuild
            # Remove from GUI
        }
        a {
            # Add a file
	    if [ALONE] SetFileDisplay
	    askforReBuild
	    buildRootFilesTable
	}
        u {
            # Update a file
	    return
        }
	p {
	    #Modify project information. 
	    set temp [PM_Status]
	    if {![string match $temp "UNDEFINED"]} {
		set depfile(project) [lindex $temp 0]
		set depfile(host)    [lindex $temp 1]
		set depfile(arch)    [lindex $temp 2]
		set depfile(root)    [lindex $temp 3]
       		set depfile(dir)     [lindex $temp 4]
	    } else {
		showError "There is no project to modify."
	    }
	    # Check for language-tool compatibility
	    if {![Lang_CheckCompatibility [PM_GetProjectLangs] $myself]} {
		showError "$myself is not compatible with the project language(s)."
		exit
	    }
	    
	    #reset globals
	    global cosy_var cosy_old_tval
	    set cosy_var(make) "norm";     # program "type": prf-, trc-, brk-
	    set cosy_var(old_make) "norm";
	    set cosy_var(makefile) "";     # name of the makefile
	    set cosy_var(warn) 1;          # show warnings?
	    set cosy_var(com) "";          # holds current command to execute
	    set cosy_var(reset) 1;         # reset text output window?
	    set cosy_var(deftarget) "";    # default make target
	    set cosy_var(filtarget) 0;     # filter out target which contain "."
	    set cosy_var(num_errs) 0;      # number of errors found
	    set cosy_old_tval(adjust) " -a";
	    set cosy_old_tval(format) "dump";

	    if [ALONE] SetFileDisplay
	    initialize
	    scanMakefile
	}
	e {
	    #This is a flag for updating during execution. No action is needed here.
	}
    }
}


# ------------
# -- main code
# ------------
set ALONE_SET 0
switch $argc {
    0   {
	set parg [pwd]
    }
    1   {
	if {[lindex $argv 0] == "-sa" } {
	    set ALONE_SET 1
	    set parg [pwd]
        } else {
  	    set parg [lindex $argv 0]
	    if {[file extension $parg] != ".pmf"} {
	        set parg "$parg.pmf"
	    }
	}
    }
    2   {
	if {[lindex $argv 0] == "-sa" } {
	    set ALONE_SET 1
	    set parg [lindex $argv 1]
	    if {[file extension $parg] != ".pmf"} {
		set parg "$parg.pmf"
	    }
	} else {
	    puts stderr "usage: $myself [-sa] \[\[host:\]projFile \| \[host:\]directory\]"
	    exit.old
	}
    }
    default {
	puts stderr "usage: $myself [-sa] \[\[host:\]projFile \| \[host:\]directory\]"
	exit.old
    }
}

# Init the project manager (taud)
launchTauDaemon -waitfor
PM_AddTool $myself

# -- read system rc file
source "$TAUDIR/inc/cosyrc"

# -- read user rc file
if [ file readable $env(HOME)/.cosyrc ] {
  if [catch "source $env(HOME)/.cosyrc" errmsg] {
    tkerror "error reading $env(HOME)/.cosyrc"
    if [winfo exists .tkerrorTrace] {
      tkwait window .tkerrorTrace
    }
  }
}

# Initialize the project
# Coordinate w/ PM
set pm_status [PM_Status]
if {[lindex $pm_status 0] == "NO_PROJECT"} {
    # Open or create a project
    set colon [string first ":" $parg]
    if {$colon > 0} {
	set hostarg [string range $parg 0 [expr $colon - 1]]
	set patharg [string range $parg [expr $colon + 1] end]
    } else {
	set hostarg localhost
	set patharg $parg
    }

    set projfile [PM_OpenProject $patharg $hostarg]
    if {$projfile == "NO_PROJECT"} {
	showError "No project opened!"
	exit
    }
    set pm_status [PM_Status]
}

set depfile(project) [lindex $pm_status 0]
set depfile(host)    [lindex $pm_status 1]
set depfile(arch)    [lindex $pm_status 2]
set depfile(root)    [lindex $pm_status 3]
set depfile(dir)     [lindex $pm_status 4]

# Check for language-tool compatibility
if {![Lang_CheckCompatibility [PM_GetProjectLangs] $myself]} {
    showError "$myself is not compatible with the project language(s)."
    exit
}

FileIO_INIT $depfile(root)

# Tool Init
initialize
createWindow
SetFileDisplay
wm protocol .$myself WM_DELETE_WINDOW doStop
bind all Q exit
if {![ALONE]} {
    launchTAU
}
removeMessage

