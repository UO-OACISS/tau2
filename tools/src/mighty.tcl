#*********************************************************************#
#*                TAU/pC++/Sage++  Copyright (C) 1994                *#
#*                        Jerry Manic Saftware                       *#
#*  Indiana University  University of Oregon  University of Rennes   *#
#*********************************************************************#
displayMessage hourglass "Loading $myself..."

source "$TAUDIR/inc/cosyutil.tcl"

set depfile(host) "localhost";  # local or remotehost?
set depfile(dir) ".";           # working directory
set cosy_var(warn) 0;           # show warnings?
set cosy_var(com) "";           # holds current command to execute
set cosy_var(reset) 1;          # reset text output window?
set cosy_var(num_errs) 0;       # number of errors found
set cosy_var(macros) "";        # list of makefile macros
set cosy_var(deftarget) "";     # default make target
set cosy_var(filtarget) 1;      # filter out target which contain "."

#
# createWindow: create main application window
#

proc createWindow {} {
  global TAUDIR
  global myself
  global cosy_var

  toplevel .$myself
  wm title .$myself "MIGHTY"
  wm minsize .$myself 50 50
  wm iconbitmap .$myself @$TAUDIR/xbm/$myself.xbm

  # -- menu bar
  frame .$myself.mbar -relief raised -borderwidth 2
  menubutton .$myself.mbar.b1 -text File -menu .$myself.mbar.b1.m1 -underline 0
  menu .$myself.mbar.b1.m1
  .$myself.mbar.b1.m1 add cascade -label "make" -underline 0 \
                          -menu .$myself.mbar.b1.m1.2
  .$myself.mbar.b1.m1 add command -label "Run" -underline 0 -command doRun
  .$myself.mbar.b1.m1 add command -label "list directory" -underline 5 \
                          -command "doExec {ls -l}"
  .$myself.mbar.b1.m1 add command -label "clear window" -underline 0 \
                    -command "untagErrors; .$myself.txt delete 1.0 end; set cosy_var(com) {}"
  .$myself.mbar.b1.m1 add command -label "rescan makefile" -underline 0 \
                    -command "scanMakefile"
  .$myself.mbar.b1.m1 add separator
  .$myself.mbar.b1.m1 add command -label "Exit" -underline 0 -command "exit"

  menu .$myself.mbar.b1.m1.1
  menu .$myself.mbar.b1.m1.3
  menu .$myself.mbar.b1.m1.2

  menubutton .$myself.mbar.b2 -text Options -menu .$myself.mbar.b2.m1 \
                              -underline 0
  menu .$myself.mbar.b2.m1
  .$myself.mbar.b2.m1 add command -label "set make parameters ..." \
                 -underline 4 -command setMakeParameter
  .$myself.mbar.b2.m1 add command -label "set run parameters ..." \
                 -underline 4 -command setRunParameter
  .$myself.mbar.b2.m1 add separator
  .$myself.mbar.b2.m1 add checkbutton -label "show warnings" \
                 -underline 0 -variable cosy_var(warn) -onvalue 1 -offvalue 0
  .$myself.mbar.b2.m1 add checkbutton -label "reset output" \
                 -underline 0 -variable cosy_var(reset) -onvalue 1 -offvalue 0
  .$myself.mbar.b2.m1 add checkbutton -label "filter targets" \
                 -underline 0 -variable cosy_var(filtarget) -onvalue 1 -offvalue 0

  menubutton .$myself.mbar.b3 -text Help -menu .$myself.mbar.b3.m1 -underline 0
  menu .$myself.mbar.b3.m1
  .$myself.mbar.b3.m1 add command -label "on $myself" -underline 3 \
                 -command "xsend tau \[list showHelp $myself 1-$myself 1\]"
  .$myself.mbar.b3.m1 add separator
  .$myself.mbar.b3.m1 add command -label "on File menu" -underline 3 \
                 -command "xsend tau \[list showHelp $myself 1.1-file 1\]"
  .$myself.mbar.b3.m1 add command -label "on Options menu" -underline 3 \
                 -command "xsend tau \[list showHelp $myself 1.2-options 1\]"
  .$myself.mbar.b3.m1 add command -label "on Buttons" -underline 3 \
                 -command "xsend tau \[list showHelp $myself 1.3-buttons 1\]"
  .$myself.mbar.b3.m1 add separator
  .$myself.mbar.b3.m1 add command -label "on using help" -underline 3 \
                 -command "xsend tau {showHelp general 1-help 1}"

  createToolMenu .$myself.mbar.b4

  # -- command button bar
  frame .$myself.mid2
  button .$myself.mid2.make -text "make" -command doMake
  button .$myself.mid2.run  -text "run" -command doRun
  button .$myself.mid2.stop -text "stop" -command "doStop 1"
  button .$myself.mid2.exit -text "exit" -command "exit"
  pack .$myself.mid2.make .$myself.mid2.run \
       .$myself.mid2.stop .$myself.mid2.exit \
       -side left -padx 5 -pady 10 -ipadx 5
  frame .$myself.line2 -borderwidth 1 -relief sunken

  # -- command line display
  frame .$myself.top
  frame .$myself.left
  label .$myself.l1 -text "executing:"
  pack .$myself.l1 -side top -anchor e -in .$myself.left

  frame .$myself.right
  label .$myself.r1 -textvariable cosy_var(com)
  pack .$myself.r1 -side top -anchor w -in .$myself.right
  pack .$myself.left .$myself.right -side left -in .$myself.top

  # -- command output display
  text .$myself.txt -width 80 -height 12 -background white -foreground black 
  scrollbar .$myself.s1 -orient vert -relief sunken \
                        -command ".$myself.txt yview"
  .$myself.txt configure -yscrollcommand ".$myself.s1 set"

  pack .$myself.mbar.b1 .$myself.mbar.b2 -side left -padx 5
  pack .$myself.mbar.b3 .$myself.mbar.b4 -side right -padx 5
  pack .$myself.mbar -side top -fill x

  pack .$myself.mid2 -side top -anchor w
  pack .$myself.line2 -side top -fill x -ipady 1
  pack .$myself.top -side top -anchor w -padx 5 -pady 5
  pack .$myself.s1 -side right -fill y
  pack .$myself.txt -side top -padx 10 -pady 10 -expand yes -fill both
}

#
# doMake: execute make
#

proc doMake {{target {}}} {
  global cosy_var mighty_macros depfile
  global env

  if { $target == {} } {
    set target $cosy_var(deftarget)
  }
  
  # -- evaluate macros in targets
  while { [regexp {\$[(\{]([^)\}]*)[)\}]} $target dummy m] } {
    if { ! [info exists cosy_var($m)] } {
      if { ! [info exists env($m)] } {
        showError "undefined variable \$$m"
        return
      } else {
        set rval $env($m)
      }
    } else {
      set rval $cosy_var($m)
    }
    regsub {\$[(\{][^)\}]*[)\}]} $target $rval newtarget
    set target $newtarget
  }

  if [info exists cosy_var(makefile)] {
    set com "make -f $cosy_var(makefile)"
  } else {
    set com "make"
  }
  foreach m $cosy_var(macros) {
    # -- if there are user defined make macros and they were changed
    # -- (different vale than default value)
    # -- then add to make command
    if { [info exists cosy_var($m)] && $cosy_var($m) != "" && \
         $cosy_var($m) != $mighty_macros($m) } {
      append com " $m='$cosy_var($m)'"
    }
  }

  # -- create timestamp file, so we can check whether "make" produced
  # -- new depfile.
  # -- Actually, the timestamp file should be written to /tmp, so we can
  # -- avoid problems when we don't have write access to the current
  # -- directory; but the current directory might be located on a different
  # -- machine than /tmp (through NFS) having a non-synchronized time (sigh!)
  if { ! [catch {set stp [open ".mighty-stamp" w]} ] } {
    close $stp
    set stp_time [file mtime ".mighty-stamp"]
  } else {
    set stp_time 999999999
  }

  # -- execute make and highlight errors
  doExec "$com $target"
  tagErrors

  # -- check for new depfiles
  foreach f [glob .* *] {
    set t [file mtime $f]
    if { ($t >= $stp_time) && ([file extension $f] == ".dep") } {
      if { $depfile(host) == "localhost" } {
        xsend tau "loadDepUncond $depfile(dir)/$f 1 0"
      } else {
        xsend tau "loadDepUncond $depfile(host):$depfile(dir)/$f 0 0"
      }
    }
  }
}

#
#   doRun: run compiled executable
#

proc doRun {} {
  global cosy_var

  # -- get program to execute
  if { [info exists cosy_var(runprog)] && $cosy_var(runprog) != "" } {
    set cmd $cosy_var(runprog)

    # -- add possible user-defined run parameters?
    if { [info exists cosy_var(runparam)] && $cosy_var(runparam) != "" } {
      append cmd " $cosy_var(runparam)"
    }

    doExec $cmd
  }
}

#
# setRunParameter: user dialog window to allow run parameter settings
#

proc setRunParameter {} {
  global cosy_var
  global mighty_old_val

  set cosy_var(oldfocus) [focus]

  if [winfo exists .rparam] {
    raise .rparam
  } else {
    toplevel .rparam
    wm title .rparam "Set Run Parameters"

    frame .rparam.top
    frame .rparam.tleft
    label .rparam.tl1 -text "program:"
    label .rparam.tl2 -text "parameters:"
    pack .rparam.tl1 .rparam.tl2 -side top -anchor e -in .rparam.tleft -padx 5

    frame .rparam.tright
    entry .rparam.tr1 -textvariable mighty_old_val(runprog) \
                      -relief sunken -width 50
    entry .rparam.tr2 -textvariable mighty_old_val(runparam) \
                      -relief sunken -width 50
    pack .rparam.tr1 .rparam.tr2 \
         -side top -anchor w -in .rparam.tright -fill x -expand yes

    pack .rparam.tleft .rparam.tright -side left -in .rparam.top
    pack .rparam.top -side top -padx 5 -pady 5 -fill x -expand yes

    frame .rparam.bottom -relief groove -bd 2
    frame .rparam.frame -relief sunken -bd 1
    button .rparam.b1 -text "set" -command {
      foreach n [array names mighty_old_val] {
        set cosy_var($n) $mighty_old_val($n)
      }
      focus $cosy_var(oldfocus)
      destroy .rparam
    }
    button .rparam.b2 -text "cancel" -command {
      foreach n [array names mighty_old_val] {
        if [info exists cosy_var($n)] {set mighty_old_val($n) $cosy_var($n)} }
      focus $cosy_var(oldfocus)
      destroy .rparam
    }
    pack .rparam.b1 -side top -ipadx 5 -padx 5 -pady 5 -in .rparam.frame
    pack .rparam.frame -side left -padx 15 -pady 10 -in .rparam.bottom
    pack .rparam.b2 -side right -ipadx 5 -padx 15 -in .rparam.bottom
    pack .rparam.bottom -side bottom -fill x -expand yes

    bind .rparam.tr1 <Return> {.rparam.b1 invoke}
    bind .rparam.tr2 <Return> {.rparam.b1 invoke}
  }
}


#
# setMakeParameter: user dialog window to allow make macro settings
#

proc setMakeParameter {} {
  global cosy_var
  global mighty_old_mval
  global mighty_macros

  set cosy_var(oldfocus) [focus]

  if [winfo exists .bparam] {
    raise .bparam
  } else {
    toplevel .bparam
    wm title .bparam "Set Make Parameters"

    frame .bparam.top
    set i 1
    foreach m $cosy_var(macros) {
      frame .bparam.f$i
      label .bparam.f$i.l -text "$m:" -width $cosy_var(macrolen) -anchor e
      pack .bparam.f$i.l -side left -anchor e -padx 5
      entry .bparam.f$i.e -textvariable mighty_old_mval($m) -relief sunken \
                         -width 50
      pack .bparam.f$i.e -side left -anchor w
      bind .bparam.f$i.e <Return> {.bparam.b1 invoke}
      pack .bparam.f$i -side top -in .bparam.top
      incr i
    }
    pack .bparam.top -side top -padx 5 -pady 5

    if { $i > 1 } {
      frame .bparam.line -borderwidth 1 -relief sunken
      pack .bparam.line -side top -fill x -ipady 1
    }
    frame .bparam.f$i
    label .bparam.f$i.l -text "default target:" -width $cosy_var(macrolen) \
                        -anchor e
    pack .bparam.f$i.l -side left -anchor e -padx 5
    entry .bparam.f$i.e -textvariable mighty_old_mval(deftarget) -relief sunken \
                       -width 50
    pack .bparam.f$i.e -side left -anchor w 
    bind .bparam.f$i.e <Return> {.bparam.b1 invoke}
    pack .bparam.f$i -side top -pady 5

    frame .bparam.bottom -relief groove -bd 2
    frame .bparam.frame -relief sunken -bd 1
    button .bparam.b1 -text "set" -command {
      foreach n [array names mighty_old_mval] {
        set cosy_var($n) $mighty_old_mval($n)
      }
      focus $cosy_var(oldfocus)
      destroy .bparam
    }
    button .bparam.b2 -text "cancel" -command {
      foreach n [array names mighty_old_mval] {
        if [info exists cosy_var($n)] {set mighty_old_mval($n) $cosy_var($n)} }
      focus $cosy_var(oldfocus)
      destroy .bparam
    }
    pack .bparam.b1 -side top -ipadx 5 -padx 5 -pady 5 -in .bparam.frame
    pack .bparam.frame -side left -padx 15 -pady 10 -in .bparam.bottom
    pack .bparam.b2 -side right -ipadx 5 -padx 15 -in .bparam.bottom
    pack .bparam.bottom -side bottom -fill x -expand yes
  }
}

#
# scanMakefile: scan existing makefile for targets and macros
#

proc scanMakefile {} {
  global cosy_var mighty_macros mighty_targets
  global myself mighty_old_mval

  set mfiles [glob -nocomplain {[mM]akefile*}]
  set mnum [llength $mfiles]
  if { $mnum == 0 } {
    showError "mighty: no makefile"
    exit
  } elseif { $mnum == 1 } {
    set mfile $mfiles
  } else {
    toplevel .$myself.mchoice
    wm title .$myself.mchoice "Makefile Selector"
    message .$myself.mchoice.txt -text "Multiple makefiles found!\nSelect one:" \
            -relief raised -width 250
    pack .$myself.mchoice.txt -fill x
    set n 0
    foreach m $mfiles {
      radiobutton .$myself.mchoice.$n -text $m -variable cosy_var(makefile) \
                  -value $m -command "destroy .$myself.mchoice" -anchor w
      pack .$myself.mchoice.$n -side top -padx 5 -pady 5 -fill x
      incr n
    }
    if [lsearch -exact $mfiles "makefile"] {
      set cosy_var(makefile) "makefile"
    } elseif [lsearch -exact $mfiles "Makefile"] {
      set cosy_var(makefile) "Makefile"
    } else {
      set cosy_var(makefile) [lindex $mfiles 0]
    }
    raise .$myself.mchoice .$myself
    tkwait variable cosy_var(makefile)
    set mfile $cosy_var(makefile)
    wm title .$myself "MIGHTY - $mfile"
  }

  set cosy_var(macrolen) [string length "default target:"]
  set mighty_targets ""
  set cosy_var(macros) ""

  if { $mfile != "" } {
    set in [open $mfile r]
    while {[gets $in line] >= 0} {
      if { ([string index $line 0] != "\t") &&
           ([string index [string trimleft $line] 0] != "#")} {
        # -- no comment or commando line
        if [regexp {([^#]*):} $line dummy target] {
          # -- target definition
          if { !$cosy_var(filtarget) || ([string first "." $target] == -1) } {
            lappend mighty_targets [string trim $target]
          }
        } elseif [regexp {([^#=]*)=(.*)} $line dummy macro def] {
          # -- macro definition: need to handle continuation lines in definition
          set definition ""
          set l [expr [string length $def] - 1]
          while { [string index $def $l] == "\\" } {
            set d [string range $def 0 [expr $l - 1]]
            # -- check for comments
            if { [set c [string first "#" $d]] != -1 } {
              append definition " [string trim [string range $d 0 [expr $c -1]]]"
            } else {
              append definition " [string trim $d]"
            }
            if {[gets $in def] >= 0} {
              set l [expr [string length $def] - 1]
            } else {
              break
            }
          }
          # -- check for comments
          if { [set c [string first "#" $def]] != -1 } {
            append definition " [string trim [string range $def 0 [expr $c -1]]]"
          } else {
            append definition " [string trim $def]"
          }

          set m [string trim $macro]
          set mighty_macros($m) [string trimleft $definition]
          lappend cosy_var(macros) $m
          set mighty_old_mval($m) $mighty_macros($m)
          set cosy_var($m) $mighty_macros($m)

          if { [string length "$m:"] > $cosy_var(macrolen) } {
            set cosy_var(macrolen) [string length "$m:"]
          }
        }
      }
      set line ""
    }
    if [catch {close $in} errmsg] {
      showError "reading $mfile: `$errmsg'."
      exit
    }
  }

  # -- initialize make menu
  .$myself.mbar.b1.m1.2 delete 0 last
  .$myself.mbar.b1.m1.2 add command -label "default" -command doMake
  foreach t $mighty_targets {
    .$myself.mbar.b1.m1.2 add command -label $t -command "doMake $t"
  }
}

#
# initialize: initialize application
#
#        dir: next working directory
#       root: TAUROOT of current machine
#       arch: architecture of current machine
#

proc initialize {path} {
  global cosy_var depfile
  global myself

  if { ! [regexp {(.*):(.*)} $path dummy depfile(host) depfile(dir)] } {
    # -- local host
    set depfile(host) "localhost"
    set depfile(dir) $path

    # -- if not remote, set working directory to new application directory
    cd $depfile(dir)
  }

  # -- reset command output display
  if [winfo exists .$myself.txt] {
    if { $cosy_var(reset) } {
      untagErrors
      .$myself.txt delete 1.0 end
    } else {
      .$myself.txt insert end \
                   "===> switched to $depfile(host):$depfile(dir)\n"
      .$myself.txt yview -pickplace end
    }
  }
  set cosy_var(com) ""
  set cosy_var(macros) ""
  if [winfo exists .$myself.mbar.b1.m1.2] scanMakefile
}

#
# atExit: cleanup procedure; makes sure all launched tools are exited
#

proc atExit {fromstop} {
  doStop 0
  if [file exists ".mighty-stamp"] {
    exec rm ".mighty-stamp"
  }
}

# ------------
# -- main code
# ------------
if {$argc == 0} {
  initialize .
} elseif {$argc == 1} {
  initialize [lindex $argv 0]
} else {
  puts stderr "usage: $myself \[\[host:\]directory\]"
  exit
}

# -- create new toplevel window
createWindow
scanMakefile
wm protocol .$myself WM_DELETE_WINDOW atExit
bind all Q exit
removeMessage

# -- launch tau if not already running
if { $depfile(host) == "localhost" } {
  launchTAU {} "$depfile(dir)"
} else {
  launchTAU {} "$depfile(host):$depfile(dir)"
}
