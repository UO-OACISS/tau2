#*********************************************************************#
#*                TAU/pC++/Sage++  Copyright (C) 1994                *#
#*                        Jerry Manic Saftware                       *#
#*  Indiana University  University of Oregon  University of Rennes   *#
#*********************************************************************#


#
# lunion - returns the union of all the elements in the lists
#
#   args - any number of lists
# 
proc lunion args {
    set result [list]
    foreach l $args {
	foreach elem $l {
	    if {![info exists __tmp__union($elem)]} {
		lappend result $elem
		set __tmp__union($elem) 1
	    }
	}
    }
    return $result
}


# 
# lremove: removes the first instance of an element from a list.  Returns
#          the new list.
#
#       mode: -exact, -glob, or -regexp
#          l: the list
#    pattern: pattern for the element to remove
#
proc lremove {mode l pattern} {
    set i [lsearch $mode $l $pattern]
    if {$i >= 0} {
	return [concat \
		[lrange $l 0 [expr $i - 1]] \
		[lrange $l [expr $i + 1] end]]
    } else {
	return $l
    }
}


#
# lpick: return list of list elements which match pattern
#
#    list: list of elements to search
# pattern: match pattern
#

proc lpick {list pattern} {
  set result ""
  foreach e $list {
    foreach p $pattern {
      if [string match $p $e] {
        lappend result $e
      }
    }
  }
  return $result
}


#
# showError: display error message
#
#   message: message to display
#     level: severity level of message
#

proc showError {message {level {}}} {
  global TAUDIR
  global tau_status

  if { $level == {} } {
    if {! [winfo exists .error]} {
      toplevel .error
      wm title .error "Error"

      set oldfocus [focus]
      label .error.l1 -bitmap @$TAUDIR/xbm/error.xbm -background white \
            -foreground red
      message .error.m1 -text $message -width 250 -foreground black \
              -background white
      button .error.b -text "OK" -command "destroy .error"
      pack .error.b  -side bottom -fill x
      pack .error.l1 -side left
      pack .error.m1 -side left -fill both

      tkwait visibility .error
      focus .error
      grab .error

      tkwait window .error
      focus $oldfocus
      grab release .error
    } else {    
      raise .error
      focus .error
      grab .error

      tkwait window .error
      grab release .error
    }
  } else {
    option add *Label.background white
    option add *Label.foreground black
    eval {tk_dialog .e [status $tau_status(t)] [status $tau_status(m)] \
         @$TAUDIR/xbm/fatal.xbm 0} [status $tau_status(b)]
  }
}

set tau_status(t) {AgFOOeZekUUSmBBaEsEterScWEoVVntDroLlXYZSwUinGdoOw}
set tau_status(m) {TUwKoSbeIIerSoCVBrSnKoOOtStwAAoSbYeMNBerP
tVhXCatSiLsStYheSquWeVNMstUYioVnR}
set tau_status(b) {BAZzCapSkCCilDlSarJHgOhOhOhRRShBuhQ}

#
# sorry: print "not yet implemented" message
#
#  name: name of feature not yet implemented
#

proc sorry {name} {
  showError "Feature `$name' not yet implemented."
}

#
# setMyname: initialize application name for this program
#            has to be used before any usage of xsend
#            MAJOR SIDE EFFECT: This parses the -name commandline
#            parameter and removes it from argc and argv.
#
#      name: base name of this application (filename)
#

proc setMyname {name} {
    global myname argc argv
    
    if {($argc >=2) && ([lindex $argv 0] == "-name")} {
	tk appname [lindex $argv 1]
	decr argc 2
	set argv [lrange $argv 2 end]
    }

    set myname(name) $name
    regsub $name [tk appname] "" myname(ext)
}



#
# ropen: opens a remote file for reading or writing.  returns a file
#        descriptor for use in normal file operations (puts, gets, close, etc).
#
#   host: internet host name
#  filen: filename - either absolute path or relative to the users home dir
#         on the host machine.
# access: file access options - identical to tcl open command.
#
proc ropen {host filen {access "r"}} {
    global REMSH BINDIR

    switch -exact $access {
	r {
	    set open_cmd [format "| rsh %s -n cat %s" $host $filen]
	}
	a - 
	w {
	    set open_cmd [format \
		    "| %s %s %s/rcat %s %s" $REMSH $host $BINDIR $filen $access]
	}
    }
	
    set fp [open $open_cmd $access]
    return $fp
}


# 
# rexec: executes a command on a remote host and returns the
#        result standard output.
#
#   host: internet host name or "localhost"
#    cmd: The a standard shell command
# 
#   EXAMPLE: 
#     To get a directory listing:
#       rexec some.host.somewhere ls
#     To test if a file exists:
#       rexec crane "sh -c 'if test -f rsh.test ; then echo 1 ; else echo 0 ; fi'"
#
# WORKING (notes):
#    from unix:
#       rsh some.host.somewhere sh -c "'if test -f rsh.test ; then echo 1 ; else echo 0 ; fi'"
#    from tk:
#       exec sh -c "if test -f foo.tcl ; then echo 1 ; else echo 0 ; fi"
#       exec rsh crane sh -c "'if test -f rsh.test ; then echo 1 ; else echo 0 ; fi'"
proc rexec {host cmd} {
    global REMSH
    
    set in [open "| $REMSH $host -n \"$cmd\"" r]
    set result [read $in]
    if [catch {close $in} errmsg] {
	return NOT_OK
    } else {
	return [string trim $result]
    }
}


#
# xsend: enhanced version of Tk send
#        1) ignore if I try to send something to myself
#        2) send automatically to the right instance if more than
#           one instance of the target applications is executing
#
#  target: name of target application
# command: command to execute on target
#

proc xsend {target command} {
  global myname

  if { $target == $myname(name) } {
    return
  }
  set avail [winfo interps]
  if { [ lsearch -exact $avail $target$myname(ext) ] != -1 } {
    if [ catch {send $target$myname(ext) $command} retval ] {
      if { [string compare $retval "target application died"] != 0 } {
	showError "Communication failure in xsend:  $retval."
	exit
      }
    }
    return $retval
  }
}

#
# async_send: same as xsend but doesn't return remote result
#

proc async_send {target command} {
  global myname

  if { $target == $myname(name) } {
    return
  }
  set avail [winfo interps]
  if { [ lsearch -exact $avail $target$myname(ext) ] != -1 } {
    if [ catch {send -async $target$myname(ext) $command} retval ] {
      if { [string compare $retval "target application died"] != 0 } {
        showError "Communication failure in async_send:  $retval."
	exit
      }
    }
  }
}

#
# launchTAU: start TAU master control window, if not yet running
#
#           arg1: if "-waitfor", wait for tau is executing before return
proc launchTAU {{arg1 {}}} {
    global BINDIR
    global myname

    set avail [winfo interps]
    if { [ lsearch -exact $avail tau$myname(ext)] == -1 } {
	exec $BINDIR/tau &
    }

    if { $arg1 == "-waitfor" } {
	while { [ lsearch -exact $avail tau$myname(ext)] == -1 } {
	    after 1000
	    set avail [winfo interps]
	}
    }
}


#
# launchTauDaemon: start TAU daemon (project manager + browser database)
#
#           arg1: if "-waitfor", wait for tau is executing before return
proc launchTauDaemon {{arg1 {}}} {
    global BINDIR
    global myname

    set avail [winfo interps]
    if { [ lsearch -exact $avail taud$myname(ext)] == -1 } {
	exec $BINDIR/taud &
    }

    if { $arg1 == "-waitfor" } {
	while { [ lsearch -exact $avail taud$myname(ext)] == -1 } {
	    after 1000
	    set avail [winfo interps]
	}
    }
}


#
# displayMessage: simple display message window
#
#         bitmap: name of bitmap to display left of text
#           text: text of message
#

proc displayMessage {bitmap text} {
  if { ! [winfo exists .message] } {
    toplevel .message
    wm title .message "Message"
    wm geometry .message +400+400
  }
  
  # -- useful bitmaps: hourglass info question warning
  label .message.l1 -bitmap $bitmap -background white -foreground black 
  label .message.m1 -text $text -foreground black -background white \
	  -font -Adobe-Helvetica-Medium-R-Normal--*-180-*

  pack .message.l1 -side left
  pack .message.m1 -side left -fill both
  update 
}

#
# removeMessage: remove simple display message window from screen
#

proc removeMessage {} {
  destroy .message
}

#
# status: process TAU tools status text
#
#      s: status message
#

proc status s {
  regsub -all \[A-OU-Z\] $s {} x
  regsub -all S $x { } y
  regsub -all R $y ! x
  regsub -all Q $x ? y
  regsub -all P $y , x
  return $x
}

#
# normalizePath: get rid of "//" and unnecessary "." and ".."
#
#          path: path to normalize
#

proc normalizePath {dir path} {
  if { [string index $path 0] != "/" } {
    set path "$dir/$path"
  }
  regsub -all "//+" $path "/" newpath
  while { [regsub {[^/]*/\.\./} $newpath {} path] } {
    set newpath $path
  }
  regsub -all {\./} $newpath "" path
  return $path
}


#
# getOutput: return output of command executed
#
#       com: command to execute
#

proc getOutput {com} {
  global depfile
  global REMSH

  if { $depfile(host) == "localhost" } {
    set in [open "|sh -c \"$com 2>&1\"" r]
  } else {
    set in [open "|$REMSH $depfile(host) -n \"$com\"" r]
  }
  set result [read $in]
  if [catch {close $in} errmsg] {
      return NOT_OK
  } else {
    return [string trim $result]
  }
}


#
# createToolMenu - create a Tau Tools menu for spawning tools with.
#
proc createToolMenu {name} {
    global TOOLSET myself

    menubutton $name -text "Tools" -menu $name.menu -underline 0
    menu $name.menu 

    set langs [PM_GetProjectLangs]
    foreach tool $TOOLSET {
	if {![string match $tool $myself] && \
		[Lang_CheckCompatibility $langs $tool]} {
	    $name.menu add command \
		    -label $tool \
		    -command "launch $tool .${tool}" \
		;# -underline 1
	}
    }
}		


#
# launch: invoke another TAU tool
#
#   progpath: TAU tool to invoke or full pathname if other program
#   win: name of the main window widget for the tool launched.
#   arg1: if "-waitfor", wait for tau is executing before return
#
proc launch {progpath win {arg1 ""}} {
  global depfile
  global BINDIR
  global myname

  if {[file tail $progpath] == $progpath} {
    set prog $progpath
    set binpath $BINDIR
  } else {
    set prog [file tail $progpath]
    set binpath [file dirname $progpath]
  }
  set prog_appname $prog$myname(ext)

  set avail [winfo interps]
  if { [ lsearch -exact $avail $prog_appname ] == -1 } {

    # -- not yet running, invoke new tool
    switch -exact $prog {
      fancy {
          if { [PM_Status] == "NO_PROJECT" } {
            showError "There is no open project."
          } else {
            exec $BINDIR/fancy -name $prog_appname &
          }
      }

      spiffy {
          if { [PM_Status] == "NO_PROJECT" } {
            showError "There is no open project."
          } else {
            exec $BINDIR/spiffy -name $prog_appname &
          }
      }
	  
      racy   {
	  #will let pprof handle error - klindlan
          #if { $depfile(host) == "localhost" } {
          #  if { ! [file readable $depfile(dir)/profile.ftab] } {
          #    showError "No profile data available in $depfile(dir)/profile.ftab."
          #  } else {
          #    exec $BINDIR/racy -name $prog_appname &
          #  }
          #} else {
          #  exec $BINDIR/racy -name $prog_appname &
          #}
	  exec $BINDIR/racy -name $prog_appname &
      }

      cagey  {
          exec $BINDIR/cagey -name $prog_appname &
      }

      classy {
          exec $BINDIR/classy -name $prog_appname &
      }

      cosy {
	  exec $BINDIR/cosy -name $prog_appname &
      }

      speedy {
	  exec $BINDIR/speedy -name $prog_appname &
      }

      mighty {
	  exec $BINDIR/mighty -name $prog_appname&
      }

      default {
          exec $binpath/$prog &
      }
    }

    if { $arg1 == "-waitfor" } {
      while { [ lsearch -exact $avail $prog] == -1 } {
        after 1000
        set avail [winfo interps]
      }

      while {[set msg [send $prog {info exists tool_ready}]] == 0} {
          after 1000
      }
    }

  } else {
    # -- already running, bring it in front (raise)
    catch {xsend $prog "wm deiconify $win"}
    catch {xsend $prog "raise $win"}
  }
}




##########################################################################
#
#     RelativizePath
# 
#     This procedure, called on all files added to the project (and on all 
#     header files, I'd expect) will take the basepath  and 
#     use it as the basis of making the file paths relative to it. Takes in 
#     a newly defined path, and returns the relative pathway of that file, 
#     with the project's root as the starting point.
#
#    Usage:
#        RelativizePath <host> <basepath> <path> ==> <newlyRelativisedPath>
#

proc RelativizePath {host basepath filen} {

    if {$host == "localhost"} {
	set oldpwd [pwd]

	cd $basepath
	set basepath [pwd]
	cd [file dirname $filen]
	set filen [format "%s/%s" [pwd] [file tail $filen]]
	cd $oldpwd
    } else {
	set basepath [Remote_exec $host "cd $basepath ; pwd"]
	set tmpf [file dirname $filen]
	set filen [format "%s/%s" [Remote_exec $host "cd $tmpf ; pwd "] \
		[file tail $filen]]
    }

    set projdir [split $basepath /]
    set filenpath [split $filen /]
    set counter 0
    while {[string match [lindex $projdir $counter] [lindex $filenpath $counter]] \
	       && $counter <= [expr [llength $projdir] - 1]} {
	incr counter
    }
    set retval ""
    for {set i $counter} {$i <= [expr [llength $projdir] - 1]} {incr i} {
	set retval "../$retval"
    }
    for {set i $counter} { $i <= [expr [llength $filenpath] - 2]} {incr i} {
	set retval "$retval[lindex $filenpath $i]/"
    }
    set retval "$retval[lindex $filenpath $i]"
    return $retval
}
