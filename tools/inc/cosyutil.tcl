#*********************************************************************#
#*                TAU/pC++/Sage++  Copyright (C) 1996                *#
#*                        Jerry Manic Saftware                       *#
#*  Indiana University  University of Oregon  University of Rennes   *#
#*********************************************************************#

# Regular expressions for error matching
# These three variables contain regular expressions to match errors,
# warnings, and unknown formats containing a filename and linenumber.
# Each element of the list is a list containing a quoted regular
# expression, the number of the parenthesized subexpression matching the
# filename (numbered beginning with 1), and the number of the parenthesized
# subexpression matching the line number.
# NOTE that backslashes in regular expressions should be doubled for 
# literal semantics and quotes in regular expressions should be preceded by
# a single backslash.

set cosy_var(error_exprs) { \
	{"^([a-z]+: )?([-a-z0-9/_.]+):([0-9]+):" 2 3} \
	{"^([a-z]+: )?\"([-a-z0-9/_.]+)\", line ([0-9]+):" 2 3} \
	{"Error on line ([0-9]+) of ([-a-z0-9/_.]+):" 2 1} \
	{"^([a-z]+: )?\\\"([-a-z0-9/_.]+)\\\", line ([0-9]+): Error:" 2 3} \
        {"^PG.*-[SF]-.*\\(([-a-z0-9/_.]+): ([0-9]+)\\)" 1 2} \
    }
set cosy_var(warning_exprs) { \
	{"^([a-z]+: )?([-a-z0-9/_.]+):([0-9]+): ?warning" 2 3} \
	{"^([a-z]+: )?\"([-a-z0-9/_.]+)\", line ([0-9]+): ?warning" 2 3} \
	{"^([a-z]+: )?\\\"([-a-z0-9/_.]+)\\\", line ([0-9]+): Warning:" 2 3} \
        {"^PG.*-W-.*\\(([-a-z0-9/_.]+): ([0-9]+)\\)" 1 2} \
    }
set cosy_var(fallback_exprs) { \
	{"([a-z/][-a-z0-9/_.]+\\.[a-z]+)[^0-9]*([0-9]+)" 1 2} \
    }

#
# Introduction of the global error structure, for scrolling
# through the error list.
#
    
set errStruct(err_index) 0
set errStruct(beenToZero) 0


#
# getProgfile : returns the profile to which any project file belongs
#
proc getProgfile {file} {
    set allprogfiles [PM_GetFiles]
    if {[lsearch -exact $allprogfiles $file] >= 0} {
	return $file
    }

    foreach pf $allprogfiles {
	set headers [Bdb_GetHeaders $pf]
	if {[lsearch -exact $headers $file] >= 0} {
	    return $pf
	}
    }
    return "-"
}


#
# showSelectedError : shows the error that the user clicked on in the text
#                     window.
#
proc showSelectedError {line file} {
    # if cosy is standalone, don't want to invoke spiffy (and tau and ...)
    if [ALONE] return

    # Launch the editor
    set avail [winfo interps]
    if { [lsearch -exact $avail spiffy ] == -1} {
	launch spiffy .spiffy -waitfor
    }

    # Global select the error line
    set progfile [getProgfile $file]
    PM_GlobalSelect $progfile global_selectLine -1 [list $line $file]
}


# 
# tagErrors - Mark lines in the text widget containing error messages and
#              bind them to the editor callback.
#
proc tagErrors {} {
    global myself cosy_var errStruct

    # set the number of lines
    scan [.$myself.bot.right.txt index end] %d numLines
    
    set err_id 0
    for {set i 1} {$i < $numLines} {incr i} {
	set line_text [.$myself.bot.right.txt get $i.0 "$i.0 lineend"]

	# Check this line against the regular expressions
	if {[matchError $line_text warning err_file err_line]} {
	    highlightError $i $err_id yellow \
		    "showSelectedError $err_line $err_file"
	    set errStruct($err_id,line) $err_line
	    set errStruct($err_id,file) $err_file
	    incr err_id
	} elseif {[matchError $line_text error err_file err_line]} {
	    highlightError $i $err_id red \
		    "showSelectedError $err_line $err_file"
	    set errStruct($err_id,line) $err_line
	    set errStruct($err_id,file) $err_file
	    incr err_id
	} elseif {[matchError $line_text fallback err_file err_line]} {
	    highlightError $i $err_id white \
		    "showSelectedError $err_line $err_file"
	    set errStruct($err_id,line) $err_line
	    set errStruct($err_id,file) $err_file
	    incr err_id
	}
    }
    set cosy_var(num_errs) $err_id
}

# Modified 4/16/96, SAS
#
# errScrollForward - move down the list of generated errors
#
# The variables associated with the procedure are incorporated into
# the global array errStruct, for more standard access.
#
proc errScrollForward {} {
    global myself cosy_var errStruct
    
    if { $cosy_var(num_errs) != 0} {
	if { $errStruct(beenToZero) == 0 } {
	    incr errStruct(beenToZero)
	    set sas_temp $errStruct(err_index)
	    showSelectedError $errStruct($sas_temp,line) \
		    $errStruct($sas_temp,file)
	} else {
	    if {$errStruct(err_index) < [expr $cosy_var(num_errs) - 1] } {
		incr errStruct(err_index)
		set sas_temp $errStruct(err_index)
		showSelectedError $errStruct($sas_temp,line) \
			$errStruct($sas_temp,file)
	    } else {
		bell
#		showError "You have reached the end of the errors."
	    }
	}
    }
}



# Modified 4/16/96, SAS
#
# errScrollBackward - move down the list of generated errors
#
# The variables associated with the procedure are incorporated into
# the global array errStruct, for more standard access.


proc errScrollBackward {} {
    global myself cosy_var errStruct
    
    if { $cosy_var(num_errs) != 0} {
	if { $errStruct(beenToZero) == 0 } {
	    incr errStruct(beenToZero)
	    set sas_temp $errStruct(err_index)
	    showSelectedError $errStruct($sas_temp,line) \
		    $errStruct($sas_temp,file)
	} else {
	    if { $errStruct(err_index) > 0 } {
		incr errStruct(err_index) -1
		set sas_temp $errStruct(err_index)
		showSelectedError $errStruct($sas_temp,line) \
			$errStruct($sas_temp,file)
	    } else {
		bell
#		showError "You have reached the beginning of the errors."
	    }
	}
    }
}
    

#
# matchError - regular expression matching against the error expresions.
#
#    lineStr - compiler output line
#       what - match type: must be "error", "warning", or "fallback"
#    fileVar - name of variable to return filename in
#    lineVar - name of variable to return line number in
#
proc matchError {lineStr what fileVar lineVar} {
    upvar $fileVar file
    upvar $lineVar line
    global cosy_var myself

    if {$what != "error" && $what != "warning" && $what != "fallback"} {
	puts "Fatal error in $myself:matchError: bad 'what' value."
	exit
    }

    foreach expr $cosy_var(${what}_exprs) {
	if {[regexp -nocase [lindex $expr 0] $lineStr \
		mvar(1) mvar(2) mvar(3) mvar(4) mvar(5) mvar(6)]} {
	    set file $mvar([expr 1 + [lindex $expr 1]])
	    set line $mvar([expr 1 + [lindex $expr 2]])
	    return 1
	}
    }
    return 0
}

#
# highlightError - highlight an error message in the text
#
#        lineNum - the number of the line in the text widget
#             id - the id number of the error
#          color - color to highlight
#        binding - binding for the tag
#
proc highlightError {lineNum id color binding} {
    global myself errStruct

    set tagname [format "err%d" $id]
    .$myself.bot.right.txt tag add $tagname $lineNum.0 $lineNum.end
    if {$color != "white"} {
	.$myself.bot.right.txt tag configure $tagname -background $color \
		-borderwidth 2 -relief raised \
		-font -Adobe-Helvetica-Medium-R-Normal--*-140-*
    }
    .$myself.bot.right.txt tag bind $tagname <ButtonPress-1> $binding
    .$myself.bot.right.txt tag bind $tagname <ButtonRelease-1> \
	"set errStruct(err_index) $id"
}

#
# untagErrors - remove the error tags from the text widget
#
proc untagErrors {} {
    global myself cosy_var errStruct

    for {set i 0} {$i < $cosy_var(num_errs)} {incr i} {
	set tagname [format "err%d" $i]
	.$myself.bot.right.txt tag delete $tagname
	unset errStruct($i,file)
	unset errStruct($i,line)
    }
    set cosy_var(num_errs) 0
    set errStruct(err_index) 0
    set errStruct(beenToZero) 0
}

#
# readLine: readline from command execution pipe
#           assumes it is called from as fileevent handler
#
proc readLine {} {
  global cosy_var

  if [eof $cosy_var(infile)] {
    doStop 0
  } else {
    gets $cosy_var(infile) line
    insertLine $line
  }
}

#
#   doStop: cleanup command execution pipe
#
# fromstop: should be 1 if called as "interrupt/kill" command
#
proc doStop {{fromstop 0}} {
  global cosy_var

  if [info exists cosy_var(infile)] {
    if { $fromstop } {
      set cosy_var(err) 1
      insertLine "===> STOPPING ..."
    }
    catch {close $cosy_var(infile)}
    unset cosy_var(infile)
  }
  set cosy_var(ready) 1
}

#
# insertLine: insert one line of text in command output display area
#
#       line: text to insert
#

proc insertLine {line} {
  global myself
  global cosy_var

  # -- test for errors in putput
  if { [regexp -nocase error $line] || [regexp "up to date" $line] } {
    set cosy_var(err) 1
  }

  # -- suppress warnings, if specified
  if { $cosy_var(warn) || [regexp -nocase warning $line] == 0 } {
    .$myself.bot.right.txt insert end "$line\n"
    .$myself.bot.right.txt yview -pickplace end
    update
  }
}

#
# doExec: execute (local or remote) command
#
#   what: command to execute
#

proc doExec {what} {
  global myself
  global cosy_var depfile
  global BINDIR REMSH

  # -- remember command without pathname
  set cmd [lreplace $what 0 0 [file tail [lindex $what 0]]]

  # -- init variables and reset command output display
  set cosy_var(err) 0
  set cosy_var(com) "[string range $cmd 0 80] ..."
  if { $cosy_var(reset) } {
    untagErrors
    .$myself.bot.right.txt delete 1.0 end
  } else {
    .$myself.bot.right.txt insert end "===> $cmd\n"
    .$myself.bot.right.txt yview -pickplace end
  }
  update

  set cosy_var(ready) 0

  # -- open pipe to requested command to execute in a (remote) shell
  # -- then set input fileevent handler and wait for completion

  if { $depfile(host) != "localhost" } {
    if [catch {open "|$REMSH $depfile(host) -n \"cd $depfile(dir) ; $what\""} cosy_var(infile)] {
      insertLine $cosy_var(infile)
    } else {
      fileevent $cosy_var(infile) readable readLine
      tkwait variable cosy_var(ready)
    }
  } else {
    if [catch {open "|sh -c \"$what 2>&1\""} cosy_var(infile)] {
      insertLine $cosy_var(infile)
    } else {
      fileevent $cosy_var(infile) readable readLine
      tkwait variable cosy_var(ready)
    }
  }

  # -- cleanup
  if [info exists cosy_var(infile)] { unset cosy_var(infile) }
  set cosy_var(com) "$cosy_var(com) done"
  update
  return $cosy_var(err)
}

proc buildRootFilesTable {} {
    global cosy_rootfiles;
    
    foreach progfile [PM_GetFiles] {
	set cosy_rootfiles([file rootname $progfile]) $progfile
    }
}

proc target2progfile {target} {
    global cosy_rootfiles

    set tmproot [file rootname [string trim $target]]
    switch -glob $tmproot {
	*-temp        {set rootname [string range $tmproot 0 \
		         [expr [string last "-temp" $tmproot] - 1]]}
        default       {set rootname $tmproot}
    }

    if {[array names cosy_rootfiles $rootname] != {}} {
	return $cosy_rootfiles($rootname)
    } else {
	return T2P_FAILED
    }
}
    

