#*********************************************************************#
#*                TAU/pC++/Sage++  Copyright (C) 1994                *#
#*                        Jerry Manic Saftware                       *#
#*  Indiana University  University of Oregon  University of Rennes   *#
#*********************************************************************#
displayMessage hourglass "Loading $myself..."

source "$TAUDIR/inc/depfile.tcl"
source "$TAUDIR/inc/stack.tcl"
source "$TAUDIR/inc/fileio.tcl"

set showFile(name) "";     # filename of currently selected function/class
set showFile(progfile) ""; # Sage++ id of currently selected function/class
set showFile(tag)  NONE;   # Sage++ id of currently selected function/class
set showFile(what) NONE;   # type of selected item
set allclasses "";         # list of all defined classes

set fancy_funclist      [list];     # list of functions in the func list
set fancy_methodlist    [list];     # list of methods in the method list

set editor(view_with) 0;   # Boolean: view files with the editor?
set editor(interp_name) spiffy$myname(ext);   # Name of the editor

initStack viewerStack

# fancy_Altn_global_binding and fancy_Altp_global_binding
# 
# A global fancy keybinding, to allow the user to scroll through the
# error messages in the editor window created by Fancy for examining
# the source code.
# SAS 4/9/96
#

proc fancy_Altn_global_binding {} {
    set avail [winfo interps]
    if { [ lsearch -exact $avail cosy] == -1 } {
	EditorMsg "Cosy isn't running; no errors to parse."
	after 1000 EditorMsg
    } else {
	async_send cosy errScrollForward
    }
}


proc fancy_Altp_global_binding {} {
    set avail [winfo interps]
    if { [ lsearch -exact $avail cosy] == -1 } {
	EditorMsg "Cosy isn't running; no errors to parse."
	after 1000 EditorMsg
    } else {
	async_send cosy errScrollBackward
    }
}


#
# exit - TAU exit function that communicates the event to other tau tools.
#
rename exit exit.old
proc exit {{status 0}} {
    global myself

    PM_RemGlobalSelect $myself \
	    { global_selectLine global_selectFuncTag global_selectClassTag }
    PM_RemTool $myself

    exit.old
}




#
# selectLine: implementation of global feature "selectLine" for fancy
#             highlight specified line in file
#
#       line: line number to select
#       file: file name to select
#        tag: tag of call site to select or -1
#

proc selectLine {progfile tag line file} {
    if {$progfile != "-"} {
	loadDep $progfile
    }
    showFile $progfile $tag Line $file $line
}

#
# selectFuncTag: implementation of global feature "selectFunction" for fancy
#                display body of selected function in text window
#
#           tag: id of function to select
#

proc selectFuncTag {progfile tag {back 0}} {
    loadDep $progfile
    if {[Cgm_FuncInfo $progfile $tag name] != "CGM_DOESNT_EXIST"} {
	showFile $progfile $tag Func "" - $back
	processTag $progfile $tag
    }
}

#
# showLine: highlight specified line in file
#
#      win: pathname of text window
#      tag: Sage++ id of call site
#     file: filename where source code of function can be found
#     line: linenumber where source code of function can be found
#

proc showLine {win $progfile tag file line} {

    $win tag add line $line.0 [expr $line+1].0
    if { [winfo depth .] == 2 } {
	$win tag configure line -background black -foreground white
    } else {
	$win tag configure line -background yellow
    }
    $win yview -pickplace $line.0

    if { $tag != -1 } {
	set name [Cgm_FuncInfo $progfile $tag name]

	set text [$win get $line.0 [expr $line+1].0]
	if [regexp -indices $name $text range] {
	    set cstart $line.[lindex $range 0]
	    set cend   $line.[expr [lindex $range 1]+1]

	    if [ lindex [Cgm_FuncInfo $progfile $tag file] 0 ] {
		# -- child with body; add bindings to select child
		# -- dchild has display attributes, bchild has bindings
		$win tag add dchild1 $cstart $cend
		$win tag add bchild1 $cstart $cend
		if { [winfo depth .] == 2 } {
		    $win tag configure dchild1 -background white \
			    -foreground black
		} else {
		    $win tag configure dchild1 -background lightblue
		}
		$win tag bind bchild1 <Button-1> \
			"PM_GlobalSelect $progfile global_selectFuncTag $tag; break"
		$win tag bind bchild1 <Button-2> \
			"PM_GlobalSelect $progfile global_showFuncTag $tag; break"
		$win tag bind bchild1 <Button-3> \
			"PM_GlobalSelect $progfile global_selectFuncTag $tag; break"
		$win tag lower bchild1

	    } else {
		# -- library function (no body available); 
		# -- don't set up bindings
		$win tag add dchild1 $cstart $cend
		if { [winfo depth .] == 2 } {
		    $win tag configure dchild1 -background white \
			    -foreground black
		} else {
		    $win tag configure dchild1 -background orchid
		}
	    }
	}
    }
}


# 
# findLocalRef: reduces the result of a BDB query to ONLY the first match
#               originating from the same progfile.
#
#     progfile: the progfile to match with
#   query_rslt: the raw result of a Bdb_Query call
#
proc findLocalRef {progfile query_rslt} {
    set len [llength $query_rslt]

    if {$len == 0} {
	return {}
    } elseif {$len == 1} {
	return [lindex $query_rslt 0]
    } else {
	foreach elem $query_rslt {
	    if {[lindex $elem 0] == $progfile} {
		return $elem
	    }
	}
    }
}

    

#
# showFunc: highlight header, body, and children of selected function
#           in text window
#
#      win: pathname of text window
#      tag: Sage++ id of selected function
#     file: filename where source code of function can be found
#     line: linenumber where source code of function can be found
#

proc showFunc {win progfile tag file line} {
    if { [scan [Cgm_FuncInfo $progfile $tag pos] \
	    "%s %s %s %s" hstart hend bstart bend] == 4 && \
	    $hstart != "-" } {
	# -- highlight header and body
	# -- dfunc has display attributes, bfunc has bindings
	$win tag add dfunc $hstart $hend
	$win tag add bfunc $hstart $hend
	$win tag add body $bstart $bend
	if { [winfo depth .] == 2 } {
	    $win tag configure dfunc -background black -foreground white
	    $win tag configure body -background black -foreground white
	} else {
	    $win tag configure dfunc -background orange
	    $win tag configure body -background yellow
	}
	$win yview -pickplace $line.0

	# -- if it is a member function, add bindings to
	# -- select its class
	set class [lindex [Cgm_FuncInfo $progfile $tag class] 0]
	if { $class != "-" } {
	    set ctag [Cgm_ClassInfo $progfile $class tag]
	    $win tag bind bfunc <Button-1> "selectClassTag $progfile $ctag; break"
	    $win tag bind bfunc <Button-3> \
		    "PM_GlobalSelect $progfile global_selectClassTag $ctag; break"
	}
	$win tag lower bfunc

	# -- highlight children (function calls within the body)
	# -- dchild has display attributes, bchild has bindings
	if {[Cgm_FuncInfo $progfile $tag childpos] != "CGM_DOESNT_EXIST"} {
	    set calls [Cgm_FuncInfo $progfile $tag calls]
	    set pos [Cgm_FuncInfo $progfile $tag childpos]
	    set i 0
	    foreach c $pos {
		scan $c "%s %s" cstart cend
		if { $cstart != "-" } {
		    set ctag [lindex $calls $i]
		    set query_rslt [Bdb_FindFuncDefByName \
			    [Cgm_FuncInfo $progfile $ctag name] \
			    [Cgm_FuncInfo $progfile $ctag mname]]
		    if {$query_rslt != {}} {
			set linkto_progfile [lindex $query_rslt 0]
			set linkto_tag [lindex $query_rslt 1]
		    } else {
			set linkto_progfile "-"
			set linkto_tag "-"
		    }

		    if {$linkto_progfile != "-"} {
			# -- children with body; add bindings to select child
			$win tag add dchild$i $cstart $cend
			$win tag add bchild$i $cstart $cend
			if { [winfo depth .] == 2 } {
			    $win tag configure dchild$i -background white \
				    -foreground black
			} else {
			    $win tag configure dchild$i -background lightblue
			}
			$win tag bind bchild$i <Button-1> \
				"PM_GlobalSelect $linkto_progfile global_selectFuncTag $linkto_tag; break"
			$win tag bind bchild$i <Button-2> \
				"PM_GlobalSelect $linkto_progfile global_showFuncTag $linkto_tag; break"
			$win tag bind bchild$i <Button-3> \
				"PM_GlobalSelect $linkto_progfile global_selectFuncTag $linkto_tag; break"
			$win tag lower bchild$i
		    } else {
			# -- library function (no body available); 
			# -- don't set up bindings
			$win tag add dchild$i $cstart $cend
			if { [winfo depth .] == 2 } {
			    $win tag configure dchild$i -background white \
				    -foreground black
			} else {
			    $win tag configure dchild$i -background orchid
			}
		    }
		}
		incr i
	    }
	}
    }
}

#
# showClass: highlight header, body, and members of selected class
#            in text window
#
#      win: pathname of text window
#      tag: Sage++ id of selected class
#     file: filename where source code of class definition can be found
#     line: linenumber where source code of class definition can be found
#

proc showClass {win progfile tag file line} {

    if { [scan [Cgm_ClassInfo $progfile $tag pos] \
	    "%s %s %s %s" hstart hend bstart bend] == 4 } {

	# -- highlight header and body
	$win tag add dfunc $hstart $hend
	$win tag add body $bstart $bend
	if { [winfo depth .] == 2 } {
	    $win tag configure dfunc -background black -foreground white
	    $win tag configure body -background black -foreground white
	} else {
	    $win tag configure dfunc -background orange
	    $win tag configure body -background yellow
	}
	$win yview -pickplace $line.0

	# -- highlight attribute markers like public, private, ...
	set pos [Cgm_ClassInfo $progfile $tag typepos]
	foreach c $pos {
	    scan $c "%s %s" cstart cend
	    if { $cstart != "-" } {
		$win tag add ctype $cstart $cend
	    }
	}
	if { [winfo depth .] != 2 } {
	    $win tag configure ctype -background orange
	}

	# -- highlight function members
	# -- dchild has display attributes, bchild has bindings
	set i 0
	set pos [Cgm_ClassInfo $progfile $tag fmempos]
	foreach c $pos {
	    scan $c "%d %s %s" ctag cstart cend
	    if { $cstart != "-" } {
		if [ lindex [Cgm_FuncInfo $progfile $ctag file] 0 ] {
		    $win tag add dchild$i $cstart $cend
		    $win tag add bchild$i $cstart $cend
		    if { [winfo depth .] == 2 } {
			$win tag configure dchild$i -background white \
				-foreground black
		    } else {
			$win tag configure dchild$i -background lightblue
		    }
		    $win tag bind bchild$i <Button-1> \
			    "PM_GlobalSelect $progfile global_selectFuncTag $ctag; break"
		    $win tag bind bchild$i <Button-2> \
			    "PM_GlobalSelect $progfile global_showFuncTag $ctag; break"
		    $win tag bind bchild$i <Button-3> \
			    "PM_GlobalSelect $progfile global_selectFuncTag $ctag; break"
		    $win tag lower bchild$i
		} else {
		    $win tag add dchild$i $cstart $cend
		    if { [winfo depth .] == 2 } {
			$win tag configure dchild$i -background white \
				-foreground black
		    } else {
			$win tag configure dchild$i -background orchid
		    }
		}
	    }
	    incr i
	}
	
	# -- add bindings to select base classes if necessary
	# -- dbase has display attributes, bbase has bindings
	set i 0
	set pos [Cgm_ClassInfo $progfile $tag basepos]
	foreach c $pos {
	    scan $c "%d %s %s" ctag cstart cend
	    if { $cstart != "-" } {
		$win tag add dbase$i $cstart $cend
		$win tag add bbase$i $cstart $cend
		$win tag bind bbase$i <Button-1> \
			"selectClassTag $progfile $ctag; break"
		$win tag bind bbase$i <Button-3> \
			"PM_GlobalSelect $progfile global_selectClassTag $ctag; break"
		$win tag lower bbase$i
	    }
	    incr i
	}
    }
}

#
# markAllFuncAndClass: setup bindings for selecting all functions or classes
#                      in a file
#
#                 win: pathname of text window
#                file: filename to look for
#
proc markAllFuncAndClass {win file progfile} {
  global depfile

  # -- look through all functions whether they are declared in the
  # -- specified file and setup bindings, if found one
  foreach t [Cgm_DepInfo $progfile allftags] {
    set f [lindex [Cgm_FuncInfo $progfile $t file] 1]
    if { $f != "-" && [string compare $f $file] == 0 } {
      if { [scan [Cgm_FuncInfo $progfile $t pos] \
	      "%s %s %s %s" hstart hend bstart bend] == 4 } {
        if { $hstart != "-" } {
          $win tag add func-$progfile-$t $hstart $bend
          if { [winfo depth .] != 2 } {
            $win tag configure func-$progfile-$t -background Gray95
          }
          $win tag bind func-$progfile-$t <Button-1> \
		  "PM_GlobalSelect $progfile global_selectFuncTag $t"
          $win tag bind func-$progfile-$t <Button-2> \
		  "PM_GlobalSelect $progfile global_showFuncTag $t"
          $win tag bind func-$progfile-$t <Button-3> \
		  "PM_GlobalSelect $progfile global_selectFuncTag $t"
        }
      }
    }
  }

  # -- do the same thing for class definitions
  foreach t [Cgm_DepInfo $progfile allctags] {
    set f [lindex [Cgm_ClassInfo $progfile $t file] 1]
    if { $f != "-" && [string compare $f $file] == 0 } {
      if {[scan [Cgm_ClassInfo $progfile $t pos] \
	      "%s %s %s %s" hstart hend bstart bend] == 4} {
        if { $hstart != "-" } {
          $win tag add class-$progfile-$t $hstart $bend
          if { [winfo depth .] != 2 } {
            $win tag configure class-$progfile-$t -background Gray95
          }
          $win tag bind class-$progfile-$t <Button-1> \
		  "selectClassTag $progfile $t"
          $win tag bind class-$progfile-$t <Button-3> \
		  "PM_GlobalSelect $progfile global_selectClassTag $t"
        }
      }
    }
  }
}


#
# readFile: read text of file into text window
#
#      win: pathname of text window
#     file: UNIX pathname of file to read
#

proc readFile {win file} {
  global depfile REMSH

  if { $depfile(host) == "localhost" } {
    set in [open $file r]
  } else {
    set in [open "|$REMSH $depfile(host) -n \"cd $depfile(dir); cat $file\"" r]
  }
  $win insert end [read $in]
  if [catch {close $in} errmsg] {
    showError "$file: `$errmsg'."
  }
  $win configure -state disabled
}

#
# backTag: switch back to last selcted class or function
#

proc backTag {} {
    global viewerStack

    # -- if there is something in the stack switch back to it
    set b [popStack viewerStack]
    if { $b != "" } {
	set progfile [lindex $b 0]
	set tag [lindex $b 1]
	if {[Cgm_FuncInfo $progfile $tag name] == "CGM_DOESNT_EXIST"} {
	    selectClassTag $progfile $tag 1 1
	} else {
	    selectFuncTag $progfile $tag 1
	}
    }

    # -- disable back button if stack now empty
    if { $viewerStack(length) == 0 } {
	.viewer.back configure -state disabled
    }
}

#
# showFile: show body of function or class in text window
#
# progfile: program file for selecting depfile
#      tag: Sage++ id of selected item
#     what: type of selected item: Func, Class, or Line
#

proc showFile {progfile tag what {fn ""} {ln -} {back 0}} {
    global myself
    global depfile
    global showFile
    global TAUDIR
    global viewerStack
    global editor

    # -- get necessary info from global database
    switch -exact $what {
	Func  {
            set file [lindex [Cgm_FuncInfo $progfile $tag file] 1]
            set line [lindex [Cgm_FuncInfo $progfile $tag file] 0]
            set name [Cgm_FuncInfo $progfile $tag name]
	}
	Class {
            set file [lindex [Cgm_ClassInfo $progfile $tag file] 1]
            set line [lindex [Cgm_ClassInfo $progfile $tag file] 0]
            set name [Cgm_ClassInfo $progfile $tag name]
	}
	Line  {
            set file $fn
            set line $ln
	}
	default {
	    puts stderr "$myself: internal error: invalid what $what"
	    exit
	}
    }
    
    if { $file == "-" } {
	return  ;# For some reason, the error dialog freezes fancy....
	showError "File information for function or class `$name' not available: function/class may be defined in binary library or bound dynamically."
	return
    } else {
	if { ![FileIO_file_readable $depfile(host) $file] } {
	    showError "Cannot open file: `$file'."
	    return
	}
	
	# If editor option on, don't open viewer.  The editor will respond to 
	# the global select function
	if {$editor(view_with) && \
		([lsearch -exact [winfo interps] $editor(interp_name)] != -1)} {
	    return
	}
	
	if { ! [winfo exists .viewer] } {
	    # -- text window does not yet exist, create and display one
	    toplevel .viewer
	    wm title .viewer "$depfile(host):[normalizePath $depfile(dir) $file]"
	    wm iconbitmap .viewer @$TAUDIR/xbm/$myself.xbm
	    wm minsize .viewer 50 50
	    
	    #
	    # Another binding, for scrolling through errors. Necessary redundancy,
	    # since the viewer and fancy have separate toplevel windows.
	    # 4/15 SAS
	    #
	    bind .viewer <Alt-n> fancy_Altn_global_binding
	    bind .viewer <Alt-p> fancy_Altp_global_binding
	    
	    
	    text .viewer.t1 -width 80 -height 32 -background white -foreground black
	    scrollbar .viewer.s1 -orient vert -relief sunken \
		    -command ".viewer.t1 yview"
	    .viewer.t1 configure -yscrollcommand ".viewer.s1 set"
	    
	    frame .viewer.bo -relief raised -borderwidth 2
	    button .viewer.back -text "back" -command "backTag"
	    button .viewer.close -text "close" -command "destroy .viewer"
	    pack .viewer.back -side left -padx 20 -pady 5 -ipadx 5 -in .viewer.bo
	    pack .viewer.close -side right -padx 20 -pady 5 -ipadx 5 -in .viewer.bo
	    pack .viewer.bo -side bottom -fill x
	    pack .viewer.s1 -side right  -fill y
	    pack .viewer.t1 -side top -expand yes -fill both
	    if { $viewerStack(length) == 0 } {
		.viewer.back configure -state disabled
	    }
	    
	    # -- read necessary file, mark all functions and classes
	    # -- highlight selected item
	    readFile .viewer.t1 $file
	    markAllFuncAndClass .viewer.t1 $file $progfile
	    show$what .viewer.t1 $progfile $tag $file $line
	} else {
	    
	    
	    #
	    # Another binding, for scrolling through errors. Necessary redundancy,
	    # since the viewer and fancy have separate toplevel windows.
	    # 4/15 SAS
	    #
	    bind .viewer <Alt-n> fancy_Altn_global_binding
	    bind .viewer <Alt-p> fancy_Altp_global_binding
	    



	    # -- there is already a text window
	    if { $file == $showFile(name) } {
		if { $tag != $showFile(tag) || $what != $showFile(what) \
			|| $progfile != $showFile(progfile) \
			|| $what == "Line" } {
		    # -- same file, but different selected item
		    # -- Or we're dealing with a new *line*
		    # -- un-highlight old one, highlight new one
		    eval .viewer.t1 tag delete dfunc bfunc body line \
			    [lpick [.viewer.t1 tag names] \
			    {dchild* bchild* dbase* bbase* ctype}]
		    show$what .viewer.t1 $progfile $tag $file $line
		}
		
		# Need some modification here. Differentiating lines are not distiguished.
		
		
		
		
	    } else {
		# -- different file; update title
		wm title .viewer "$depfile(host):[normalizePath $depfile(dir) $file]"
		.viewer.t1 configure -state normal
		.viewer.t1 delete 1.0 end
		
		# -- un-highlight old selected items
		# -- this is necessary, otherwise tags which are used for highlightning
		# -- would be applied to new text
		eval .viewer.t1 tag delete dfunc bfunc body line \
			[lpick [.viewer.t1 tag names] \
			{dchild* bchild* dbase* bbase* ctype func* class*}]
		
		# -- read new file, mark all functions and classes
		# -- highlight selected item
		readFile .viewer.t1 $file
		markAllFuncAndClass .viewer.t1 $file $progfile
		show$what .viewer.t1 $progfile $tag $file $line
	    }
	}
	# -- remember currently selected item and its file
	if { !$back && $showFile(tag) != "NONE" && \
		($tag != $showFile(tag) || \
		$progfile != $showFile(progfile)) } {
	    pushStack viewerStack \
		    [list $showFile(progfile) $showFile(tag)]
	    .viewer.back configure -state normal
	}

	set showFile(name) $file
	set showFile(tag)  $tag
	set showFile(what) $what
	set showFile(progfile) $progfile
    }
}

set selectBox(File) "";    # currently selected file in Files listbox
set selectBox(Func) "";    # currently selected function in Functions listbox
set selectBox(Class) "";   # currently selected class in Classes listbox
set selectBox(Method) "";  # currently selected method in Methods listbox

#
# processFiles: process mouse buttons for Files listbox
#
#       button: number of pressed button
#          win: pathname of Files listbox
#            y: y coordinate within window where button was pressed
#

proc processFiles {button win y} {
  global myself \
	  selectBox \
	  fancy_funclist

  update idletasks

  # -- ignore button 2 in File listbox
  if { $button == 2 } { return }

  # -- find item selected, bring it in the middle if necessary, and remember it
  set i [$win nearest $y]
  if { $i == -1 } { return }
  selectItem $win $i
  set selectBox(File) [.$myself.fi.l1 get $i]

  # -- load the right depfile
  set allprogfiles [PM_GetFiles]
  if {[lsearch -exact $allprogfiles $selectBox(File)] >= 0} {
      loadDep $selectBox(File)
  } else {
      foreach progfile $allprogfiles {
	  if {[lsearch -exact [PM_GetHeaders $progfile] $selectBox(File)] \
		  >= 0} {
	      loadDep $progfile
	      break
	  }
      }
  }

  # -- update Function listbox according to select file
  set fancy_funclist [lsort [Cgm_FuncList $selectBox(File)]]
  redrawSelectBox .$myself.fu [format "Functions (%s)" $selectBox(File)] \
	  $fancy_funclist
}

#
# processFunctions: process mouse buttons for Functions listbox
#
#           button: number of pressed button
#              win: pathname of Functions listbox
#                y: y coordinate within window where button was pressed
#

proc processFunctions {button win y} {
  global fancy_funclist

  update idletasks

  # -- find item selected, and invoke global features depending on button
  set i [$win nearest $y]
  if { $i == -1 } { return }
  set s [lindex $fancy_funclist $i]

  if { $button == 2 } {
    PM_GlobalSelect [lindex $s 1] global_showFuncTag  [lindex $s 2]
  } else {
    PM_GlobalSelect [lindex $s 1] global_selectFuncTag [lindex $s 2]
  }
}

#
# processClasses: process mouse buttons for Classes listbox
#
#         button: number of pressed button
#            win: pathname of Classes listbox
#              y: y coordinate within window where button was pressed
#

proc processClasses {button win y} {
  global myself \
	  selectBox \
	  allclasses \
	  fancy_methodlist

  update idletasks

  # -- ignore button 2 in File listbox
  if { $button == 2 } { return }

  # -- find item selected, bring it in the middle if necessary, and remember it
  set i [$win nearest $y]
  if { $i == -1 } { return }
  selectItem $win $i
  set c [lindex $allclasses $i]
  set selectBox(Class) $c

  # -- load the right depfile  
  loadDep [lindex $c 1]

  # -- update Methods listbox according to select class
  set fancy_methodlist \
	  [lsort [Cgm_MethodList [lindex $c 1] [lindex $c 0] [lindex $c 2]]]
  redrawSelectBox .$myself.me \
	  [format "Methods (%s)" [lindex $selectBox(Class) 0]] \
	  $fancy_methodlist

  # -- invoke local / global selection depending on button
  if { $button == 1 } {
    selectClassTag [lindex $c 1] [lindex $c 2] 0
  } else {
    PM_GlobalSelect [lindex $c 1] global_selectClassTag [lindex $c 2]
  }
}

#
# processMethods: process mouse buttons for Methods listbox
#
#         button: number of pressed button
#            win: pathname of Methods listbox
#              y: y coordinate within window where button was pressed
#

proc processMethods {button win y} {
  global fancy_methodlist

  update idletasks

  # -- find item selected, and invoke global features depending on button
  set i [$win nearest $y]
  if { $i == -1 } { return }
  set s [lindex $fancy_methodlist $i]

  if { $button == 2 } {
    PM_GlobalSelect [lindex $s 1] global_showFuncTag [lindex $s 2]
  } else {
    PM_GlobalSelect [lindex $s 1] global_selectFuncTag [lindex $s 2]
  }
}

#
# proc selectClassTag: implementation of global feature "selectClass"
#                      fancy selects class in Classes listbox and shows its
#                      members in the Members listbox
#                      Also, it highlights the body of the class declaration
#
#                ctag: Sage++ id of class to select
#            showdecl: highlight body?
#

proc selectClassTag {progfile ctag {showdecl 1} {back 0}} {
  global myself \
	  allclasses \
	  selectBox

    loadDep $progfile

    if {[ Cgm_ClassInfo $progfile $ctag name] != "CGM_DOESNT_EXIST" } {
    # -- class is known to fancy, select it
    set class [Cgm_ClassInfo $progfile $ctag name]
    if { $class != $selectBox(Class) } {
      # -- class not already selected, select it
      set i [lsearch -exact $allclasses [list $class $progfile $ctag]]
      if { $i != -1 } { selectItem .$myself.cl.l1 $i }
      set selectBox(Class) $class

      # -- update Members listbox
      set fancy_methodlist \
	      [lsort [Cgm_MethodList $progfile $selectBox(Class) $ctag]]
      redrawSelectBox .$myself.me \
	      [format "Methods (%s)" [lindex $selectBox(Class) 0]] \
	      $fancy_methodlist
    }

    # -- highlight body, if necessary
    if { $showdecl } {
      showFile $progfile $ctag Class "" - $back
    }
  } else {
    # -- class is unknown to fancy, un-select previous one
    set selectBox(Class) ""
    set selectBox(Method) ""
    redrawSelectBox .$myself.cl "Classes" $allclasses
    redrawSelectBox .$myself.me "Methods" {}
  }
}

#
# processTag: update listboxes if function is selected in text window
#
#        tag: Sage+= id of selected item
#

proc processTag {progfile tag} {
  global myself \
	  selectBox \
	  allfiles allclasses \
	  fancy_funclist fancy_methodlist

  set class [lindex [Cgm_FuncInfo $progfile $tag class] 0]
  set name  [Cgm_FuncInfo $progfile $tag name]

  # -- get file
  set file [lindex [Cgm_FuncInfo $progfile $tag file] 1]
  
  if { $file == "-" } { return }
  set list [lsort [Cgm_FuncList $file]]
  set fancy_funclist $list

  # -- update Files listbox, if necessary
  if { $file != $selectBox(File) } {
    set i [lsearch -exact $allfiles $file]
    selectItem .$myself.fi.l1 $i
    set selectBox(File) $file
    redrawSelectBox .$myself.fu "Functions ($file)" $list
  }

  if { $class == "-" } {
    # -- no member func: update Functions listbox
    set i [lsearch -exact $list [list $name $progfile $tag]]
    selectItem .$myself.fu.l1 $i
    set selectBox(Func) $tag
  } else {
    # -- selected function is member function: get class
    set ctag [lindex [Cgm_FuncInfo $progfile $tag class] 2]
    set list [lsort [Cgm_MethodList $progfile $class $ctag]]
    set fancy_methodlist $list

    # -- update Classes listbox, if necessary
    if { $class != $selectBox(Class) } {
      set i [lsearch -exact $allclasses [list $class $progfile $ctag]]
      selectItem .$myself.cl.l1 $i
      set selectBox(Class) $class
      redrawSelectBox .$myself.me "Methods ($class)" $list
    }

    # -- update methods listbox
    set i [lsearch -exact $list [list $name $progfile $tag]]
    selectItem .$myself.me.l1 $i
    set selectBox(method) $tag
  }
}

#
# selectItem: select and display item in listbox
#
#        win: pathname of listbox
#      index: index of selected item
#

proc selectItem {win index} {
    $win see $index
    $win selection clear 0 end
    $win selection set $index
}

#
# selectBox: create and display new selectbox (listbox+title+scrollbar)
#
#       win: pathname of selectBox
#     title: title for selectBox
#     names: list of items to display in listbox
#

proc selectBox {win title names} {
  global myself

  frame $win
  label $win.t -relief raised -text $title
  listbox $win.l1 -relief sunken -width 30 -height 10 -exportselection false \
	  -selectmode single
  foreach n $names {
    $win.l1 insert end [lindex $n 0]
  }
  scrollbar $win.s1 -orient vert -relief sunken -command "$win.l1 yview"
  $win.l1 configure -yscroll "$win.s1 set"
  scrollbar $win.s2 -orient horiz -relief sunken -command "$win.l1 xview"
  $win.l1 configure -xscroll "$win.s2 set"

  pack $win.t -side top -fill x
  pack $win.s1 -side right -fill y
  pack $win.l1 $win.s2 -side top -fill x

  bind $win.l1 <Button-1> "process$title 1 $win.l1 %y"
  bind $win.l1 <Button-2> "process$title 2 $win.l1 %y"
  bind $win.l1 <Button-3> "process$title 3 $win.l1 %y"
  set bt [list Listbox $win.l1 .$myself all]
  bindtags $win.l1 $bt

  return $win
}

#
# redrawSelectBox: update selectBox
#
#             win: pathname of selectBox
#        newtitle: new title for selectBox
#           names: new list of items to display in listbox
#

proc redrawSelectBox {win newtitle names} {
  $win.l1 delete 0 end
  foreach n $names {
    $win.l1 insert end [lindex $n 0]
  }
  set strlen [string length $newtitle]
  if {$strlen > 27} {
      regexp {([a-zA-Z]+) } $newtitle fullmatch keyword
      set newtitle [format "%s ...%s" $keyword \
	      [string range $newtitle [expr $strlen - 27] end]]
  }
  $win.t configure -text $newtitle
}


#
# loadDep - load the depfile for a given progfile
#
proc loadDep {progfile} {
  global \
	  depfile \
	  allclasses \
	  allfiles

    # abort if already loaded
    if [Cgm_IsDepLoaded $progfile] {
	return
    }

    # else loaded it
    displayMessage hourglass "Loading browser info..."
    Cgm_LoadDep $progfile -dumptxt
    removeMessage
}


# initAddInfo: Setup additional project info when starting fancy.
#
# Sets up:
#       - List of files  (can get from PM)
#       - List of classes
proc initAddInfo {} {
    global \
	    allfiles \
	    allclasses
    
    set allfiles [concat \
	    [lsort [PM_GetFiles]] \
	    [lsort [PM_GetHeaders]]]
    set allclasses [lsort [Bdb_GetClasses]]
}

#
# createWindow: create and display main window of fancy
#

proc createWindow {} {
  global TAUDIR
  global myself
  global allclasses allfiles
  global editor

  toplevel .$myself
  wm title .$myself "FANCY"
  wm iconbitmap .$myself @$TAUDIR/xbm/$myself.xbm

  frame .$myself.mbar -relief raised -borderwidth 2
  menubutton .$myself.mbar.b1 -text File -menu .$myself.mbar.b1.m1 -underline 0
  menu .$myself.mbar.b1.m1
  .$myself.mbar.b1.m1 add command -label "Exit" -underline 0 -command "exit"

  menubutton .$myself.mbar.b2 -text Options -menu .$myself.mbar.b2.m1 \
	  -underline 0
  menu .$myself.mbar.b2.m1
    .$myself.mbar.b2.m1 add checkbutton -label "View with Editor" \
      -variable editor(view_with)
    #
    # Menu options for scrolling through any generated errors
    # Added 4/15, SAS
    #
    
    .$myself.mbar.b2.m1 add separator
    .$myself.mbar.b2.m1 add command -label "Next Error" \
      -command fancy_Altn_global_binding \
      -accelerator {Alt-n}
    .$myself.mbar.b2.m1 add command -label "Previous Error" \
      -command fancy_Altp_global_binding \
      -accelerator {Alt-p}
    
  createToolMenu .$myself.mbar.b4

  menubutton .$myself.mbar.b3 -text Help -menu .$myself.mbar.b3.m1 -underline 0
  menu .$myself.mbar.b3.m1
  .$myself.mbar.b3.m1 add command -label "on $myself" -underline 3 \
                   -command "xsend tau \[list showHelp $myself 1-$myself 1\]"
  .$myself.mbar.b3.m1 add command -label "on viewer" -underline 3 \
                   -command "xsend tau \[list showHelp $myself 1.1-viewer 1\]"
  .$myself.mbar.b3.m1 add separator
  .$myself.mbar.b3.m1 add command -label "on using help" -underline 3 \
                   -command "xsend tau {showHelp general 1-help 1}"

  pack .$myself.mbar.b1 -side left -padx 5
  pack .$myself.mbar.b2 -side left -padx 5
  pack .$myself.mbar.b3 .$myself.mbar.b4 -side right -padx 5
  pack .$myself.mbar -side top -fill x

  frame .$myself.top
  selectBox .$myself.fi Files $allfiles
  selectBox .$myself.fu Functions {}
  pack .$myself.fi .$myself.fu -side left -padx 15 -pady 15 -in .$myself.top
  pack .$myself.top -side top

  frame .$myself.bottom
  selectBox .$myself.cl Classes $allclasses
  selectBox .$myself.me Methods {}
  pack .$myself.cl .$myself.me -side left -padx 15 -pady 15 -in .$myself.bottom

  if { $allclasses != "" } { pack .$myself.bottom -side top }

#
# Added 4/15, SAS
# A global binding, so that Alt-n and Alt-p scrol through any errors 
# generated in Cosy in the Fancy editor window, if it exists.
#

    bind .$myself <Alt-n> fancy_Altn_global_binding
    bind .$myself <Alt-p> fancy_Altp_global_binding


}


proc resetGUI {{keep_contents}} {
    global myself depfile \
	    showFile selectBox \
	    fancy_funclist fancy_methodlist \
	    allfiles allclasses \
	    viewerStack

    initAddInfo

    if {$keep_contents} {
	redrawSelectBox .$myself.fi "Files" $allfiles
	set i [lsearch -exact $allfiles $selectBox(File)]
	if {$i >= 0} {
	    selectItem .$myself.fi.l1 $i
	}
    } else {
	set selectBox(File) ""
	set selectBox(Func) ""
	set selectBox(Class) ""
	set selectBox(Method) ""
	redrawSelectBox .$myself.fi "Files" $allfiles
	redrawSelectBox .$myself.cl "Classes" $allclasses
	redrawSelectBox .$myself.fu "Functions" {}
	redrawSelectBox .$myself.me "Methods" {}
	set fancy_funclist      [list]
	set fancy_methodlist    [list]

	initStack viewerStack
	if [winfo exists .viewer] {
	    destroy .viewer
	}
    }
    
    if { $allclasses != "" } {
	pack .$myself.bottom -side top
    } else {
	pack forget .$myself.bottom
    }
}


proc Tool_AcceptChanges {progfiles flag} {
    global myself depfile \
	    showFile selectBox

    switch $flag {

        d { 
	    foreach pf $progfiles {
		Cgm_RemoveDep $pf
		if {$showFile(progfile) == $pf} {
		    if [winfo exists .viewer] {
			destroy .viewer
		    }
		    set showFile(name) ""
		    set showFile(progfile) ""
		    set showFile(tag)  NONE
		    set showFile(what) NONE
		}
	    }
	    resetGUI 0
        }


        a { 
            resetGUI 1
        }


        u { 
	    foreach pf $progfiles {
		Cgm_RemoveDep $pf
		if {$showFile(progfile) == $pf} {
		    if [winfo exists .viewer] {
			destroy .viewer
		    }
		    set showFile(name) ""
		    set showFile(progfile) ""
		    set showFile(tag)  NONE
		    set showFile(what) NONE
		}
	    }
            resetGUI 0
        }
	
	p {
	    Cgm_RemoveAllDeps
	    set pm_status [PM_Status]
	    if {$pm_status != "NO_PROJECT"} {
		set depfile(project) [lindex $pm_status 0]
		set depfile(host)    [lindex $pm_status 1]
		set depfile(arch)    [lindex $pm_status 2]
		set depfile(root)    [lindex $pm_status 3]
		set depfile(dir)     [lindex $pm_status 4]
	    }
	    # Check for language-tool compatibility
	    if {![Lang_CheckCompatibility [PM_GetProjectLangs] $myself]} {
		showError "$myself is not compatible with the project language(s)."
		exit
	    }

	    # Reset state
	    global showFile allclasses
	    set showFile(name) "";    
	    set showFile(progfile) "";
	    set showFile(tag)  NONE;  
	    set showFile(what) NONE;  
	    set allclasses "";        
	    resetGUI 0
	}

	e {
	    #This is a flag for updating during execution. No action is needed here.
	}
    }
}



# ------------
# -- main code
# ------------
if {$argc == 0} {
  set parg [pwd]
} elseif {$argc == 1} {
    set parg [lindex $argv 0]
    if {[file extension $parg] != ".pmf"} {
	set parg "$parg.pmf"
    }
} else {
  puts stderr "usage: $myself \[\[host:\]projFile \| \[host:\]directory\]"
  exit
}

# Init the project manager (taud)
launchTauDaemon -waitfor
PM_AddTool $myself
PM_AddGlobalSelect $myself {global_selectLine global_selectFuncTag global_selectClassTag }

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

# Tool Init
initAddInfo
#computeAddInfo
createWindow
launchTAU

wm protocol .$myself WM_DELETE_WINDOW exit

removeMessage
