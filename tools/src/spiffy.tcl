#*********************************************************************#
#*                TAU/pC++/Sage++  Copyright (C) 1994                *#
#*                        Jerry Manic Saftware                       *#
#*  Indiana University  University of Oregon  University of Rennes   *#
#*********************************************************************#
displayMessage hourglass "Loading $myself..."

###
### TODO
###
###   - Look into possible problems with remote use.
###   - Color file names in check list based on clean/dirty.
###   - Color file names in check list by whether they are in depfile.
###   - language sensitive man pages?
###   - Save options in user's preferences.
###

# This is spiffy, based on fancy.
# lth@cs.uoregon.edu / 950523
# kurtw@cs.uoregon.edu

###
### Exported Global Varaibles
###   jedit_file_open - array indexed by the filename and giving a boolean
###                     indicating whether the file is open.  Bound to the
###                     the checklist.
###   jedit_file_w    - array indexed by the filename and giving the 
###                     editor text widget
###   SPIFFY(global)  - boolean indicating whether to do global select.
###   SPIFFY(msg)     - Message string to display at the top of the window.
###   SPIFFY(autosave) - boolean indicating whether to autosave
###   SPIFFY(selection) - return from the Bdb_Querry
###


source "$TAUDIR/inc/depfile.tcl"
source "$TAUDIR/inc/stack.tcl"
source "$TAUDIR/inc/checklist.tcl"
source "$TAUDIR/inc/fileio.tcl"


set SPIFFY(msg) "Click a file to open/close:"
set SPIFFY(global) 1
set SPIFFY(autosave) 0
set SPIFFY(selection) ""


#
# A global SPIFFY keybinding, to allow the user to scroll through the
# error messages in the jedit window created by Spiffy for examining
# the source code.
# SAS 4/9/96
#

proc spiffy_Altn_global_binding {} {
    set avail [winfo interps]
    if { [ lsearch -exact $avail cosy] == -1 } {
	EditorMsg "Cosy isn't running; no errors to parse."
	after 1000 EditorMsg
    } else {
	async_send cosy errScrollForward
    }
}


proc spiffy_Altp_global_binding {} {
    set avail [winfo interps]
    if { [ lsearch -exact $avail cosy] == -1 } {
	EditorMsg "Cosy isn't running; no errors to parse."
	after 1000 EditorMsg
    } else {
	aysnc_send cosy errScrollBackward
    }
}


#
# exit - TAU exit function that communicates the event to other tau tools.
#
rename exit exit.old
proc exit {{status 0}} {
    global jedit_file_w SPIFFY myself

    # Autosave
    if {[info exists jedit_file_w]} {
	if {[llength [array names jedit_file_w]] > 0} {
	    if {$SPIFFY(autosave)} {
		saveAll
	    } else {
		if {![winfo exists .exitdialog]} {
		    set cont [tk_dialog .exitdialog "Exit" \
			    "Unsaved changes will be lost." \
			    question 1 "Cancel" "Exit without saving"]
		    if {$cont == 0} {
			return
		    }
		}
	    }
	}

	PM_RemGlobalSelect $myself {global_selectLine global_selectFuncTag global_selectClassTag }
	PM_RemTool $myself
	after 1000 "exit.old $status"
    } else {
	PM_RemTool $myself
	exit.old $status
    }
}


# 
# saveAll - Save all open jedit files.
#
proc saveAll {} {
    global jedit_file_w depfile

    # Autosave
    if {[info exists jedit_file_w]} {
	if {[llength [array names jedit_file_w]] > 0} {
	    EditorMsg "Autosaving..."
	    foreach file [array names jedit_file_w] {
		jedit:cmd:save $jedit_file_w($file)
	    }      
	}
    }
}


#
# absolute_path - returns the absolute path of a file or directory
#    
#       rel_pat - path relative to the current directory given by pwd
#
proc absolute_path {rel_path} {
    set currdir [pwd]
    set filename [file tail $rel_path]
    set rel_path [file dirname $rel_path]

    cd $rel_path
    set abs_path [format "%s/%s" [pwd] $filename]
    cd $currdir

    return $abs_path
}


#
# selectLine: implementation of global feature "selectLine" for spiffy
#             highlight specified line in file
#
#       line: line number to select
#       file: file name to select
#        tag: tag of call site to select or -1 (UNUSED)
#   progfile: depfile containing the tag (UNUSED)
#
proc selectLine {progfile tag line file} {
  showFile $progfile $tag Line $file $line
}


#
# selectFuncTag: implementation of global feature "selectFunction" for spiffy
#                display body of selected function in text window
#
#      progfile: depfile containing the tag (UNUSED)
#           tag: id of function to select
#
proc selectFuncTag {progfile tag {back 0}} {
    global SPIFFY

    if {$SPIFFY(selection) == ""} {
	set SPIFFY(selection) [Bdb_QueryFuncTag $progfile $tag]
    }
    showFile $progfile $tag Func "" -
}


#
# proc selectClassTag: implementation of global feature "selectClass"
#
#            progfile: depfile containing the tag (UNUSED)
#                ctag: Sage++ id of class to select
#            showdecl: highlight body?
#

proc selectClassTag {progfile ctag {showdecl 1} {back 0}} {
    global SPIFFY

    if {$SPIFFY(selection) == ""} {
	set SPIFFY(selection) [Bdb_QueryClassTag $progfile $tag]
    }
    showFile $progfile $ctag Class "" -
}


#
# showFile: show body of function or class in text window
#
# progfile: depfile containing the tag
#      tag: Sage++ id of selected item
#     what: type of selected item: Func, Class, or Line
#
proc showFile {progfile tag what {fn ""} {ln -}} {
    global myself \
	    depfile \
	    TAUDIR \
	    SPIFFY
    
  # -- get necessary info from global database
  # -- Note that the Depfile is already loaded.
  switch -exact $what {
    Func  {
            set file [lindex $SPIFFY(selection) 3]
            set line [lindex $SPIFFY(selection) 2]
            set name [lindex $SPIFFY(selection) 4]
          }
    Class {
            set file [lindex $SPIFFY(selection) 3]
            set line [lindex $SPIFFY(selection) 2]
            set name [lindex $SPIFFY(selection) 4]
          }
    Line  {
            set file $fn
            set line $ln
            if {$tag != -1} {
		set name [Cgm_FuncInfo $progfile $tag name]
	    } else {
		set name ""
	    }
          }
    default {
            puts stderr "$myself: internal error: invalid what $what"
            exit
          }
  }

  if { $file == "-" } {
    showError "Filename for function or class `$name' not specified."
    return
  }

  set SPIFFY(selection) ""
  Editor_ShowFile $file $line $name
}



proc resetGUI {} {
  global myself

  # -- reset litboxes
  destroy .$myself.flist
  set button_list [list]
  foreach file [concat [PM_GetFiles] [PM_GetHeaders]] {
      lappend button_list [list \
	      $file \
	      [list checklist_callback $file] \
	      jedit_file_open($file) ]
  }
  checklist .$myself.flist $button_list 300
  pack .$myself.flist
}


#
# createWindow: create and display main window of spiffy
#
proc createWindow {} {
    global TAUDIR
    global myself 
    global allfiles
    global jedit_file_open
    global SPIFFY
    global depfile
    
    toplevel .$myself
    wm title .$myself "SPIFFY"
    wm iconbitmap .$myself @$TAUDIR/xbm/$myself.xbm
    wm protocol .$myself WM_DELETE_WINDOW "saveAll; exit"
    wm protocol .$myself WM_SAVE_YOURSELF saveAll
    frame .$myself.mbar -relief raised -borderwidth 2

    #
    # The Alt-n and Alt-p binding:
    # Allows for scrolling through the list of generated errors
    # in both directions.
    # SAS 4/9
    #
    
    bind .$myself <Alt-n> spiffy_Altn_global_binding
    bind .$myself <Alt-p> spiffy_Altp_global_binding

    # File Menu
    menubutton .$myself.mbar.b1 -text File -menu .$myself.mbar.b1.m1 \
	    -underline 0
    menu .$myself.mbar.b1.m1
    .$myself.mbar.b1.m1 add command -label "New" -underline 0 \
	    -command {jedit:jedit -mode tau}
    .$myself.mbar.b1.m1 add command -label "Open..." -underline 0 \
	    -command \
	    {jedit:cmd:load [jedit:top_to_text [jedit:jedit -mode tau]]}
    .$myself.mbar.b1.m1 add command -label "Save All" -underline 0 \
	    -command "saveAll"
    .$myself.mbar.b1.m1 add separator
    .$myself.mbar.b1.m1 add command -label "Exit" -underline 0 -command "exit"

    # Options Menu
    menubutton .$myself.mbar.b2 -text Options -menu .$myself.mbar.b2.m1 \
	    -underline 0
    menu .$myself.mbar.b2.m1
    .$myself.mbar.b2.m1 add checkbutton -label "Tau Global Select" \
	    -variable SPIFFY(global)
    .$myself.mbar.b2.m1 add checkbutton -label "Autosave on Close" \
	    -variable SPIFFY(autosave)

    # Help Menu
    menubutton .$myself.mbar.b3 -text Help -menu .$myself.mbar.b3.m1 \
	    -underline 0
    menu .$myself.mbar.b3.m1
    .$myself.mbar.b3.m1 add command -label "on $myself" -underline 3 \
	    -command "xsend tau \[list showHelp $myself 1-$myself 1\]"
    .$myself.mbar.b3.m1 add command -label "on jedit" -underline 3 \
	    -command "exec jdoc jedit &"
    .$myself.mbar.b3.m1 add command -label "on tau editing mode" -underline 3 \
	    -command "xsend tau \[list showHelp $myself 1.1-taumode 1\]"
    .$myself.mbar.b3.m1 add separator
    .$myself.mbar.b3.m1 add command -label "on using help" -underline 3 \
	    -command "xsend tau {showHelp general 1-help 1}"

    createToolMenu .$myself.mbar.b4

    pack .$myself.mbar.b1 -side left -padx 5
    pack .$myself.mbar.b2 -side left -padx 5
    pack .$myself.mbar.b3 .$myself.mbar.b4 -side right -padx 5
    pack .$myself.mbar -side top -fill x
    
    # Message Line
    label .$myself.label -text $SPIFFY(msg) -width 42
    pack .$myself.label -side top -fill x

    # File List
    set button_list [list]
    foreach file [concat [PM_GetFiles] [PM_GetHeaders]] {
	lappend button_list [list \
		$file \
		[list checklist_callback $file] \
		jedit_file_open($file) ]
    }
    checklist .$myself.flist $button_list 300
    pack .$myself.flist -side top -fill both -expand yes
    wm minsize .$myself 331 183
}


proc checklist_callback {file checkbut} {
    global jedit_file_open jedit_file_w SPIFFY depfile

    EditorMsg "Opening editor window..."
    update idletasks

    if {$jedit_file_open($file)} {

	# File is unopened - open it
	Editor_ShowFile $file

    } else {
	
	# File is already open - raise it
	set jedit_file_open($file) 1
	if {![info exists jedit_file_w($file)]} return
	wm deiconify [jedit:text_to_top $jedit_file_w($file)]
	raise [jedit:text_to_top $jedit_file_w($file)]
    }
    EditorMsg
}


proc Editor_ShowFile {file {line -1} {name ""}} {
    global jedit_file_w jedit_file_open depfile

    # If it's not associated w/ a button and it's absolute, relativize it.
    if {![info exists jedit_file_open($file)] && \
	    [string index $file 0] == "/"} {
	set file [RelativizePath $depfile(host) $depfile(dir) $file]
    }

    # if this file is not already open, start a new editor window
    if {![info exists jedit_file_w($file)]} {
	jedit:jedit -mode tau -file $file
    } else {
	wm deiconify [jedit:text_to_top $jedit_file_w($file)]
	raise [jedit:text_to_top $jedit_file_w($file)]
    }

    set t $jedit_file_w($file)
    set jedit_file_open($file) 1

    # Go to the right place in the file
    if {$line >= 0} {
	jedit:go_to_line $t $line

	#move to start of the line
	j:text:move $t $line.0

	if {$name != ""} {
	    j:find:find_pattern $name $t
	} else {
	    $t tag add sel $line.0 $line.end
	}
    }
}


# 
# EditorMsg - display the given message in the spiffy file window.
#
#      msg - the message to display (optional).  This is argument is
#            not given, the message will be the default stored in the
#            editor_msg global.
#
proc EditorMsg {{msg ""}} {
    global myself SPIFFY
    
    if {$msg == ""} {
	.$myself.label configure -text $SPIFFY(msg)
    } else {
	.$myself.label configure -text $msg	
    }
}
    

proc Tool_AcceptChanges {progfiles flag} {
    global myself depfile
    global jedit_file_w jedit_file_open SPIFFY
    global JEDIT_WINDOW_COUNT 

    switch $flag {

	d {
	    foreach pf $progfiles {
		Cgm_RemoveDep $pf
	    
		set file $pf
		if {[info exists jedit_file_w($file)]} {
		    if [j:text:is_dirty $jedit_file_w($file)] {
			set save [tk_dialog .savedialog "Save" \
				"Closing $file.  Save changes?" \
				question 1 "Don't Save" "OK"]
			if {$save == 1} {
			    jedit:cmd:save $jedit_file_w($file)
			}			  
		    }
		    incr JEDIT_WINDOW_COUNT -1      ;# one fewer window
		    destroy [jedit:text_to_top $jedit_file_w($file)]
		    unset jedit_file_w($file)
		    unset jedit_file_open($file)
		}
	    }
	    resetGUI
	}


	a {
	    resetGUI
	}


	u {
	    # Update a file
	    foreach pf $progfiles { Cgm_RemoveDep $pf; }

	    # Reset GUI
	    resetGUI
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
	    # Reset GUI
	    resetGUI
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

# Initialize jstools for the editor
j:jstools_init 

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
createWindow
launchTAU
removeMessage


