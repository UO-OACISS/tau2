##
### tau-mode.tcl
### Kurt Windisch (kurtw@cs.uoregon.edu)
### Created 11/13/95
###
### Tau/pC++ editing mode for the Spiffy/Jedit editor.  This file 
### implements all of the tau-specific editor functionality: special
### menus and button, quit/close overrides, show file, hyper,
### show function, and autoindenting.
###
### Exported global variables:
###   jedit_file_w    - array of text widget names indexed by the filename
###   jedit_file_open - array of booleans indexed by filename indicating 
###                     whether the file is open.
###   
### Local global varaibles:
###   TAU_MODE(multi_geometry) - tk geometry string for the hyper_select window
###   TAU_MODE(indent) - the number of character to autoindent
###   TAU_MODE(disable_flag) - boolean, 1 if the file being opened is disabled.
###   

### ToDo
###   - put tkman info in preferences.
###   - C++ mode


if {[info procs FileIO_ls] == ""} {
    source "$TAUDIR/inc/fileio.tcl"
    set depfile(host) "localhost"
    set depfile(dir)  "."
}



#
# mode:tau:in_tau - Determines if the TAU tools are running or if jedit is
#                   being used stand-alone.  Returns boolean.
#
proc mode:tau:in_tau {} {
    global SPIFFY

    if {[info exists SPIFFY(global)]} {
	return 1
    } else {
	return 0
    }
}


#
# mode:tau:init - initialize the tau mode and redefine the file modes.
#
#             t - the jedit text widget
#
proc mode:tau:init { t } {
    global JEDIT_MODEPREFS TAU_MODE

    j:read_prefs { {bindings emacs} }

    j:read_prefs -array JEDIT_MODEPREFS -prefix tau \
	    -directory ~/.tk/jeditmodes -file tau-defaults {
	{textfont default}
	{textwidth 80}
	{textheight 24}
	{textwrap char}
	{sabbrev 0}
	{dabbrev 0}
	{autobreak 0}
	{autoindent 1}
	{parenflash 1}
	{savestate 0}
	{buttonbar 1}
	{menu,editor 1}
	{menu,file 1}
	{menu,edit 1}
	{menu,prefs 0}
	{menu,abbrev 1} 
	{menu,filter 1} 
	{menu,format 0}
	{menu,display 0}
	{menu,mode1 1}
	{menu,mode2 1}
	{menu,user 1}
    }

  set TAU_MODE(indent) 2		;# number of chars per nesting level
  set TAU_MODE(multi_geometry) ""       ;# position geometry of the multiwindow
  set TAU_MODE(disable_flag) 0          ;# boolean: disable file being opened?

  # Handle Window Manager events nicely
  wm protocol [jedit:text_to_top $t] WM_DELETE_WINDOW \
	  "jedit:cmd:save $t {}; mode:tau:close $t"
  wm protocol [jedit:text_to_top $t] WM_SAVE_YOURSELF \
	  "jedit:cmd:save $t {}"
}


proc wmdelete {t} {
    if {![winfo exists .savedialog]} {
	jedit:cmd:close $t {}
    } elseif {![winfo ismapped .savedialog]} {
	jedit:cmd:close $t {}
    }
}

#
# mode:tau:cleanup - Cleanup anything left by tau-mode when switching modes.
#
#                t - text widget name
proc mode:tau:cleanup t {
    
    # Destroy the multi window if it exists
    if {[winfo exists .spiffymulti]} {
	destroy .spiffymulti
    }

    # Destroy the man window if it exists
    if {[winfo exists .man]} {
	destroy .man
    }
}


#
# mode:tau:mkmenu1 - Make the tau functionality menu
#
#             menu - menu widget name
#                t - text widget name
#
proc mode:tau:mkmenu1 { menu t } {
    menubutton $menu -text {Tau} -menu $menu.m
    
    menu $menu.m
    $menu.m add command -label {Local Hyper Select} \
	    -command "mode:tau:hyper $t local" \
	    -accelerator {^z}
    $menu.m add command -label {Global Hyper Select} \
	    -command "mode:tau:hyper $t global" \
	    -accelerator {^Z}
    $menu.m add command -label {Function Info} \
	    -command "mode:tau:func_info $t" \
	    -accelerator {^u}
    $menu.m add separator
    $menu.m add command -label {Show Manual Page} \
	    -command "mode:tau:man_page $t" \
	    -accelerator {^m}
    $menu.m add command -label {Build} \
	    -command "mode:tau:build $t" \
	    -accelerator {^c}
    $menu.m add separator
    $menu.m add command -label {Add File to Project} \
	    -command "mode:tau:addfile $t"
    $menu.m add command -label {Remove File from Project} \
	    -command "mode:tau:remfile $t"
    $menu.m add separator
    $menu.m add command -label {TAU Help} \
	    -command "mode:tau:help $t"
    
#
# Added 4/10/96, SAS
#

    $menu.m add command -label {Next Error} \
	-command "async_send cosy errScrollForward" \
	-accelerator {Alt-n}
    $menu.m add command -label {Prev Error} \
	-command "async_send cosy errScrollBackward" \
	-accelerator {Alt-p}
#
# End additions SAS
#

    bind $t <Control-z> "mode:tau:hyper $t local"
    bind $t <Control-Z> "mode:tau:hyper $t global"
    bind $t <Control-u> "mode:tau:func_info $t"
    bind $t <Control-m> "mode:tau:man_page $t"
    bind $t <Control-c> "mode:tau:build $t"

#
# Added 4/10/96, SAS
#
    bind $t <Alt-n> "async_send cosy errScrollForward"
    bind $t <Alt-p> "async_send cosy errScrollBackward"
#
# End additions SAS
#
}


#
# mode:tau:mkbuttons - make the top row TAU-specific buttons.
#
#                  w - toplevel widget name
#                  t - text widget name
#
proc mode:tau:mkbuttons { w t } {
    j:buttonbar $w -pady 2 -buttons [format {
	{close "Close" {jedit:cmd:close %s {}}}
	{build "Build" {mode:tau:build %s}}
	{manp "Manual Page" {mode:tau:man_page %s}}
	{showf "Function Info" {mode:tau:func_info %s}}
	{hyper "Hyper Select" {mode:tau:hyper %s}}
    } $t $t $t $t $t] 
	
    return $w
}


proc mode:tau:help {t} {
  xsend tau "showHelp spiffy 1.1-taumode 1"
}


#
# mode:tau:pre_read_hook - called before a file is read into the editor.
#                          This funciton issues a warning to the user if
#                          a file is being opened that is currently open
#                          already.
#
#                filename - the name of the opened file
#                       t - the name of the text widget
if {[mode:tau:in_tau]} {
proc mode:tau:pre_read_hook {filename t} {
    global jedit_file_w global jedit_file_open SPIFFY depfile
    global TAU_MODE

    if {![info exists jedit_file_w]} return

    # If there's a file open in this window, close it first.
    foreach file [array names jedit_file_w] {
	if {$jedit_file_w($file) == $t} {
	    # Save and close the file
	    if {$SPIFFY(autosave)} {
		EditorMsg "Autosaving..."
		after 2500 EditorMsg
		mode:tau:pre_write_hook $file $t
		jedit:write $file $t
	    } else {
		set save [tk_dialog .savedialog "Save" \
			"Save changes to $file?" \
			question 2 "Cancel" "Don't Save" "OK"]
		if {$save == 0} {
		    # Cancel
		    return -code return
		} elseif {$save == 2} {
		    mode:tau:pre_write_hook $file $t
		    jedit:write $file $t
		}
	    }
	    unset jedit_file_w($file)
	    set jedit_file_open($file) 0
	    break
	}
    }

    # search the list of open files
    set names [array names jedit_file_w]
    for {set i 0} {$i < [llength $names]} {incr i 1} {
	if {[lindex $names $i] == $filename} {
		
	    # Warn and disable the new window
	    j:alert -title Alert -text "WARNING: You are attempting to open the same file in two windows.  The second window will be read-only and will not reflect changes made in the first.\n\nOnly the first window will be responsive to the TAU tools."
	    set TAU_MODE(disable_flag) 1
	    break
	}
    }
}
}


#
# mode:tau:read - Override the jedit read proc from jedit_io.tcl
#
proc mode:tau:read {filename t} {
    global depfile

    global JEDIT_MODEPREFS
    set mode [jedit:get_mode $t]
    j:default JEDIT_MODEPREFS($mode,savestate) 0
    set mode [jedit:get_mode $t]
    
    if {[string index $filename 0] != "/"} {
	set path "$depfile(dir)/$filename"
    } else {
	set path "$filename"
    }

    if {! [FileIO_file_exists $depfile(host) $path]} then {
      jedit:save_checkpoint $t			;# in case you were editing sth
      j:text:delete $t 1.0 end
      j:text:move $t 1.0
      #########################################	;# THIS ISN'T WORKING:
      jedit:set_label [jedit:text_to_top $t] [j:ldb JEio:...new_file]
      #########################################
    } else {
      if [FileIO_dir_exists $depfile(host) $path] {
        j:alert -text JEio:...is_directory
        return 1
      }
      if { ! [FileIO_file_readable $depfile(host) $path]} {
        j:alert -text JEio:...unreadable
        return 1
      }
      jedit:save_checkpoint $t			;# so you can undo a load
      # should do error checking
      j:text:delete $t 1.0 end
      $t insert end  [j:fileio:read $filename]
      j:text:move $t 1.0
      #
      if $JEDIT_MODEPREFS($mode,savestate) {
        jedit:read_annotation $filename $t
        jedit:yview_insert $t
      }
      jedit:save_checkpoint $t			;# alows undo to original state
    }
}


#
# mode:tau:post_read_hook - called after a file is read into the editor.
#                           This funciton adds an element to the array
#                           jedit_file_w, indexed by the name of the file
#                           that contains the
#                           name of the window's text widget.
#
#                filename - the name of the opened file
#                       t - the name of the text widget
if {[mode:tau:in_tau]} {
proc mode:tau:post_read_hook {filename t} {
    global jedit_file_w global jedit_file_open depfile TAU_MODE

    j:text:mark_clean $t

    # If it's not associated w/ a button and it's absolute, relativize it.
    if {![info exists jedit_file_open($filename)] && \
	    [string index $filename 0] == "/"} {
	set filename [RelativizePath $depfile(host) $depfile(dir) $filename]
    }

    if {$TAU_MODE(disable_flag) == 0} {
	set jedit_file_w($filename) $t
	set jedit_file_open($filename) 1
    } else {
	$t configure -state disabled
	$t configure -background [lindex [. configure -background] 4]
	jedit:set_label [jedit:text_to_top $t] \
		[format "%s (READ ONLY)" [file tail $filename]]
	set TAU_MODE(disable_flag) 0
    }
}
}

#
# mode:tau:pre_write_hook - called before a file is saved.  Implements 
#                           the autobackup function.
#
#                filename - the name of the opened file
#                       t - the name of the text widget
#
proc mode:tau:pre_write_hook {filename t} {
    if [file exists $filename] {
	set backupname "$filename~"
	exec cp $filename $backupname
    }
}


#
# mode:tau:pre_close_hook - Called before a file is closed to auto or prompt
#                           for saving.
#
if {[mode:tau:in_tau]} {
proc mode:tau:pre_close_hook t {
    global SPIFFY \
	    depfile

    # Not necessary if this window is disabled
    if {[lindex [$t configure -state] 4] == "disabled"} return

    # Save
    set file [jedit:get_filename $t]

    if {![info exists jedit_file_open($file)] && \
	    [string index $file 0] == "/"} {
	set file [RelativizePath $depfile(host) $depfile(dir) $file]
    }

    if [j:text:is_dirty $t] {
	if {$SPIFFY(autosave)} {
	    EditorMsg "Autosaving..."
	    update idletasks
	    after 2500 EditorMsg
	    jedit:cmd:save $t
	} else {
            set save [tk_dialog .savedialog "Save" "Save changes to $file?" \
		    question 2 "Cancel" "Don't Save" "OK"]
	    if {$save == 0} {
		# Cancel
		return -code return
	    } elseif {$save == 2} {
		jedit:cmd:save $t
	    }
	}
    }
    
    # Close spiffymulti window
    if {[winfo exists .spiffymulti]} {
	destroy .spiffymulti
    }
}
}
    
#	    if [j:confirm -text "Save changes to $file?" \
#		    -title "Save" -nobutton "Don't Save"] {
#		jedit:cmd:save $t
#


#
# mode:tau:close - Overrides the jedit close function.  This ALWAYS does
#                  the close function instead of quiting via 'exit.  Also,
#                  This function removes the array element created in 
#                  mode:tau:post_read_hook for this file.
#
#              t - the name of the text widget
if {[mode:tau:in_tau]} {
proc mode:tau:close {t} {
    global jedit_file_w \
	    jedit_file_open \
	    depfile \
	    JEDIT_WINDOW_COUNT
  
    set mode [jedit:get_mode $t]
    set filename [jedit:get_filename $t]

    if {![info exists jedit_file_open($filename)] && \
	    [string index $filename 0] == "/"} {
	set filename [RelativizePath $depfile(host) $depfile(dir) $filename]
    }

    incr JEDIT_WINDOW_COUNT -1      ;# one fewer window
    set win_state [lindex [$t configure -state] 4]
    destroy [jedit:text_to_top $t]
    
    if {$win_state != "disabled"} {
	if [info exists jedit_file_w($filename)] {
	    unset jedit_file_w($filename)
	    set jedit_file_open($filename) 0
	}
    }
}
}


#
# mode:tau:quit - Calls the tau mode close function, overrideding the
#                 jedit quit function.
#
#             t - the name of the text widget
if {[mode:tau:in_tau]} {
proc mode:tau:quit {t} {
    set mode [jedit:get_mode $t]
    if {[info procs mode:$mode:pre_close_hook] != {}} {
      mode:$mode:pre_close_hook $t
    }
    mode:tau:close $t
}
}


#
# mode:tau:select_func - Call tau's global selection fuction that option 
#                        is selected in the options menu, otherwise call
#                        the local selection fucntion.
#
#             progfile - depfile containing the tag
#                  tag - The tag to select
#                where - invoke selection locally (local) or globablly 
#                        (global)
#
proc mode:tau:select_func {progfile tag {where ""}} {
    global SPIFFY

    if {(($where == "") && $SPIFFY(global)) || ($where == "global")} {
	PM_GlobalSelect $progfile global_selectFuncTag $tag
    } else {
	selectFuncTag $progfile $tag
    }
}


#
# mode:tau:show_func - Call tau's global show fuction
#
#             progfile - depfile containing the tag
#                  tag - The tag to select
#                where - (IGNORED)
#
proc mode:tau:show_func {progfile tag {where ""}} {
    PM_GlobalSelect $progfile global_showFuncTag $tag
}


#
# mode:tau:select_class - Call tau's global selection fuction that option 
#                         is selected in the options menu, otherwise call
#                         the local selection fucntion.
#
#              progfile - depfile containing the tag#
#                   tag - The tag to select
#                 where - invoke selection locally (local) or globablly 
#                         (global)
#
proc mode:tau:select_class {progfile tag {where ""}} {
    global SPIFFY

    if {(($where == "") && $SPIFFY(global)) || ($where == "global")} {
	PM_GlobalSelect $progfile global_selectClassTag $tag
    } else {
	selectClassTag $progfile $tag
    }
}


#
# mode:tau:hyper - make a hypertext jump to the function/class selected
#                  in the editor window.
#
#              t - the editor text widget name
#          where - invoke selection locally (local) or globablly (global)
#
proc mode:tau:hyper {t {where ""}} {
    global matching_funcs
    global matching_classes

    if {![mode:tau:in_tau]} {
	j:alert -title Alert -text \
		"The TAU 'dep' file is not loaded.\n\
		The HYPER function is only available\n\
		TAU environment."
	return
    }

    # Get the name selected or near
    set name [j:selection_if_any]
    if {$name == ""} {
	while { ([string trim [$t get "insert wordstart" "insert wordend"]] \
		== "") && \
		!([$t compare insert == end]) } {
	    $t mark set insert "insert + 1 char"
	}
	set name [$t get "insert wordstart" "insert wordend"]
	$t tag add sel "insert wordstart" "insert wordend"
    }

    # Find matches in broswer db
    set matching_funcs   [Bdb_QueryFunc  $name]
    set matching_classes [Bdb_QueryClass $name]
    set num_funcs   [llength $matching_funcs]
    set num_classes [llength $matching_classes]
    
    if {[expr $num_funcs + $num_classes] > 1} {
	mode:tau:multi $t $name select $num_funcs $num_classes $where

    } elseif {[expr $num_funcs + $num_classes] == 1} {
	if {$num_funcs} {
	    set progfile [lindex [lindex $matching_funcs 0] 0]
	    set tag      [lindex [lindex $matching_funcs 0] 1]
	    set SPIFFY(selection) [lindex $matching_funcs 0]
	    mode:tau:select_func $progfile $tag $where
	} else {
	    set progfile [lindex [lindex $matching_classes 0] 0]
	    set tag      [lindex [lindex $matching_classes 0] 1]
	    set SPIFFY(selection) [lindex $matching_classes 0]
	    mode:tau:select_class $progfile $tag $where
	}
    }
}


# mode:tau:multi - create or display a small window for selecting
#                  which function if more than one is appropriate.
#
#                    t - the jedit text widget
#                 name - the function name
#                 what - what action to take, either select or show
#            num_funcs - the number of appropriate functions
#          num_classes - the number of appropriate classes
#                where - invoke selection locally (local) or globablly 
#                        (global)
#
proc mode:tau:multi {t name what num_funcs num_classes {where ""}} {
    global matching_funcs matching_classes \
	    TAU_MODE TAU_GEOMTRY SPIFFY

    set lab .spiffymulti.label
    set lbf .spiffymulti.lbf
    set lb .spiffymulti.lbf.lb
    set sb .spiffymulti.lbf.sb
    set fr .spiffymulti.f
    
    if {![winfo exists .spiffymulti]} {
	toplevel .spiffymulti -width 10c
	wm title .spiffymulti "Definitions"
	wm withdraw .spiffymulti
	
	label $lab -text "Double-click to select:"
	frame $lbf -borderwidth 2 -relief raised
	listbox $lb -yscrollcommand "$sb set" -borderwidth 2 -relief raised \
		-exportselection false
	scrollbar $sb -command "$lb yview"
	frame $fr
	pack $lab $lbf $fr -side top -fill x -fill y	
	pack $lb -side left -fill x -fill y
	pack $sb -side right -fill y
	focus $lb
	button $fr.cancel -text "Cancel" -command mode:tau:multi_cancel
	pack $fr.cancel -fill x
    }

    set sum [expr $num_funcs + $num_classes]
    if {$sum <= 10} {
	$lb configure -width 40 -height $sum
    } else {
	$lb configure -width 40 -height 10
    }

    $lb delete 0 end
    for {set i 0} {$i < $num_funcs} {incr i} {
	set tag      [lindex [lindex $matching_funcs $i] 1]
	set progfile [lindex [lindex $matching_funcs $i] 0]
	set filen    [lindex [lindex $matching_funcs $i] 3]
	if {$filen == "-"} {
	    set filen "SOURCE NOT OBTAINABLE"
	}
	$lb insert end "$i function ($filen)"
	if {$i <= 9} {
	    bind $lb <Key-$i> "mode:tau:multi_key_callback $name $what \
		    $num_funcs $num_classes $i $where"
	}
    }
    if {$what == "select"} {
	for {set i 0} {$i < $num_classes} {incr i} {
	    set tag      [lindex [lindex $matching_classes $i] 1]
	    set progfile [lindex [lindex $matching_classes $i] 0]
	    set filen    [lindex [lindex $matching_classes $i] 3]
	    if {$filen == "-"} {
		set filen "SOURCE NOT OBTAINABLE"
	    }
	    set idx [expr $num_funcs + $i]
	    $lb insert end "$idx class ($filen)"
	    if {$idx <= 9} {
		bind $lb <KeyPress-$idx> "mode:tau:multi_key_callback $name \
			$what $num_funcs $num_classes $idx $where"
	    }
	}
    }
    
    bind $lb <Double-Button-1> \
	    "mode:tau:multi_callback $name $what \
	    $num_funcs $num_classes $where"
    focus $lb

    if {$TAU_MODE(multi_geometry) == ""} {
	regsub -all {\+|x} [wm geometry [jedit:text_to_top $t]] " " parts
	scan $parts "%d %d %d %d" win_w win_h win_x win_y
	set new_geo [format "+%d+%d" [expr $win_x + 150] [expr $win_y + 200]]
	wm geometry .spiffymulti $new_geo
    } else {
	wm geometry .spiffymulti $TAU_MODE(multi_geometry)
    }
    wm deiconify .spiffymulti
}


#
# mode:tau:multi_cancel - cancels the multiselect window
#
proc mode:tau:multi_cancel {} {
    global TAU_MODE

    set TAU_MODE(multi_geometry) [wm geometry .spiffymulti]
    set TAU_MODE(multi_geometry) [string range $TAU_MODE(multi_geometry) \
	    [string first "+" $TAU_MODE(multi_geometry)] end]
    wm withdraw .spiffymulti
}


# 
# mode:tau:multi_callback - execute the global action when the user has
#                           double clicked on a function or class from the
#                           multiselect window.
#
#                 name - the function name
#                 what - what action to take, either select or show
#            num_funcs - the number of appropriate functions
#          num_classes - the number of appropriate classes
#                where - invoke selection locally (local) or globablly 
#                        (global)
#
proc mode:tau:multi_callback {name what num_funcs num_classes {where ""}} {
    global matching_funcs matching_classes \
	    TAU_MODE SPIFFY

    set num [lindex [.spiffymulti.lbf.lb curselection] 0]

    set TAU_MODE(multi_geometry) [wm geometry .spiffymulti]
    set TAU_MODE(multi_geometry) [string range $TAU_MODE(multi_geometry) \
	    [string first "+" $TAU_MODE(multi_geometry)] end]
    wm withdraw .spiffymulti

    if {$num < $num_funcs} {
	set cmd_name mode:tau:${what}_func
	set tag      [lindex [lindex $matching_funcs $num] 1]
	set progfile [lindex [lindex $matching_funcs $num] 0]
	set SPIFFY(selection) [lindex $matching_funcs $num]
	$cmd_name $progfile $tag $where
    } else {
	set cmd_name mode:tau:${what}_class
	set tag [lindex [lindex $matching_classes \
		[expr $num - $num_funcs]] 1]
	set progfile [lindex [lindex $matching_classes \
		[expr $num - $num_funcs]] 0]
	set SPIFFY(selection) [lindex $matching_classes \
		[expr $num - $num_funcs]]
	$cmd_name $progfile $tag $where
    }

}


# 
# mode:tau:multi_key_callback - execute the global action when the user has
#                               selected w/ the keyboard from the
#                               multiselect window.
#
#                 name - the function name
#                 what - what action to take, either select or show
#            num_funcs - the number of appropriate functions
#          num_classes - the number of appropriate classes
#            selection - the number selected
#                where - invoke selection locally (local) or globablly 
#                        (global)
#
proc mode:tau:multi_key_callback \
	{name what num_funcs num_classes selection {where ""}} {
    global matching_funcs matching_classes \
	    TAU_MODE SPIFFY
    
    for {set i 0} {$i <= 9} {incr i} {
	bind .spiffymulti.lbf.lb <KeyPress-$i> ""
    }

    set TAU_MODE(multi_geometry) [wm geometry .spiffymulti]
    set TAU_MODE(multi_geometry) [string range $TAU_MODE(multi_geometry) \
	    [string first "+" $TAU_MODE(multi_geometry)] end]
    wm withdraw .spiffymulti

    if {$selection < $num_funcs} {
	set cmd_name mode:tau:${what}_func
	set tag      [lindex [lindex $matching_funcs $selection] 1]
	set progfile [lindex [lindex $matching_funcs $selection] 0]
	set SPIFFY(selection) [lindex $matching_funcs $selection]
	$cmd_name $progfile $tag $where
    } else {
	set cmd_name mode:tau:${what}_class
	set tag [lindex [lindex $matching_classes \
		[expr $selection - $num_funcs]] 1]
	set progfile [lindex [lindex $matching_classes \
		[expr $selection - $num_funcs]] 0]
	set SPIFFY(selection) [lindex $matching_classes \
		[expr $selection - $num_funcs]]
	$cmd_name $progfile $tag $where
    }
}


#
# mode:tau:func_info - execute the global show function operation on the
#                      selected function name.
#
#                  t - the editor's text widget name
#
proc mode:tau:func_info t {
    global matching_funcs matching_classes

    if {![mode:tau:in_tau]} {
	j:alert -title Alert -text \
		"The TAU 'dep' file is not loaded.\n\
		The FUNC INFO function is only available\n\
		TAU environment."
	return
    }
    
    # Get the name selected or near
    set name [j:selection_if_any]
    if {$name == ""} {
	while { ([string trim [$t get "insert wordstart" "insert wordend"]] \
		== "") && \
		!([$t compare insert == end]) } {
	    $t mark set insert "insert + 1 char"
	}
	set name [$t get "insert wordstart" "insert wordend"]
	$t tag add sel "insert wordstart" "insert wordend"
    }

    # Find matches in broswer db
    set matching_funcs [Bdb_QueryFunc $name]
    set num_funcs [llength $matching_funcs]

    if {$num_funcs > 1} {
	mode:tau:multi $t $name show $num_funcs 0

    } elseif {$num_funcs == 1} {
	set progfile [lindex [lindex $matching_funcs 0] 0]
	set tag      [lindex [lindex $matching_funcs 0] 1]
	mode:tau:show_func $progfile $tag
    }
}



#
# mode:tau:autoindent - adjust indentation based on nesting from jedit's tcl
#                       mode.
#
proc mode:tau:autoindent { t } {
  global TAU_MODE

  set indentlevel 0
  set current [$t get {insert linestart} {insert}]
  set prevline [$t get {insert -1lines linestart} {insert -1lines lineend}]
  set antepenult [$t get {insert -2lines linestart} {insert -2lines lineend}]
  
  set indent ""
  regexp "^  *" $prevline indent
  set indentlevel [string length $indent]
  
  set anteindent ""
  regexp "^  *" $antepenult anteindent
  set antelevel [string length $anteindent]
  
  set close "^\[ \t\]*\}"			;# brace at beginning of line
  if {[regexp $close $prevline]} {
    if {$indentlevel == $antelevel && $indentlevel >= $TAU_MODE(indent)} {
      # change current indentation level:
      incr indentlevel -$TAU_MODE(indent)
      # and adjust previous line's indentation:
      $t delete {insert -1lines linestart} \
        "insert -1lines linestart +$TAU_MODE(indent)chars"
    }
  }
  set comment "\{\[ \t;\]*#\[^\}\]*$"		;# brace followed by comment
  if {[regexp "\{$" $prevline] || [regexp $comment $prevline]} {
    incr indentlevel $TAU_MODE(indent)
  }
  if {[string match {*[\]} $prevline]} {	;# line continued
    if {![string match {*[\]} $antepenult]} {
      incr indentlevel $TAU_MODE(indent)
    }
  } else {
    if {[string match {*[\]} $antepenult]} {
      # last line was a continuation, but this one isn't
      incr indentlevel -$TAU_MODE(indent)
    }
  }
  if {$indentlevel < 0} {set indentlevel 0}
  
  for {set i 0} {$i < $indentlevel} {incr i} {
    $t insert insert " "
  }
}


#
# mode:tau:man_page - Displays the man page for the selection.
#
proc mode:tau:man_page {t} {
    global tkman

    set name [j:selection_if_any]
    if {$name == ""} {
	while { ([string trim [$t get "insert wordstart" "insert wordend"]] \
		== "") && \
		!([$t compare insert == end]) } {
	    $t mark set insert "insert + 1 char"
	}
	set name [$t get "insert wordstart" "insert wordend"]
	$t tag add sel "insert wordstart" "insert wordend"
    }

    # Is tkman running?
    set avail [winfo interps]
    if { [ lsearch -exact $avail $tkman(interp_name) ] == -1 } {
	exec $tkman(path) &
	while { [ lsearch -exact $avail $tkman(interp_name)] == -1 } {
	    after 1000
	    set avail [winfo interps]
	}

	# wait for it to initialize
	for {set ready 0} {!$ready} {after 200} {
	    catch {set ready [send $tkman(interp_name) set manx(init)]}
	}

    } else {
	xsend $tkman(interp_name) "wm deiconify $tkman(win)"
	xsend $tkman(interp_name) "raise $tkman(win)"
    }

    # Is it the right version?
    set tries 100
    while {$tries > 0 && [catch \
	    {set man_version [send $tkman(interp_name) "set manx(version)"]} \
	    ]} {
	after 1000
	incr tries -1
    }
    if {$tries == 0} {
	j:alert -title Alert -text "Waiting for TkMan TIMED OUT."
	return
    }

    if {![string match "$tkman(version)*" $man_version]} {
	j:alert -title Alert -text \
		"The wrong version of TkMan is in use.\n\
		Expecting version $tkman(version)"
	return
    }

    # Display the page
    catch {send $tkman(interp_name) "manShowMan $name"}
}


#
# mode:tau:build - Starts cosy (if not running) and executes a build on
#                  the application being edited.
#
proc mode:tau:build {t} {
    launch cosy .cosy -waitfor
    set avail [winfo interps]
    while { [lsearch -exact $avail cosy] == -1 } {
	set avail [winfo interps]
    }
    xsend cosy ".cosy.mid2.build invoke;.cosy.mid2.make invoke"
}


proc mode:tau:addfile {t} {
    global depfile jedit_file_w jedit_file_open
    
    set file [jedit:get_filename $t]

    if {$file == ""} {
	jedit:cmd:saveas $t
    }
    set file [jedit:get_filename $t]
    set relfile [RelativizePath $depfile(host) $depfile(dir) $file]
    
    set jedit_file_w($relfile) $t
    set jedit_file_open($relfile) 1
    PM_AddFile $file
}

proc mode:tau:remfile {t} {
    global depfile jedit_file_w jedit_file_open

    set file [jedit:get_filename $t]
    PM_RemoveFile $file
}

###########################################################################
#
# Override jedit procedures for TAU mode
#

#
# jfileio.tcl
#

proc j:fileio:read { filename } {
    global depfile FileIO_Error

    if {[string index $filename 0] != "/"} {
	set path "$depfile(dir)/$filename"
    } else {
	set path "$filename"
    }

    set file [FileIO_file_open $depfile(host) $path r]
    if {$file == "FILEIO_ERROR"} {
	j:alert "Couln't read file $filename: $FileIO_Error"
	return ""
    }
    set result [read $file]
    set closed [FileIO_file_close $depfile(host) $file]
    if {$closed == "FILEIO_ERROR"} {
	j:alert "Couln't read file $filename: $FileIO_Error"
	return ""
    } 
  
    return $result
}

proc j:fileio:write { filename text } {
    global depfile FileIO_Error

    if {[string index $filename 0] != "/"} {
	set path "$depfile(dir)/$filename"
    } else {
	set path "$filename"
    }

    set file [FileIO_file_open $depfile(host) $path w]
    if {$file == "FILEIO_ERROR"} {
	j:alert -title Alert -text \
		"Couln't write file $filename: $FileIO_Error"
	return
    }
    puts -nonewline $file $text
    set closed [FileIO_file_close $depfile(host) $file]
    if {$closed == "FILEIO_ERROR"} {
	j:alert -title Alert -text \
		"Couln't read write $filename: $FileIO_Error"
    }
}


#
# jedit_modes.tcl
#

proc jedit:guess_mode { f } {
  j:debug "jedit:guess_mode $f"
  global FILE_MODES LINE_MODES depfile
  
  #
  # first, try matching on name
  #
  foreach i $FILE_MODES {
    if [string match [lindex $i 0] $f] {
      return [lindex $i 1]
    }
  }
  #
  # then, check first line (might be a script)
  #
  if {[FileIO_file_exists $depfile(host) $f]} {
      set line1 ""			;# in case file not readable
      catch {
	  set file [FileIO_file_open $depfile(host) $f r]
	  if {$file == "FILEIO_ERROR"} {
	      return plain
	  }
	  set line1 [read $file 80] ;# unix sees 32 chars, but be generous
	  set closed [FileIO_file_close $depfile(host) $file]
	  if {$closed == "FILEIO_ERROR"} {
	      return plain
	  }
      }
      foreach i $LINE_MODES {
	  if [string match [lindex $i 0] $line1] {
	      return [lindex $i 1]
	  }
      }
  }
  #
  # no matches - just use `plain'
  #
  return plain
}

