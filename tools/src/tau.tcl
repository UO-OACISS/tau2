#*********************************************************************#
#*                TAU/pC++/Sage++  Copyright (C) 1994                *#
#*                        Jerry Manic Saftware                       *#
#*  Indiana University  University of Oregon  University of Rennes   *#
#*********************************************************************#
displayMessage hourglass "Loading $myself..."

source "$TAUDIR/inc/selectfile.tcl"
source "$TAUDIR/inc/depfile.tcl"
source "$TAUDIR/inc/stack.tcl"
source "$TAUDIR/inc/help.tcl"
source "$TAUDIR/inc/pm_interface.tcl"
source "$TAUDIR/inc/tauutil.tcl"


# A debugging framework, for conditional scaffolding.
set DEBUG_SET 0

proc DEBUG {} {
    global DEBUG_SET

    return $DEBUG_SET
}


set selectfunc(show) showFuncTag
set selectfunc(select) global_selectFuncTag

set ${myself}winnr 0
# Global for setting SAGROOT in project options
set srVar ""

#
# loadCgmIfNeeded - check if the depfile needs to be loaded and load it if so.
#        progfile : the depfile to load
#

proc loadCgmIfNeeded {progfile} {
    if [Cgm_IsDepLoaded $progfile] {
	return 1
    } else {
	Cgm_LoadDep $progfile -dumpall
    }
}

#
# showFuncTag: implementation of global feature "showFunction" for this tool
#              show small function information window in form of a table
#    progfile: depfile containing the tag
#         tag: id of function to show
#

proc showFuncTag {progfile tag} {
    global myself
    global ${myself}winnr

    if {[loadCgmIfNeeded $progfile] == "NOT_OK" } {
	showError "No valid depfile for $progfile."
	return
    }

    if {[Cgm_FuncInfo $progfile $tag name] == "NOT_OK" } {
	showError "No info about tag $tag."
	return
    } elseif { ! [winfo exists .$myself.t$tag] } {
	set name  [Cgm_FuncInfo $progfile $tag name]
	set class [Cgm_FuncInfo $progfile $tag class]
	set file  [Cgm_FuncInfo $progfile $tag file]
	set ptype [lindex [Cgm_FuncInfo $progfile $tag type] 0]
	set type  [lindex [Cgm_FuncInfo $progfile $tag type] 1]
	set num   [Cgm_FuncInfo $progfile $tag num]
	set calls [llength [Cgm_FuncInfo $progfile $tag calls]]
	
	if { [lindex $class 0] == "-" } {
	    set class ""
	} elseif { [lindex $class 1] == "-" } {
	    set class [lindex $class 0]
	} else {
	    set class "Collection [lindex $class 0]"
	}

	if { [lindex $file 0] == "0" } {
	    set file ""
	} else {
	    set file "[lindex $file 1]:[lindex $file 0]"
	}

	switch $ptype {
	    seq { set ptype "sequential" }
	    par { set ptype "parallel" }
	}

	switch $type {
	    Mem { set type "Member Function" }
	    MoE { set type "Method of Element" }
	    MoC { set type "Member of Collection" }
	    Ord { set type "Ordinary Function" }
	    Main { set type "Main Function" }
	    Prcs { set type "Fortran-M Process" }
	}

	set w [toplevel .$myself.t$tag]
	wm title $w "Function Info"

	button $w.b -text "close" -command "destroy $w"
	frame $w.l
	frame $w.r
	
	pack $w.b -side bottom -fill x
	pack $w.l $w.r -side left -padx 5 -pady 5
	
	label $w.l1 -text "name:"
	label $w.l2 -text "class:"
	label $w.l3 -text "file:"
	label $w.l4 -text "type:"
	label $w.l5 -text "call sites:"
	label $w.l6 -text "calls:"
	pack $w.l1 $w.l2 $w.l3 $w.l4 $w.l5 $w.l6 -in $w.l -side top -anchor e
	
	label $w.r1 -text $name -relief sunken -anchor w
	label $w.r2 -text $class -relief sunken -anchor w
	label $w.r3 -text $file -relief sunken -anchor w
	label $w.r4 -text "$ptype $type" -relief sunken -anchor w
	label $w.r5 -text $num -relief sunken -anchor w
	label $w.r6 -text $calls -relief sunken -anchor w
	pack $w.r1 $w.r2 $w.r3 $w.r4 $w.r5 $w.r6 -in $w.r -side top -fill x
    } else {
	raise .$myself.t$tag
    }
}

#
# selectFuncTag: implementation of global feature "selectFunction"
#                this tool doesn't provide one
#

proc selectFuncTag {progfile tag} {}

#
# bitmapbutton: button widget which has a text AND a bitmap
#
#          win: pathname for new button
#       bitmap: bitmap to display
#         text: text label to display
#      command: command to execute if button is pressed
#

proc bitmapbutton {win bitmap text command} {
    frame $win
    button $win.b -bitmap "$bitmap" -command "$command" -back white -fore black \
	-highlightthickness 0
    button $win.t -text "$text" -command "$command" -back white -fore black \
	-takefocus 0 -highlightthickness 0
    bind $win.t <Enter> "$win.b configure -state active"
    bind $win.t <Leave> "$win.b configure -state normal"
    bind $win.b <Enter> "$win.t configure -state active"
    bind $win.b <Leave> "$win.t configure -state normal"
    pack $win.b $win.t -side top -fill x
    return $win
}

#
# createWindow: create and display main window for TAU master control
#

proc createWindow {} {
    global myself
    global TAUDIR
    global depfile

    toplevel .$myself
    wm title .$myself "TAU"
    wm iconbitmap .$myself @$TAUDIR/xbm/$myself.xbm
    wm minsize .$myself 655 445

    # -- menubar
    frame .$myself.bar -relief raised -borderwidth 2
    menubutton .$myself.bar.b1 -text File -menu .$myself.bar.b1.m1 -underline 0 
    menu .$myself.bar.b1.m1
    .$myself.bar.b1.m1 add command -label "Load New Project" -underline 5 \
	-command "tau_OpenProject"
    .$myself.bar.b1.m1 add command -label "Copy Project (Backup)" -underline 0 \
	-command "tau_CopyProject"
    .$myself.bar.b1.m1 add command -label "Add Source File" -underline 0 \
	-command "tau_AddFile"
    .$myself.bar.b1.m1 add command -label "Delete Selected Source File" -underline 0 \
	-command "tau_DeleteFile"
    .$myself.bar.b1.m1 add command -label "View/Change Project Options" \
	-underline 0 \
	-command "tau_ProjectOptions"
    .$myself.bar.b1.m1 add separator
    .$myself.bar.b1.m1 add command -label "Exit"  -underline 0 -command "atExit"
    
    menubutton .$myself.bar.b2 -text Appl -menu .$myself.bar.b2.m1 -underline 0 
    menu .$myself.bar.b2.m1

    createToolMenu .$myself.bar.b4
    menubutton .$myself.bar.b3 -text Help -menu .$myself.bar.b3.m1 -underline 0 
    menu .$myself.bar.b3.m1
    .$myself.bar.b3.m1 add command -label "on $myself" -underline 3 \
	-command "showHelp $myself 1-$myself 1"
    .$myself.bar.b3.m1 add separator
    .$myself.bar.b3.m1 add command -label "on menubar" -underline 3 \
	-command "showHelp $myself 1.1-menu 1"
    .$myself.bar.b3.m1 add command -label "on display area" -underline 3 \
	-command "showHelp $myself 1.2-display 1"
    .$myself.bar.b3.m1 add command -label "on command buttons" -underline 3 \
	-command "showHelp $myself 1.3-buttons 1"
    .$myself.bar.b3.m1 add separator
    .$myself.bar.b3.m1 add command -label "on using help" -underline 3 \
	-command "showHelp general 1-help 1"

    pack .$myself.bar.b1 .$myself.bar.b2 -side left -padx 5
    pack .$myself.bar.b3 .$myself.bar.b4 -side right -padx 5
    pack .$myself.bar -side top -fill x

    # -- display area
    frame .$myself.info1
    frame .$myself.info1.left
    label .$myself.info1.l1 -text "host:"
    label .$myself.info1.l2 -text "dir:"
    label .$myself.info1.l3 -text "project:"
    pack .$myself.info1.l1 .$myself.info1.l2 .$myself.info1.l3\
	-side top -anchor e -in .$myself.info1.left
    pack .$myself.info1.left -side left -padx 5

    frame .$myself.info1.right
    label .$myself.info1.r1 \
	-relief sunken \
	-anchor w \
	-textvariable depfile(host)
    label .$myself.info1.r2 \
	-relief sunken \
	-anchor w \
	-textvariable depfile(dir)
    label .$myself.info1.r3 \
	-relief sunken \
	-anchor w \
	-textvariable depfile(project)
    pack .$myself.info1.r1 .$myself.info1.r2 .$myself.info1.r3 \
	-side top -anchor w -in .$myself.info1.right -fill x -expand 1 -ipadx 10
    pack .$myself.info1.right -side right -padx 5 -fill x -expand 1

    frame .$myself.info2
    #Modified 7/9/96, to accomodate new file information. SAS

    set temp [frame .$myself.info2.ext]
    frame $temp.int
    frame $temp.int.top
    label $temp.int.top.filelabel \
	-text File
    label $temp.int.top.langlabel \
	-text Language
    pack $temp.int.top.filelabel \
	-side left \
	-padx 10 \
	-anchor nw 
    pack $temp.int.top.langlabel \
	-side right \
	-padx 20 \
	-anchor ne
    pack $temp.int.top \
	-side top \
	-fill x
    frame $temp.int.bot
   listbox $temp.int.bot.list \
	-width 75 \
	-font -*-courier-bold-r-*-*-12-*-*-*-*-*-*-* \
	-yscrollcommand "$temp.int.sb set"
    pack $temp.int.bot.list \
	-side left \
	-anchor nw \
	-fill y \
	-expand 1
    pack $temp.int.bot \
	-side left \
	-expand 1 \
	-fill y \
	-anchor n
    scrollbar $temp.int.sb \
	-command "$temp.int.bot.list yview" 
    pack $temp.int.sb \
	-side left \
	-fill y \
	-expand 1
    pack $temp.int \
	-side top \
	-anchor n \
	-expand 1 \
	-fill y
    pack $temp \
	-side top \
	-expand 1 \
	-anchor n \
	-fill y

    pack .$myself.info1 -side top -padx 10 -pady 10 -fill x
    pack .$myself.info2 -side top -padx 10 -pady 10 -fill both -expand 1

    # -- command button bar
    frame .$myself.coms
    global TOOLSET
    foreach tool $TOOLSET {
	bitmapbutton .$myself.${tool}-button @${TAUDIR}/xbm/${tool}.xbm $tool \
		"launch $tool .${tool}"
	pack .$myself.${tool}-button \
	-side left -padx 10 -pady 5 -in .$myself.coms
    }
    pack .$myself.coms -side bottom -pady 10

    wm protocol .$myself WM_DELETE_WINDOW atExit
    bind all Q atExit
}

#
# toggleButtons: map/unmap command buttons depending on depfile type
#

proc toggleButtons {} {
    global myself TOOLSET

    set langs [PM_GetProjectLangs]
    foreach tool $TOOLSET {
	if {[Lang_CheckCompatibility $langs $tool]} {
	    pack .$myself.${tool}-button \
		    -side left -padx 10 -pady 5 -in .$myself.coms
	} else {
	    pack forget .$myself.${tool}-button
	}
    }
}

#
# atExit: cleanup procedure; makes sure all launched tools are exited
#

proc atExit {} {
    global myself

    async_send racy exit
    async_send cagey exit
    async_send classy exit
    async_send fancy exit
    async_send cosy exit
    async_send speedy exit
    async_send spiffy exit
    async_send mighty exit
    PM_RemTool $myself
    exit
}

#
# getLocalHost: get hostname of computer
#

proc getLocalHost {} {
    global depfile

    if { $depfile(host) == "localhost" } {
	if [catch {exec uname -n} h] {
	    if { ![catch {exec hostname} h] } {
		set depfile(hostarch) "$h ($depfile(arch))"
	    } else {
		set depfile(hostarch) "localhost"
	    }
	} else {
	    set depfile(hostarch) "$h ($depfile(arch))"
	}
    } else {
	set depfile(hostarch) "$depfile(host) ($depfile(arch))"
    } 
}


proc Tool_AcceptChanges {progfiles flag} {
    if [DEBUG] { puts "In tau: Project Manager says: $progfile has changed ($flag)."}

    switch $flag {

	d {
	    # Delete a file
	    # Remove from GUI
	    SetFileDisplay
	}
	a {
	    # Add a file
	    # Add to GUI
	    SetFileDisplay
	}
	u {
	    # Update a file
	    # Reset GUI
	}
	p {
	    #Modify project information
	    # Reset GUI
	    SetFileDisplay
	}
	e {
	    #This flag is for udating during execution of the project binary. No action needed.
	}
    }
    if [DEBUG] {puts "in tau: returning from Tool_AcceptChanges"}
}

# ------------
# -- main code
# ------------

if {$argc == 0} {
    set parg [pwd]
} elseif {$argc == 1} {
    set parg [lindex $argv 0]
} else {
    puts stderr "usage: $myself \[\[host:\]projFile \| \[host:\]directory\]"
    exit
}

# Init the project manager (taud)
if [DEBUG] {puts "TAU - launching daemon"}
launchTauDaemon -waitfor
if [DEBUG] {puts "TAU - daemon running - initing PM for tool."}
PM_AddTool $myself
PM_AddGlobalSelect $myself { \
				 global_showFuncTag }

# Initialize the project
# Coordinate w/ PM
if [DEBUG] {puts "TAU - opening a project"}
set pm_status [PM_Status]
if [DEBUG] {puts "PM_Status - $pm_status"}
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

    if [DEBUG] {puts "hostarg - $hostarg"}
    if [DEBUG] { puts "patharg - $patharg"}
    set projfile [PM_OpenProject $patharg $hostarg]
    if {$projfile == "NO_PROJECT"} {
	showError "No project opened!  This should work but it doesn't!"
	atExit
    }
    set pm_status [PM_Status]
}

set depfile(project) [lindex $pm_status 0]
set depfile(host)    [lindex $pm_status 1]
set depfile(arch)    [lindex $pm_status 2]
set depfile(root)    [lindex $pm_status 3]
set depfile(dir)     [lindex $pm_status 4]
if {[string match $depfile(dir) "."]} {
    set depfile(dir) [pwd]
}

# Tool Init
if [DEBUG] {puts "TAU - initializing tool GUI"}
getLocalHost
createWindow
set dispHost [PM_GetHost]
set dispDir [PM_GetDir]
SetFileDisplay
toggleButtons
removeMessage
if [DEBUG] {puts "TAU - done"}
