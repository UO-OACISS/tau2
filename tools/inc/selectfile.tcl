#
# file selector box
# -----------------

if ![info exists FileIO_Error] {
    source $TAUDIR/inc/fileio.tcl
}

set DEBUG_SET 0


proc DEBUG {} {
    global DEBUG_SET

    return $DEBUG_SET
}

proc fileselect {w dir file pattern title showdot {pmcod 0} {host "" } } {

    # Added parameter pmcod (== "Project Manager Create Or Die")
    # to accomodate the need for new project creation in the taud.
    # If this variable is set (following boolean rules, 0 or non-zero)
    # then a "New Project" button will be created.
    # SAS, 6/7/96
    # Also added the host parameter, to check for remote operations.
    # SAS, 6/18/96
    #
    # Remodified 6/28 SAS
    # The fileselct procedure is used throughout the tool structure, so it
    # needs to be more generic. Extended fileselect to contain current 
    # information.
    # The fileselect array consists of the fields
    # fileselect(button) -> used to signal completion, error conditions
    # fileselect(selection) -> contains the final selection of the file (full path)
    # fileselect(path) -> contains the current directory
    # fileselect(file) -> contains the [file tail x] of the selected file x (from file listbox)
    # fileselect(showdot) -> a boolean as to whether to show dotted files (e.g. .cshrc)
    # fileselect(pattern) -> a regexp pattern to match for file endings

    global fileselect languages

    # Added 6/18. to allow for PM operations. SAS
    if $pmcod {
	global PM_Globals
	global languages
	global depfile
	# To have access to the window name everywhere, I'm going to put it
	# in the global array. If this becomes a problem, there may be a need
	# for some testing. . .
	set fileselect(wvar) $w
    }

    if [string match $host ""] {
	set fileselect(host) $depfile(host)
    } else {
	set fileselect(host) $host
    }

    # local functions
    # ---------------

    proc settabstops {toplevel list} {
	for {set i 0} {$i < [llength $list]} {incr i} {
	    set tab1 [lindex $list $i]
	    set tab2 [lindex $list [expr ( $i + 1 ) % [llength $list]]]
	    bindtags $tab1 [list $toplevel [winfo class $tab1] $tab1]
	    bind $tab1 <Tab> "focus $tab2"
	    bind $tab2 <Shift-Tab> "focus $tab1"
	}
    }


    proc modifyRootAndHostGUI {} {
	global fileselect depfile

	proc mRAHGUILogic {} {
	    global fileselect depfile Local_Hostname REMSH PM_Globals
	    
	    if {$fileselect(hostTemp) != $fileselect(host)} {
		# The host got modified. Make sure that that host exists,
		# and that the user can connect to the host.
		set remote [open "| $REMSH $fileselect(hostTemp) -n ls" {CREAT RDONLY NONBLOCK}]
		if [DEBUG] {puts "Made it into the ls proc."}
		if [DEBUG] {puts "fileselect(hostTemp) == $fileselect(hostTemp)"}
		if [DEBUG] {puts "and remote (the file) == $remote"}
		set result [read $remote]
		if [DEBUG] {puts "result == $result"}
		if {[set errcode [catch "close $remote" errmsg]] != 0} {
		    showError "Error in contacting $fileselect(hostTemp): Try again."
		    return
		}
		set remote [open "| $REMSH $fileselect(hostTemp) -n pwd" {CREAT RDONLY NONBLOCK}]
		set result [gets $remote]
		if [DEBUG] {puts "pwd on the remote host is $result"}
		if {[set errcode [catch "close $remote" errmsg]] != 0} {
		    showError "Error in getting working directory on $fileselect(hostTemp)"
		    return
		}
		set fileselect(pathTemp) $result
	    }
	    if {$fileselect(rootTemp) != $fileselect(root)} {
		if {($fileselect(hostTemp) == "localhost") || \
			($fileselect(hostTemp) == $Local_Hostname)} {
		    if {!([file exists $fileselect(rootTemp)] && \
			      [file isdirectory $fileselect(rootTemp)])} {
			showError "$fileselect(rootTemp) is invalid on $fileselect(hostTemp)"
			return
		    }
		    if {![file exists "$fileselect(rootTemp)/utils/archfind"]} {
			showError "$fileselect(rootTemp) does not appear to be a Sage directory."
			return
		    }
		} else {
		    set tempRootDir [file dirname $fileselect(rootTemp)]
		    set tempDir [file tail $fileselect(rootTemp)]
		    set remote [open "| $REMSH $fileselect(hostTemp) -n \"ls -l $tempRootDir\"" r]
		    set result [read $remote]
		    if {[set errcode [catch "close $remote" errmsg]] != 0 } {
			showError "Error in accessing $fileselect(rootTemp) on $fileselect(hostTemp)"
			return
		    }
		    set result [split $result \n]
		    foreach lsline $result {
			if { [lsearch $lsline $tempDir] != -1 } {
			    set first [lindex $lsline 0]
			    if { [string index $first 0] != "d" } {
				if { [string index $first 0] != "l" } {
				    showError "$fileselect(rootTemp) is not a valid directory on $fileselect(hostTemp)"
				    return
				}
			    }
			}
		    }; # End foreach
		    # So we ostensibly have a valid directory. Now to check for archfind,
		    # to be sure that it's a valid *Sage* directory. . .
		    set remote [open "| $REMSH $fileselect(hostTemp) -n \"ls $filselect(rootTemp)/utils/archfind\"" r]
		    set result [read $remote]
		    if {[set errcode [catch "close $remote" errmsg]] != 0 } {
			showError "Error in accessing\n$fileselect(rootTemp)/utils/archfind\non $fileselect(hostTemp)\nNot a valid Sage distribution."
			return
		    }
		    
		}; # End else
	    }
	    # All is cool. Remove the GUI, set the variables, and return to our regularly 
	    # scheduled program.
	    if {$fileselect(host) != $fileselect(hostTemp)} {
		set fileselect(host) $fileselect(hostTemp)
		set pathTemp [lindex [split $fileselect(pathTemp) \n] 0]
		set fileselect(path) $pathTemp
		if [DEBUG] {puts "At the end of mRAHGUILogic, fileselect(path) == $fileselect(path)"}
		set fileselect(selection) $fileselect(path)
		readdir $fileselect(wvar) "" "" .pmf
	    }
	    set fileselect(root) $fileselect(rootTemp)
	    set depfile(root) $fileselect(root)
	    set PM_Globals(root) $fileselect(root)
	    FileIO_INIT $PM_Globals(root)
	    set PM_Globals(host) $fileselect(host)

	    # This next line gets the architecture of the remote machine, using rsh.
	    # So far, it only works on a handful of machines. Big Bug, but
	    # necessary for now. Works on the SGI Trio at the U of O, and Fullsail at the U of O.
	    set tempArch [FileIO_exec $fileselect(host) $fileselect(root)/utils/archfind]
	    set depfile(arch) $tempArch
	    set PM_Globals(arch) $tempArch
	    catch {focus $fileselect(focus)}
	    destroy .modRootandHost
	} ; #End mRAHGUILogic
	
	toplevel .modRootandHost
	wm title .modRootandHost "Modify Root and Host"
	set top [frame .modRootandHost.top]
	set middle [frame .modRootandHost.middle]
	set bottom [frame .modRootandHost.bottom]
	
	label $top.lab \
	    -text "Host:" \
	    -anchor e \
	    -width 15
	entry $top.ent \
	    -textvariable fileselect(hostTemp) \
	    -width 35
	pack $top.lab $top.ent \
	    -side left \
	    -anchor nw \
	    -fill x \
	    -expand 1
	
	label $middle.lab \
	    -text TAUROOT \
	    -width 15 \
	    -anchor e
	entry $middle.ent \
	    -textvariable fileselect(rootTemp) \
	    -width 35
	pack $middle.lab $middle.ent \
	    -side left \
	    -anchor nw \
	    -fill x \
	    -expand 1
	
	# Init the temp vars with the current state
	set fileselect(hostTemp) $fileselect(host)
	set fileselect(rootTemp) $fileselect(root)
	
	button $bottom.enter \
	    -text Enter \
	    -command mRAHGUILogic
	button $bottom.cancel \
	    -text Cancel \
	    -command {grab release .modRootandHost; \
			  focus $fileselect(focus); \
			  destroy .modRootandHost}
	pack $bottom.enter \
	    -side left \
	    -anchor nw
	pack $bottom.cancel \
	    -side right \
	    -anchor ne
	
	pack $top $middle $bottom \
	    -expand 1 \
	    -fill x
	set fileselect(focus) [focus]
	grab .modRootandHost
    }


    proc dirnormalize {dir} {
	
	# The filesection container array
	global fileselect

	if [string match $dir ""] {
	    set dir "."
	}
	if [DEBUG] {puts "dir == $dir"}
	if [string match */ $dir] {
	    set dir [string range $dir 0 [expr [string last / $dir] - 1]]
	    if [DEBUG] {puts "dir matched */"}
	} else {
	    if [DEBUG] {puts "There was no terminal / on $dir."}
	}
	set list [split $dir "/"]
	
	foreach i $list {
	    switch $i {
		""      -
		"."     { }
		".."	{
		    if ![string match $fileselect(path) "/"] {
			set temp [split $fileselect(path) /]
			set temp2 ""
			for {set i 1} {$i < [expr [llength $temp] -2]} {incr i} {
			    set temp2 $temp2/[lindex $temp $i]
			}
			set fileselect(path) $temp2/
		    }
		}
		default {
		    set fileselect(path) $fileselect(path)$i/ 
		}
	    }
	}
	if ![string match */ $fileselect(path)] {
	    set fileselect(path) $fileselect(path)/
	}
	return $fileselect(path)
    }
    
    proc readdir {w dir file pattern} {
	global fileselect PM_Globals
	
	if {"$dir/" != $fileselect(path)} {
	    set dir [dirnormalize $dir]
	} else {
	    if [DEBUG] {puts "dir and fileselect(path) were the same."}
	    set dir "$dir/"
	}
	set temp [FileIO_dir_exists $fileselect(host) $dir]
	if [string match $temp "FILEIO_ERROR"] {
	    bell
	    if [DEBUG] {}
	    return
	} else {
	    if !$temp {
		bell 
		return
	    }
	}

	if [string match $pattern ""] { set pattern "*" }

	set fileselect(path) $dir
	set fileselect(file) $file
	set fileselect(pattern) $pattern
	set fileselect(selection) $dir$file
	$w.filter.entry delete 0 end
	$w.filter.entry insert 0 "$dir$pattern"
	
	# Here we need a selction based on the host, to allow the FileIO routines to do
	# the right thing. Just a matter of some single quotes. . .
	if [string match $fileselect(host) "localhost"] {
	    set file_list [lsort [FileIO_ls $fileselect(host) \
				      "-Fa $fileselect(path)"]]
	} else {
	    set file_list [lsort \
			   [FileIO_ls $fileselect(host) \
				"'-Fa $fileselect(path)'"]]
	}
	$w.lists.frame1.list delete 0 end
	foreach i $file_list {
	    if [string match */ $i] { 
		set i [string range $i 0 [expr [string last / $i] - 1]]
		$w.lists.frame1.list insert end $i 
	    }
	}
	if {[$w.lists.frame1.list size] > 0} {
	    $w.lists.frame1.list xview \
		    [string last "/" [$w.lists.frame1.list get 0]]
	}
	set file_list2 [list]
	foreach item $file_list {
	    if ![string match */ $item] {
		lappend file_list2 [file tail $item]
	    }
	}
	set file_list [list]
	if $fileselect(showdot) {
	    foreach item $file_list2 {
		if [string match *$pattern $item] {
		    lappend file_list $item
		}
	    }
	} else {
	    foreach item $file_list2 {
		if {[string match *$pattern $item] && ![string match .* $item]} {
		    lappend file_list $item
		}
	    }
	}
	$w.lists.frame2.list delete 0 end
	foreach i $file_list {
	    $w.lists.frame2.list insert end [file tail $i]
	}
	if {[$w.lists.frame2.list size] > 0} {
	    $w.lists.frame2.list xview \
		[string last "/" [$w.lists.frame2.list get 0]]
	}
	if [string match *.pmf $pattern] {
	    # So if you're randomly searching for files that have a .pmf
	    # suffix, then you'll be unable to select them, if you've filtered 
	    # for them. . . Not a big bug. A feature, rather.
	    if ![llength $file_list] {
		$w.buttons.ok config -state disabled
	    } else {
		$w.buttons.ok config -state normal
	    }
	}
    }


    # event actions
    # -------------

    proc cmd_showdot {w} {
	global fileselect

	readdir $w $fileselect(path) $fileselect(file) $fileselect(pattern)
    }

    proc cmd_filesel_filter {w} {
	set dir [file dirname [$w.filter.entry get]]
	set pattern [file tail [$w.filter.entry get]]
	readdir $w $dir "" $pattern
    }

    proc cmd_filesel_left_dblb1 {w list} {
	global fileselect

	if {[$list size] > 0 } {
	    readdir $w "[$list get [$list curselection]]/" "" \
		    $fileselect(pattern)
	}
    }

    proc cmd_filesel_right_b1 {w list pmcod} {
	global fileselect

	if {[$list size] > 0} {
	    set fileselect(file) [$list get [$list curselection]]
	    set fileselect(selection) "$fileselect(path)$fileselect(file)"
	    if {[string match *.pmf $fileselect(selection)] && \
		    $pmcod } {
		set tempfile [FileIO_file_open \
				  $fileselect(host) $fileselect(selection)]
		gets $tempfile temproot
		gets $tempfile temproot
		FileIO_file_close $fileselect(host) $tempfile
		set fileselect(root) $temproot
		$w.buttons.create conf -state disabled
		$w.buttons.ok conf -state active
	    }
	}
    }

    proc cmd_filesel_right_dblb1 {w list} {
	if {[$list size] > 0} {
	    $w.buttons.ok flash
	    $w.buttons.ok invoke
	}
    }


    proc FS_EntryChecker {w} {
	global fileselect

	set isFile [FileIO_file_exists $fileselect(host) $fileselect(selection)]
	if $isFile {
	    $w.buttons.ok conf -state active
	    $w.buttons.create conf -state disabled
	} else {
	    $w.buttons.ok conf -state disabled
	    $w.buttons.create conf -state active
	}
    }
    
    proc ButtonChecker {w} {
	global fileselect

	if {[string match [$w.buttons.create cget -state] "disabled"] } {
	    $w.buttons.ok flash
	    $w.buttons.ok invoke
	} else {
	    if {![string match "*.pmf" $fileselect(selection)]} {
		set fileselect(selection) "$fileselect(selection).pmf"
	    }
	    $w.buttons.create flash
	    $w.buttons.create invoke
	}
    }


    # GUI
    # ---

    catch {destroy $w}
    toplevel $w
    wm protocol $w WM_DELETE_WINDOW "$w.buttons.cancel invoke;"
    wm transient $w [winfo toplevel [winfo parent $w]]
    wm title $w $title

    frame $w.title -relief raised -borderwidth 1
    frame $w.top -relief raised -borderwidth 1
    frame $w.bottom -relief raised -borderwidth 1
    pack $w.title $w.top $w.bottom -fill both -expand 1

    label $w.title.label -text $title
    pack $w.title.label -in $w.title -expand 1 -pady 5


    frame $w.filter
    label $w.filter.label -text "Filter:" -anchor w
    checkbutton $w.filter.showdot -text "Show Dot Files" \
	    -command "cmd_showdot $w" -variable fileselect(showdot)
    entry $w.filter.entry
    pack $w.filter.entry -side bottom -fill x
    pack $w.filter.label -side left
    pack $w.filter.showdot -side right

    # Added 6/17. SAS
    # This addition is to add the host to the interface, if the selector
    # is for a PM-related function.

    if $pmcod {
	frame $w.host
	frame $w.host.f1
	label $w.host.f1.hostlabel \
		-text "Host:" \
		-anchor w 
	label $w.host.f1.hostentry \
	    -width 35 \
	    -textvariable fileselect(host)\
	    -relief sunken \
	    -anchor w
	label $w.host.f1.langlabel \
		-text "Language:"
	set fileselect(lang) [lindex $languages 0]
	menubutton $w.host.f1.langmb \
		-width 10 \
		-menu $w.host.f1.langmb.m \
		-textvariable fileselect(lang) \
		-relief raised \
		-border 3
	set temp $w.host.f1.langmb.m
	menu $temp
	foreach i $languages {
	    $temp add radiobutton \
		    -label $i \
		    -variable fileselect(lang) \
		    -value $i 
		    #-command ""
	}
	pack $w.host.f1.hostlabel -side left
	pack $w.host.f1.hostentry -side left -anchor w
	pack $w.host.f1.langmb $w.host.f1.langlabel -side right

	frame $w.host.f2
	label $w.host.f2.srlabel \
		-text "TAUROOT:" \
		-anchor w 
	label $w.host.f2.srentry \
	    -textvariable fileselect(root) \
	    -width 60 \
	    -anchor w \
	    -relief sunken
	pack $w.host.f2.srlabel $w.host.f2.srentry \
		-side left \
		-anchor w \
		-fill x
	frame $w.host.f3
	button $w.host.f3.modHRbut \
	    -text "Modify Host And/Or Root" \
	    -command modifyRootAndHostGUI
	pack $w.host.f3.modHRbut \
	    -side top \
	    -fill x \
	    -expand 1 \
	    -anchor n
	pack $w.host.f1 $w.host.f2 $w.host.f3 \
		-side top \
		-padx 20 \
		-fill x \
		-pady 5 \
		-in $w.host
    }

    # End Additions SAS 6/19

    frame $w.selection
    label $w.selection.label -text "Selection" -anchor w
    entry $w.selection.entry -textvariable fileselect(selection)
    pack $w.selection.label $w.selection.entry -fill x

    frame $w.lists
    frame $w.lists.frame1
    label $w.lists.frame1.label -text "Directories" -anchor w
    scrollbar $w.lists.frame1.yscroll -relief sunken \
	    -command "$w.lists.frame1.list yview"
    scrollbar $w.lists.frame1.xscroll -relief sunken -orient horizontal \
	    -command "$w.lists.frame1.list xview"
    listbox $w.lists.frame1.list -width 30 -height 10 \
	    -yscroll "$w.lists.frame1.yscroll set" \
	    -xscroll "$w.lists.frame1.xscroll set" \
	    -relief sunken -setgrid 1 -selectmode single
    pack $w.lists.frame1.label -side top -fill x
    pack $w.lists.frame1.yscroll -side right -fill y
    pack $w.lists.frame1.xscroll -side bottom -fill x
    pack $w.lists.frame1.list -expand yes -fill y

    frame $w.lists.frame2
    label $w.lists.frame2.label -text "Files" -anchor w
    scrollbar $w.lists.frame2.yscroll -relief sunken \
	    -command "$w.lists.frame2.list yview"
    scrollbar $w.lists.frame2.xscroll -relief sunken -orient horizontal \
	    -command "$w.lists.frame2.list xview"
    listbox $w.lists.frame2.list -width 30 -height 10 \
	    -yscroll "$w.lists.frame2.yscroll set" \
	    -xscroll "$w.lists.frame2.xscroll set" \
	    -relief sunken -setgrid 1 -selectmode single
    pack $w.lists.frame2.label -side top -fill x
    pack $w.lists.frame2.yscroll -side right -fill y
    pack $w.lists.frame2.xscroll -side bottom -fill x
    pack $w.lists.frame2.list -expand yes -fill y

    frame $w.lists.fill
    pack $w.lists.frame1 -side left
    pack $w.lists.frame2 -side right
    pack $w.lists.fill -padx 10

    frame $w.buttons
    button $w.buttons.ok -text "Okay" -width 10
    button $w.buttons.filter -text "Filter" -width 10
    button $w.buttons.cancel -text "Cancel" -width 10

    # Added 6/7/96 SAS
    # The Create button: for creating new projects. Only used within
    # the context of the PM in the taud. But why not make it anyway? . . .
    button $w.buttons.create -text "Create" -width 10

    if $pmcod {
	pack $w.buttons.ok $w.buttons.create $w.buttons.filter $w.buttons.cancel \
		-side left \
		-expand 1
	pack $w.filter $w.host $w.lists $w.selection \
		-side top \
		-padx 20 \
		-fill x \
		-pady 5 \
		-in $w.top
	pack $w.buttons \
		-expand 1 \
		-fill both \
		-pady 10 \
		-in $w.bottom
    } else {
	pack $w.buttons.ok $w.buttons.filter $w.buttons.cancel \
		-side left \
		-expand 1
	pack $w.filter $w.lists $w.selection \
		-side top \
		-padx 20 \
		-fill x \
		-pady 5 \
		-in $w.top
	pack $w.buttons \
		-expand 1 \
		-fill both \
		-pady 10 \
		-in $w.bottom
    }
    


    # event bindings
    # --------------

    $w.buttons.ok config -command "set fileselect(button) 0"
    $w.buttons.cancel config -command "set fileselect(button) 2"
    $w.buttons.filter config -command "cmd_filesel_filter $w"

    # Added 6/7/96 SAS
    # An interface to the file creation routine.
    $w.buttons.create conf -command "cmd_create_pmf $w"

    bind $w.lists.frame1.list <Double-Button-1> "cmd_filesel_left_dblb1 $w %W"
    bind $w.lists.frame1.list <Double-space> "cmd_filesel_left_dblb1 $w %W"
    bind $w.lists.frame2.list <Button-1> "cmd_filesel_right_b1 $w %W $pmcod"
    bind $w.lists.frame2.list <space> "cmd_filesel_right_b1 $w %W"
    bind $w.lists.frame2.list <Double-Button-1> "cmd_filesel_right_dblb1 $w %W"
    bind $w.lists.frame2.list <Double-space> "cmd_filesel_right_dblb1 $w %W"

    bind $w.filter.entry <Return> \
	    "$w.buttons.filter flash; $w.buttons.filter invoke"
    if $pmcod {
	bind $w.selection.entry <Any-KeyRelease> "FS_EntryChecker $w"
	bind $w.selection.entry <Return> \
	    "ButtonChecker $w"
    } else {
    bind $w.selection.entry <Return> \
	"$w.buttons.ok flash;\
	    $w.buttons.ok invoke"
    }
    bind $w <Escape> "$w.buttons.cancel flash; $w.buttons.cancel invoke"

    settabstops $w [list $w.filter.showdot $w.filter.entry \
	    $w.lists.frame1.list $w.lists.frame2.list $w.selection.entry \
	    $w.buttons.ok $w.buttons.filter $w.buttons.cancel]


    # initialization
    # --------------

    set fileselect(path) $dir
    set fileselect(file) ""
    set fileselect(pattern) $pattern
    set fileselect(showdot) $showdot
    set fileselect(button) 0
    set fileselect(selection) ""
    
    readdir $w "." $file $pattern
    wm withdraw $w
    update idletasks
    set x [expr [winfo screenwidth $w]/2 - [winfo reqwidth $w]/2 \
	       - [winfo vrootx [winfo parent $w]]]
    set y [expr [winfo screenheight $w]/2 - [winfo reqheight $w]/2 \
            - [winfo vrooty [winfo parent $w]]]
    wm geom $w +$x+$y
    wm deiconify $w
    set old_focus [focus]
    grab $w
    focus $w.selection.entry
    update idletasks
    tkwait variable fileselect(button)
    grab release $w
    catch {focus $old_focus}
    destroy $w
}

proc getFile {str {pattern *} {pmcod 0} {host ""} {file ""} {curdir ""} } {

    # Added parameter pmcod (== "Project Manager Create Or Die")
    # to accomodate the need for new project creation in the taud.
    # If this variable is set (following boolean rules, 0 or non-zero)
    # then a "New Project" button will be created.
    # SAS, 6/7/96
    
    global fileselect

    if $pmcod {
	global PM_Globals
	set fileselect(root) $PM_Globals(root)
    }

    if [string match $host ""] {
	set host "localhost"
    }
    set fileselect(host) $host
    if ![string match $curdir ""] {
	set fileselect(path) $curdir
    } else {
	set fileselect(path) [FileIO_pwd $host]
	if [string match $fileselect(path) "FILEIO_ERROR"] {
	    if [DEBUG] { puts "in getFile: error getting working directory from $host"}
	    return FILEIO_ERROR
	}
    }
    if [string match $file ""] {
	fileselect .fsel $fileselect(path) "" \
	    $pattern $str 0 $pmcod $fileselect(host)
    } else {
	fileselect .fsel $fileselect(path) "[file tail $file]" \
		$pattern $str 0 $pmcod $fileselect(host)
    }
    if {$fileselect(button) == 0} {
	if $pmcod {
	    set PM_Globals(curdir) $fileselect(path)
	    set PM_Globals(host) $fileselect(host)
	}
	return $fileselect(selection)

    } else {
	return ""
    }
}


proc cmd_create_pmf {w} {
    # This procedure added 6/7/96. Interface to the taud project
    # creation routine.
    # Modified 6/18, to allow for remote hosts. Uses global values,
    # stored externally. Ought to allow for better paramter passing.
    
    global fileselect

    if {![string match "*.pmf" $fileselect(selection)]} {
	set fileselect(selection) "$fileselect(selection).pmf"
    }
    update

    if [string match $fileselect(path) $fileselect(selection)] {
	showError "No valid project file selected/created."
	return
    }
    if [FileIO_file_exists $fileselect(host) $fileselect(selection)] {
	showError "[file tail $fileselect(selection)] already exists as a project. Opening [file tail $fileselect(selection)]."
	set fileselect(button) 0
    } else {
	PM__CreateNewProject $fileselect(selection) $fileselect(lang)\
	    $fileselect(host)
	set fileselect(button) 0
    }
}




