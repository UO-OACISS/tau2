#*********************************************************************#
#*                TAU/pC++/Sage++  Copyright (C) 1994                *#
#*                        Jerry Manic Saftware                       *#
#*  Indiana University  University of Oregon  University of Rennes   *#
#*********************************************************************#
displayMessage hourglass "Loading classy..."

source "$TAUDIR/inc/depfile.tcl"
source "$TAUDIR/inc/printcan.tcl"
source "$TAUDIR/inc/bdbm_utils.tcl"; #To get the maintag information.
# Need fileio.tcl for remote operations
source "$TAUDIR/inc/fileio.tcl"

set ch_maxlevel 0;     # depth of class hierarchy (maximum x coordinate)
set ch_maxpos 0;       # maximum y coordinate
set ch_showtype 1;     # mark collections with "||" ?
set ch_sel_ctag NONE;  # id of currently selected class
set ch_sel_cfile NONE; # progfile of currently selected class
set classy_classes(1) "";#setting up the local storage array SAS

# A debugging framework, for installing scaffolding.
set DEBUG_SET 0


proc DEBUG {} {
    global DEBUG_SET

    return $DEBUG_SET
}

#
# traverseHierarchy: determine graph layout by recursively traversing it
#                    position determined by "level" (x) and "(max)pos" (y)
#
#               sub: Sage++ id of root of subgraph
#             level: current level in the graph (passed down)
#

proc traverseHierarchy {file sub level} {
    global myself classy_classes
    global ch_maxlevel ch_maxpos
    global classy_equivmapping
    
    if {![info exists classy_classes($file,$sub,fold)]} {
	set classy_classes($file,$sub,fold) 0
    }

    set classy_classes($file,$sub,level) $level
    if { $level > $ch_maxlevel } {set ch_maxlevel $level}
    
    if { $classy_classes($file,$sub,fold) } {
	# -- current node is "folded"
	incr ch_maxpos
	set mypos $ch_maxpos
    } else {
	incr level
	set temp_subInfo $classy_classes($file,$sub,subs)
	switch -exact [llength $temp_subInfo] {
	    0 {
		# -- no subclasses
		# -- take next free position
		incr ch_maxpos
		set mypos $ch_maxpos
	    }
	    1 {
		# -- exactly one subclass
		set pf [lindex [lindex $temp_subInfo 0] 0] 
		set c [lindex [lindex $temp_subInfo 0] 1] 
		if [ info exists classy_classes($pf,$c,pos) ] {
		    # -- subclass has already y position elsewhere
		    # -- handle like no subclass case
		    incr ch_maxpos
		    set mypos $ch_maxpos
		} else {
		    # -- subclass has NOT yet y position
		    # -- use the y coordinate of subclass for myself too
		    set mypos [traverseHierarchy $pf $c $level]
		}
	    }
	    default {
		# -- more than one subclass
		# -- position myself in the middle between first and last
		set startpos $ch_maxpos
		foreach subclass $temp_subInfo {
		    set pf [lindex $subclass 0]
		    set c  [lindex $subclass 1]
		    if { ! [info exists classy_classes($pf,$c,pos)] } {
			set endpos [traverseHierarchy $pf $c $level]
		    }
		}
		set mypos [expr $startpos+($endpos-$startpos+1)/2]
	    }
	}
    }

    set classy_classes($file,$sub,pos) $mypos

    return $mypos
}

#
# functions to invoke in class member table
#

set selectclass(show) global_showFuncTag
set selectclass(select) global_selectFuncTag

#
# proc selectClassTag: implementation of global feature "selectClass"
#                      classy displays the selected class in red
#                      and pop up a class member table
#
#                   c: Sage++ id of class to select
#

proc selectClassTag {file c} {
    global classy_classes selectclass classes \
	    ch_sel_ctag ch_sel_cfile myself classy_equivmapping

    
    # -- de-select previously selected class if necessary
    if { $ch_sel_ctag != "NONE" } {
	if [info exists classy_classes($ch_sel_cfile,$ch_sel_ctag,obj)] {
	    .$myself.graph.can itemconfigure \
		    $classy_classes($ch_sel_cfile,$ch_sel_ctag,obj) -fill black
	}
    }
    
    # -- display selected class in red, if it is displayed
    set equivs [split $classy_equivmapping($file,$c) ","]
    set equiv_cfile [lindex $equivs 0]
    set equiv_ctag  [lindex $equivs 1]
    if [info exists classy_classes($equiv_cfile,$equiv_ctag,obj)] {
	.$myself.graph.can itemconfigure \
		$classy_classes($equiv_cfile,$equiv_ctag,obj) -fill red
    }
    set ch_sel_ctag $equiv_ctag
    set ch_sel_cfile $equiv_cfile
    
    # -- pop up class member table, if class has members
    set classy_classes($equiv_cfile,$equiv_ctag,mem) \
	    [Cgm_ClassInfo $equiv_cfile $equiv_ctag mem]
    if {$classy_classes($equiv_cfile,$equiv_ctag,mem) != "CGM_DOESNT_EXIST"} {
	# -- build sorted list of member functions
	set mems [list]
	foreach m $classy_classes($equiv_cfile,$equiv_ctag,mem) {
	    # mems = { {<name> <progfile> <tag> <type>} ...}
	    lappend mems [list \
		    [lindex $m 2] \
		    [lindex $m 0] \
		    [lindex $m 1] \
		    [lindex [Cgm_MemInfo $equiv_cfile [lindex $m 2]] 2]]
	}
	set mems [lsort $mems]
	set selectclass(mem) $mems
	set selectclass(file) $file

	if [winfo exists .member] {
	    # -- there is already a class member window; reset it
	    .member.l1 delete 0 end
	} else {
	    # -- create new class member window
	    toplevel .member
	    wm minsize .member 50 50
	    listbox .member.l1 -relief sunken -width 60 -height 15 \
		-exportselection false -selectmode single \
		-font -misc-fixed-medium-*-*-*-*-140-*-*-*-*-*-*
	    scrollbar .member.s1 -orient vert -relief sunken \
		-command ".member.l1 yview"
	    .member.l1 configure -yscrollcommand ".member.s1 set"
	    
	    frame .member.bottom
	    frame .member.f1 -relief sunken -bd 1
	    button .member.b1 -text "show" -command {
		set s [.member.l1 curselection]
		if { $s != "" } {
		    set type [lindex [lindex $selectclass(mem) $s] 3]
		    if { [string first f $type] == 1 } {
			set progf [lindex [lindex $selectclass(mem) $s] 1]
			set tag [lindex [lindex $selectclass(mem) $s] 2]
			PM_GlobalSelect $progf $selectclass(show) $tag
		    }
		}
	    }
	    pack .member.b1 -in .member.f1 -side left -padx 5 -pady 5
	    pack .member.f1 -in .member.bottom -side left -padx 10 -pady 10
	    button .member.b2 -text "select" -command {
		set s [.member.l1 curselection]
		if { $s != "" } {
		    set type [lindex [lindex $selectclass(mem) $s] 3]
		    if { [string first f $type] == 1 } {
			set tag [lindex [lindex $selectclass(mem) $s] 2]
			set progf [lindex [lindex $selectclass(mem) $s] 1]
			PM_GlobalSelect $progf $selectclass(select) $tag
		    }
		}
	    }
	    pack .member.b2 -in .member.bottom -side left -padx 15 -pady 15
	    button .member.b3 -text "cancel" -command "destroy .member"
	    pack .member.b3 -in .member.bottom -side right -padx 15 -pady 15
	    
	    pack .member.bottom -side bottom -fill x
	    pack .member.s1 -side right -fill y
	    pack .member.l1 -side left  -expand yes -fill both
	    
	    bind .member.l1 <Double-1> ".member.b1 invoke"
	}
	
	# -- display title
	set classy_classes($equiv_cfile,$equiv_ctag,coll) \
		[Cgm_ClassInfo $file $c "coll"]
	if {$classy_classes($equiv_cfile,$equiv_ctag,coll) == \
		"CGM_DOESNT_EXIST"} {
	    showError "Error checking for collection information"
	    return
	}
	set classy_classes($equiv_cfile,$equiv_ctag,name) \
		[Cgm_ClassInfo $file $c "name"]
	if {$classy_classes($equiv_cfile,$equiv_ctag,name) == \
		"CGM_DOESNT_EXIST"} {
	    showError "Error checking for class name information"
	    return
	}
	if { $classy_classes($equiv_cfile,$equiv_ctag,coll) == "-" } {
	    wm title .member "class  $classy_classes($equiv_cfile,$equiv_ctag,name)"
	} else {
	    wm title .member "collection  $classy_classes($equiv_cfile,$equiv_ctag,name)"
	}
	
	# -- display formatted class member list
	foreach m $mems {
	    .member.l1 insert end \
		"[format "%-25s" [lindex $m 0]] [typeDescr [lindex $m 3]]"
	}
    }
}

#
# typeDescr: transform short typedescription into readable string
#
#      type: short form
#

proc typeDescr {type} {
  set t [split $type ""]

  switch -exact [lindex $t 0] {
    o  {set r "protected"}
    u  {set r "public   "}
    i  {set r "private  "}
    e  {set r "element  "}
  }

  switch -exact [lindex $t 4] {
    " "  { switch -exact [lindex $t 1] {
             v  {append r " variable   "}
             f  {append r " function   "}
             c  {append r " constructor"}
           }
         }
    c    {append r " constructor"}
    d    {append r " destructor "}
    o    {append r " operator   "}
  }

  if { [lindex $t 2] == "v" } {append r " virtual"}
  if { [lindex $t 3] == "i" } {append r " inline"}

  return $r
}

#
# displayHierarchy: display class hierarchy by recursively traversing graph
#                   traverseHierarchy has to be called first to compute
#                   positions!
#
#              sub: Sage++ id of root of graph to display
#

proc displayHierarchy {file sub} {
    global myself classy_classes
    global ch_sel_ctag ch_sel_cfile ch_showtype
    
    # -- compute screen coordinates from level/pos computed by traverseHierarchy
    set x [expr 30+$classy_classes($file,$sub,level)*200]
    set y [expr 30+$classy_classes($file,$sub,pos)*30]
    set name [Cgm_ClassInfo $file $sub "name"]
    if {[string match $name "NOT_OK"]} {
	puts "Not name info for tag $sub in $file."
	return
    }
    
    
    # -- mark collections with "||" if necessary
    set temp_collInfo [Cgm_ClassInfo $file $sub "coll"] 
    if { $ch_showtype && $temp_collInfo != "-" } {
	set name "|| $name"
    }
    
    # -- if class selected, display name in red
    if { $sub == $ch_sel_ctag && $file == $ch_sel_cfile} {
	set obj [.$myself.graph.can create text $x $y \
		     -text $name -fill red -anchor w \
		     -font -Adobe-Helvetica-Bold-R-Normal--*-120-*]
    } else {
	set obj [.$myself.graph.can create text $x $y \
		     -text $name -fill black -anchor w \
		     -font -Adobe-Helvetica-Bold-R-Normal--*-120-*]
    }
    .$myself.graph.can bind $obj <Button-1> "toggleClass $file $sub"
    .$myself.graph.can bind $obj <Button-2> "selectClassTag $file $sub"
    .$myself.graph.can bind $obj <Button-3> "PM_GlobalSelect $file global_selectClassTag $sub"
    set classy_classes($file,$sub,obj) $obj
    
    # -- compute center of node
    set bbox [.$myself.graph.can bbox $obj]
    set ex [expr [lindex $bbox 0] + ([lindex $bbox 2] - [lindex $bbox 0]) / 2]
    set ey [expr [lindex $bbox 1] + ([lindex $bbox 3] - [lindex $bbox 1]) / 2]
    
    # This was going in the space below.
    set temp_subInfo $classy_classes($file,$sub,subs)

    # -- if class has subclasses, draw them and connect with lines
    if { ! $classy_classes($file,$sub,fold) } {
	foreach subclass $temp_subInfo {
	    set pf [lindex $subclass 0]
	    set c  [lindex $subclass 1]
	    set pos [displayHierarchy $pf $c]
	    .$myself.graph.can create line $ex $ey [lindex $pos 0] [lindex $pos 1] \
		-arrow last
	}
    }
    
    # -- draw surrounding rectangle
    .$myself.graph.can create rectangle \
	[expr [lindex $bbox 0]-3] [expr [lindex $bbox 1]-3] \
	[expr [lindex $bbox 2]+3] [expr [lindex $bbox 3]+3] \
	-fill white
    
    # -- if node folded, draw double-border rectangle
    if { $classy_classes($file,$sub,fold) && [llength $temp_subInfo] } {
	.$myself.graph.can create rectangle \
	    [expr [lindex $bbox 0]-1] [expr [lindex $bbox 1]-1] \
	    [expr [lindex $bbox 2]+1] [expr [lindex $bbox 3]+1] \
	    -fill white
    }
    
    # -- raise text above last drawn rectangles
    # -- cannot done the other way round, as the size of the rectangle
    # -- depends on the text
    .$myself.graph.can raise $obj
    
    return [list [expr $x-10] $y]
}

#
# toggleClass: toggle class node between "expanded" and "folded"
#
#          pf: progfile of class to toggle
#           c: id of class to toggle
#

proc toggleClass {pf c} {
    global classy_classes myself
    
    set f $classy_classes($pf,$c,fold)
    set classy_classes($pf,$c,fold) [expr 1-$f]
    set hasSubs [llength $classy_classes($pf,$c,subs)]
    
    if { $hasSubs } {
	#KURT - things seem fine without all this jerky scrolling...
	#set oldobj [expr 30+$classy_classes($pf,$c,pos)*30]
	#set oldwin [.$myself.graph.can canvasy 0]
	
	redrawHierarchy
	
	#set maxyscroll [lindex [.$myself.graph.can cget -scrollregion] 3]
	#set newobj [expr 30+$classy_classes($pf,$c,pos)*30]
	#.$myself.graph.can yview moveto \
	#    [expr (($newobj-$oldobj+$oldwin)/$maxyscroll)+1]
    }
}

#
# redrawHierarchy: redraw class hierarchy by deleteing old one,
#                  then redrawing it
#

proc redrawHierarchy {{type partial}} {
    global myself classy_classes classy_baseclasses
    global classes ch_maxlevel ch_maxpos
    global depfile
    
    # -- reset variables
    .$myself.graph.can delete all
    set ch_maxlevel 0
    set ch_maxpos   0
    foreach c [Bdb_GetClasses] {
	set cpfile [lindex $c 1]
	set ctag [lindex $c 2]
	if [info exists classy_classes($cpfile,$ctag,pos)] {
	    unset classy_classes($cpfile,$ctag,pos)
	}
    }

    # -- recompute positions and display new graph
    # -- start with "virtual" root which has id -1 and level -1
    foreach bc $classy_baseclasses {
	traverseHierarchy [lindex $bc 0] [lindex $bc 1] 0
    }
    .$myself.graph.can configure -scrollregion \
	    [list 0 0 [expr 260+$ch_maxlevel*200] [expr 60+$ch_maxpos*30]]
    foreach bc $classy_baseclasses {
	displayHierarchy [lindex $bc 0] [lindex $bc 1]
    }
}

#
# updateDep: switch to another application
#          this is invoked from the TAU master control window
#
#

proc updateDep {} {
    global myself classy_classes classy_equivmapping classy_baseclasses \
	    ch_sel_ctag ch_sel_cfile classy_progfiles

    # -- delete old class database
    set ch_sel_ctag  NONE
    set ch_sel_cfile NONE

    foreach pf $classy_progfiles {
	Cgm_RemoveDep $pf
    }
    if [info exists classy_classes] {
	unset classy_classes
    }
    if [info exists classy_equivmapping] {
	unset classy_equivmapping
    }
    if [info exists classy_baseclasses] {
	unset classy_baseclasses
    }
    set classy_classes(1) ""
    set classy_classes(maintag) [Bdb_GetMaintag] 

    initialize
    redrawHierarchy
    
    # -- adjust "expand" menu list
    .$myself.bar.b2.m1.1 delete 0 last
    expandMenu
}

#
# createWindow: create and display main class hierarchy window
#

proc createWindow {} {
    global myself TAUDIR classy_classes
    
    toplevel .$myself
    wm title .$myself "CLASSY"
    wm iconbitmap .$myself @$TAUDIR/xbm/$myself.xbm
    wm minsize .$myself 50 50
    
    frame .$myself.graph
    
    # -- configure graph area
    canvas .$myself.graph.can -width 600 -height 350 -background white
    bind .$myself.graph.can <2> ".$myself.graph.can scan mark %x %y"
    bind .$myself.graph.can <B2-Motion> ".$myself.graph.can scan dragto %x %y"
    
    frame .$myself.graph.f1
    
    scrollbar .$myself.graph.hscroll -orient horiz -relief sunken \
	-command ".$myself.graph.can xview"
    .$myself.graph.can configure -xscrollcommand ".$myself.graph.hscroll set"
    
    frame .$myself.graph.f2 -width 20 -height 20
    
    pack .$myself.graph.hscroll -in .$myself.graph.f1 -side left \
	-expand yes -fill x
    pack .$myself.graph.f2      -in .$myself.graph.f1 -side right
    
    scrollbar .$myself.graph.vscroll -orient vert -relief sunken \
	-command ".$myself.graph.can yview"
    .$myself.graph.can configure -yscrollcommand ".$myself.graph.vscroll set"
    
    pack .$myself.graph.f1      -side bottom -fill x
    pack .$myself.graph.vscroll -side right  -fill y
    pack .$myself.graph.can     -side top -padx 15 -pady 15 -fill both -expand yes
    
    # -- configure menu bar
    frame .$myself.bar -relief raised -borderwidth 2
    
    menubutton .$myself.bar.b1 -text File -menu .$myself.bar.b1.m1 -underline 0
    menu .$myself.bar.b1.m1
    .$myself.bar.b1.m1 add command -label "Print graph" -underline 0 \
	-command "printCanvas .$myself.graph.can $myself"
    .$myself.bar.b1.m1 add separator
    .$myself.bar.b1.m1 add command -label "Exit"  -underline 0 -command "exit"
    
    menubutton .$myself.bar.b2 -text View -menu .$myself.bar.b2.m1 -underline 0
    menu .$myself.bar.b2.m1
    .$myself.bar.b2.m1 add checkbutton -label "Show type of class" \
	-underline 0 -variable ch_showtype -onvalue 1 -offvalue 0 \
	-command "redrawHierarchy"
    .$myself.bar.b2.m1 add separator
    
    .$myself.bar.b2.m1 add cascade -label "Expand" -underline 0 \
	-menu .$myself.bar.b2.m1.1
    menu .$myself.bar.b2.m1.1
    
    menubutton .$myself.bar.b3 -text Help -menu .$myself.bar.b3.m1 -underline 0
    menu .$myself.bar.b3.m1
    .$myself.bar.b3.m1 add command -label "on $myself" -underline 3 \
	-command "xsend tau \[list showHelp $myself 1-$myself 1\]"
    .$myself.bar.b3.m1 add separator
    .$myself.bar.b3.m1 add command -label "on menubar" -underline 3 \
	-command "xsend tau \[list showHelp $myself 1.1-menu 1\]"
    .$myself.bar.b3.m1 add command -label "on display area" -underline 3 \
	-command "xsend tau \[list showHelp $myself 1.2-display 1\]"
    .$myself.bar.b3.m1 add separator
    .$myself.bar.b3.m1 add command -label "on using help" -underline 3 \
	-command "xsend tau {showHelp general 1-help 1}"
    
    createToolMenu .$myself.bar.b4
    
    pack .$myself.bar.b1 .$myself.bar.b2 -side left -padx 5
    pack .$myself.bar.b3 .$myself.bar.b4 -side right -padx 5
    
    pack .$myself.bar   -side top -fill x
    pack .$myself.graph -side left -padx 15 -pady 15 -fill both -expand yes

    wm protocol .$myself WM_DELETE_WINDOW exit
}

#
# expandMenu: update "expand" menu, so that it has an entry for each
#             level of the callgraph
#

proc expandMenu {} {
    global myself ch_maxlevel ch_exp_level
    
    set ch_exp_level $ch_maxlevel
    for {set i 0} {$i<=$ch_maxlevel} {incr i} {
	.$myself.bar.b2.m1.1 add radiobutton -label "upto level $i" -underline 11 \
	    -variable ch_exp_level -value $i \
	    -command "expandHierarchy $i"
    }
}

#
# expandHierarchy: expand class hierarchy after user selected level through
#                  the "expand" menu
#
#           level: level up to which graph has to be expanded
#

proc expandHierarchy {level} {
    global classy_classes

    set allclasses [Bdb_GetClasses]
    foreach t $allclasses {
	set pf  [lindex $t 1]
	set tag [lindex $t 2]
	if { $classy_classes($pf,$tag,level) < $level } {
	    set classy_classes($pf,$tag,fold) 0
	} elseif { $classy_classes($pf,$tag,level) == $level } {
	    set classy_classes($pf,$tag,fold) 1
	}
    }
    redrawHierarchy
}

proc Tool_AcceptChanges {progfiles flag} {
    global depfile classy_classes classy_progfiles myself

    switch $flag {

	d {
	    # Delete a file
	    # Since the file that has the main procedure in it may have
	    # been the file deleted, we need to reinitialize classy with the 
	    # updated file list. . .
	    set classy_classes(maintag) [Bdb_GetMaintag]
	    if {[llength $classy_classes(maintag)] == 1} {
		if {[string match $classy_classes(maintag) "BDBM_FAILED"] \
			|| !$classy_classes(maintag)} {
		    showError "No project loaded, or incomplete compilation"
		    return
		}
	    }
	    updateDep
	}


	a {
	    # Add a file
	    # If classy was already running, then we already have a maintag function. However,
	    # we need to check for incomplete compilation of the new file. . .
	    updateDep
	}


	u {
	    # Update a file
	    updateDep 
	}
	
	p {
	    #Modify project information. 
	    Cgm_RemoveAllDeps
	    set temp [PM_Status]
	    if {![string match $temp "UNDEFINED"]} {
		set depfile(project) [lindex $temp 0]
		set depfile(host) [lindex $temp 1]
		set depfile(arch) [lindex $temp 2]
		set depfile(root) [lindex $temp 3]
       		set depfile(dir) [lindex $temp 4]
	    } else {
		showError "There is no project to modify."
	    }
	    # Check for language-tool compatibility
	    if {![Lang_CheckCompatibility [PM_GetProjectLangs] $myself]} {
		showError "$myself is not compatible with the project language(s)."
		exit
	    }
	    set classy_progfiles [PM_GetFiles]
	    updateDep
	}

	e {
	    #This is a flag for updating during execution. No action is needed here.
	}
    }
}


#
# exit - TAU exit function that communicates the event to other tau tools.
#
rename exit exit.old
proc exit {{status 0}} {
    # Taken and modified from Spiffy SAS 7/31/96
    global myself

    PM_RemGlobalSelect $myself { global_selectClassTag }
    PM_RemTool $myself
    exit.old $status
}

#
# initialize the tool - call with first_time 1 for initial setup
#
proc initialize {{first_time 0}} {
    global classy_classes classy_progfiles

    # -- Check that the BDBM is responding correctly (the project is compiled)
    set classy_classes(maintag) [Bdb_GetMaintag]
    if {[llength $classy_classes(maintag)] == 1} {
	if {[string match $classy_classes(maintag) "BDBM_FAILED"] \
		|| !$classy_classes(maintag)} {
	    showError "No project loaded, or incomplete compilation"
	    exit
	}
    }

    # -- Get the set of classes and initialize them
    foreach c [Bdb_GetClasses] {
	set cname [lindex $c 0]
	set cpfile [lindex $c 1]
	set ctag [lindex $c 2]
	set classy_classes($cpfile,$ctag,fold) 0
    }

    # -- Get the set of progfiles 
    set classy_progfiles [PM_GetFiles]
    foreach pf $classy_progfiles {
	set classy_classes($pf,-1,fold) 0

	# -- read new data
	if { [Cgm_LoadDep $pf "-dumpch"] \
		== "NOT_OK" } {
	    puts      "Could not load browsing infor for: $pf."
	    showError "Could not load browsing infor for: $pf."
	    exit
	}
    }

    joinEquivClasses
}

# -- Make the class hierarchy consistent accross multiple files:
# -- Two classes are equivalent iff they have the same name and file
# -- location.
# -- Look for equivalent classes and join over their subclasses
# -- (ie, create classy_classes(<progfile>,<tag>,subs)
proc joinEquivClasses {} {
    global classy_progfiles classy_baseclasses classy_equivmapping classy_hash

    set classy_baseclasses {}
    set classy_hash(__NULL__) 1
    foreach pf $classy_progfiles {
	foreach c [Cgm_ClassInfo $pf -1 subs] {
	    joinEquivHelper $pf $c -1
	}
    }
    unset classy_hash
}

proc joinEquivHelper {pf c baseclass} {
    global classy_baseclasses classy_equivmapping classy_hash classy_classes

    set classinfo [Bdb_QueryClassTag $pf $c]
    set floc [lindex $classinfo 3]
    set name [lindex $classinfo 4]
    if {![info exists classy_hash($floc,$name)]} {
	if {$baseclass == -1} { 
	    lappend classy_baseclasses [list $pf $c];
	} else {
	    lappend classy_classes($classy_equivmapping($pf,$baseclass),subs) \
		    [list $pf $c]
	}
	set classy_hash($floc,$name) "$pf,$c"
	set classy_equivmapping($pf,$c) "$pf,$c"
	set classy_classes($classy_equivmapping($pf,$c),subs) {}
    } else {
	set classy_equivmapping($pf,$c) $classy_hash($floc,$name)
    }
    set subs [Cgm_ClassInfo $pf $c subs]
    if {[llength $subs] > 0} {
	foreach subclass $subs {
	    joinEquivHelper $pf $subclass $c
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
PM_AddGlobalSelect $myself { global_selectClassTag }


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

# Check for language-tool compatibility
if {![Lang_CheckCompatibility [PM_GetProjectLangs] $myself]} {
    showError "$myself is not compatible with the project language(s)."
    exit
}

set depfile(project) [lindex $pm_status 0]
set depfile(host)    [lindex $pm_status 1]
set depfile(arch)    [lindex $pm_status 2]
set depfile(root)    [lindex $pm_status 3]
set depfile(dir)     [lindex $pm_status 4]
set depfile(hostarch) "$depfile(host) ($depfile(arch))"

createWindow
initialize 1

# -- Start with the virtual root for each file
foreach bc $classy_baseclasses {
    traverseHierarchy [lindex $bc 0] [lindex $bc 1] 0
}
.$myself.graph.can configure -scrollregion \
	[list 0 0 [expr 260+$ch_maxlevel*200] [expr 60+$ch_maxpos*30]]
foreach bc $classy_baseclasses {
    displayHierarchy [lindex $bc 0] [lindex $bc 1]
}
expandMenu

removeMessage

