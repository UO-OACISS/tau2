#*********************************************************************#
#*                TAU/pC++/Sage++  Copyright (C) 1994                *#
#*                        Jerry Manic Saftware                       *#
#*  Indiana University  University of Oregon  University of Rennes   *#
#*********************************************************************#
#
# SAS 7/22/96
# since all of the indices into the funcs array now have the form 
# funcs(<file><tag><whatever>)
# all procedures with calls to the funcs array need to take in a file
# as well as a tag. In fact, just about everything needs to have an associated file
# with it, to keep the multiple file functionality straight.
#

set DEBUG_SET 0


proc DEBUG {} {
    global DEBUG_SET

    return $DEBUG_SET
}


displayMessage hourglass "Loading $myself..."

source "$TAUDIR/inc/depfile.tcl"
source "$TAUDIR/inc/bdbm_utils.tcl"
# Need fileio.tcl for remote operations
source "$TAUDIR/inc/fileio.tcl"

set cg_colors {black #0000A0000000}; # basic display colors
set cg_showlib 0;                # display library (non-body) functions?
set cg_showclass 1;              # display class name of member functions too?
set cg_showtype 1;               # display type of functions?
set cg_sel_tag NONE;             # current selected function
set cg_sel_file NONE;
set cg_sel_cs_obj -1;            # currently selected callsite
set cg_sel_cs_info "";           # info about currently selected callsite
set cg_drawmode "EXPANDED";      # drawing mode (EXPANDED | COMPACT)
set cg_showcolor 0;              # display functions in color?
set cagey_funcs(1) "";           #Setting up local funcs array

#
# setColors: color callgraph nodes according to racy color mapping
#

proc setColors {warning {do_redraw 1}} {

  global cg_showcolor cg_tagcol

  if { $cg_showcolor } {
    set map [xsend racy getColorCode]
    if { $map != "" } {
      foreach m $map {
        set cg_tagcol([lindex $m 0]) [lindex $m 1]
      }
      if {$do_redraw} { redrawGraph; }
    } else {
      set cg_showcolor 0
      if { $warning } {
        showError "Color information not available. Racy not running."
      } else {
        if {$do_redraw} { redrawGraph; }
      }
    }
  } else {
    if [info exists cg_tagcol] { unset cg_tagcol }
      if {$do_redraw} { redrawGraph; }
  }
}

#
# selectLine: implementation of global feature "selectLine" for fancy
#             highlight specified line in file
#
#       line: line number to select
#       file: file name to select
#        tag: tag of call site to select or -1
#

proc selectLine {line file {tag -1}} {

  global myself
  global cagey_funcs
  global cg_sel_cs_obj cg_sel_cs_info cg_drawmode

  # -- if there is already a function selected, un-select it
  # -- (display in black)
  if { $cg_sel_cs_obj != -1 } {
    .$myself.graph.can delete animtag
  }

  # -- if selected call site is part of the callgraph
  # -- select it (display it in blue)
  set obj -1
  if { $cg_drawmode == "EXPANDED" } {
    if [info exists cagey_funcs($file,$tag,css)] {
      set idx [lsearch -exact $cagey_funcs($file,$tag,css) [list $line $file]]
      if { $idx != -1 } {
        set obj [lindex $cagey_funcs($file,$tag,objs) $idx]
        HighLightObj $obj oval blue animtag
        set cg_sel_cs_info [list $line $file $tag]
        displayObj $obj
      }
    }
  } else {
    if [info exists cagey_funcs($file,$tag,objs)] {
      set obj [lindex $cagey_funcs($file,$tag,objs) 0]
      HighLightObj $obj oval blue animtag
      set cg_sel_cs_info [list $line $file $tag]
      displayObj $obj
    }
  }
  set cg_sel_cs_obj $obj
}

#
# HighLightObj: highlight a canvas object by drawing a colored shape
#               around it
#
#          obj: canvas object to highlight
#        shape: rectangle | oval | roundrec
#        color: color of shape outline
#          tag: canvas tag for created shape object
#

proc HighLightObj {obj shape color tag} {

  global myself

  set bbox [.$myself.graph.can bbox $obj]
  if { $shape != "roundrec" } {
    .$myself.graph.can create $shape \
               [expr [lindex $bbox 0]-3] [expr [lindex $bbox 1]-3] \
               [expr [lindex $bbox 2]+3] [expr [lindex $bbox 3]+3] \
               -outline $color -tag $tag
    .$myself.graph.can raise $obj
  } else {
    set u [expr [lindex $bbox 3]-[lindex $bbox 1]]
    .$myself.graph.can create arc \
               [expr [lindex $bbox 0]-3] [expr [lindex $bbox 1]-3] \
               [expr [lindex $bbox 0]+$u+3] [expr [lindex $bbox 1]+$u+3] \
               -extent 180 -start 90 -fill $color -tag $tag -style arc
    .$myself.graph.can create arc \
               [expr [lindex $bbox 2]-$u-3] [expr [lindex $bbox 3]-$u-3] \
               [expr [lindex $bbox 2]+3] [expr [lindex $bbox 3]+3] \
               -extent -180 -start 90 -fill $color -tag $tag -style arc
    .$myself.graph.can create line \
               [expr [lindex $bbox 0]+$u/2-3] [expr [lindex $bbox 1]-3] \
               [expr [lindex $bbox 2]-$u/2+3] [expr [lindex $bbox 1]-3] \
               -fill $color -tag $tag
    .$myself.graph.can create line \
               [expr [lindex $bbox 0]+$u/2-3] [expr [lindex $bbox 3]+3] \
               [expr [lindex $bbox 2]-$u/2+3] [expr [lindex $bbox 3]+3] \
               -fill $color -tag $tag
  }
}

#
# selectFuncTag: implementation of global feature "selectFunction" for cagey
#                display all instances of selected function in red
#
#           tag: id of function to select
#

proc selectFuncTag {file tag} {

  global myself
  global cagey_funcs
  global cg_sel_tag cg_sel_file

  # -- if there is already a function selected, un-select it
  # -- (display in black)
  if { $cg_sel_tag != "NONE" } {
    if [info exists cagey_funcs($cg_sel_file,$cg_sel_tag,objs)] {
      .$myself.graph.can delete seltag
    }
  }

  # -- if selected function is part of the callgraph
  # -- select it (display it in red)
  if [info exists cagey_funcs($file,$tag,objs)] {
    foreach o $cagey_funcs($file,$tag,objs) {
      HighLightObj $o rectangle red seltag
    }
    displayObj [lindex $cagey_funcs($file,$tag,objs) 0]
  }
  set cg_sel_file $file
  set cg_sel_tag $tag
}

#
# createWindow: create and display main window of cagey
#

proc createWindow {} {

  global TAUDIR
  global myself

  toplevel .$myself
  wm title .$myself "CAGEY"
  wm iconbitmap .$myself @$TAUDIR/xbm/$myself.xbm
  wm minsize .$myself 300 400

  frame .$myself.graph

  # -- configure graph area
  canvas .$myself.graph.can -width 400 -height 600 -background white
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

  # Rather than make our apologies, remove the option. . .
  # This'll get linked to one or more of the instrumentation modules in time.
  #.$myself.bar.b1.m1 add command -label "Generate instrfile" -underline 0 \
  \#                 -command "sorry Generate"

  .$myself.bar.b1.m1 add command -label "Print graph" -underline 0 \
                   -command "printGraph"
  .$myself.bar.b1.m1 add separator
  .$myself.bar.b1.m1 add command -label "Exit"  -underline 0 -command "exit"

  menubutton .$myself.bar.b2 -text View -menu .$myself.bar.b2.m1 -underline 0
  menu .$myself.bar.b2.m1
  .$myself.bar.b2.m1 add radiobutton -label "expanded callgraph" \
                   -underline 1 -variable cg_drawmode -value EXPANDED \
                   -command "redrawGraph"
  .$myself.bar.b2.m1 add radiobutton -label "compact callgraph" \
                   -underline 1 -variable cg_drawmode -value COMPACT \
                   -command "redrawGraph"
  .$myself.bar.b2.m1 add separator
  .$myself.bar.b2.m1 add checkbutton -label "Show library functions" \
                   -underline 5 -variable cg_showlib -onvalue 1 -offvalue 0 \
                   -command "redrawGraph"
  .$myself.bar.b2.m1 add checkbutton -label "Show class names" \
                   -underline 5 -variable cg_showclass -onvalue 1 -offvalue 0 \
                   -command "redrawGraph"
  .$myself.bar.b2.m1 add checkbutton -label "Show type of function" \
                   -underline 5 -variable cg_showtype -onvalue 1 -offvalue 0 \
                   -command "redrawGraph"
  .$myself.bar.b2.m1 add checkbutton -label "Color functions" \
                   -underline 5 -variable cg_showcolor -onvalue 1 -offvalue 0 \
                   -command "setColors 1"
  .$myself.bar.b2.m1 add separator
  .$myself.bar.b2.m1 add cascade -label "Expand graph" -underline 0 \
                                 -menu .$myself.bar.b2.m1.1
  menu .$myself.bar.b2.m1.1

  createToolMenu .$myself.bar.b4

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

  pack .$myself.bar.b1 .$myself.bar.b2 -side left -padx 5
  pack .$myself.bar.b3 .$myself.bar.b4 -side right -padx 5

  pack .$myself.bar   -side top -fill x
  pack .$myself.graph -side left -padx 15 -pady 15 -fill both -expand yes
}

#
# loadDep: switch to another application
#          this is invoked from the TAU master control window
#
#  SAS: This procedure needs to accept those changes through the PM, and update
#  its local copy of depfile accordingly. 
#

proc loadDep {} {
    global myself \
	    cagey_funcs \
	    cg_sel_tag cg_sel_file \
	    cg_sel_cs_obj cg_sel_cs_info
    
    # -- delete old database of functions
    Cgm_RemoveAllDeps

    # -- reinitialize variables
    set cg_sel_tag NONE
    set cg_sel_file NONE
    set cg_sel_cs_obj -1
    set cg_sel_cs_info ""
    if [info exists cagey_funcs] { unset cagey_funcs; }
    set cagey_funcs(1) ""

    # -- set up the information about the maintag
    set cagey_funcs(maintag) [Bdb_GetMaintag]
    set cagey_funcs([lindex $cagey_funcs(maintag) 0],[lindex $cagey_funcs(maintag) 1],equiv_func) $cagey_funcs(maintag)
    set file [lindex $cagey_funcs(maintag) 0]
    if {[llength $cagey_funcs(maintag)] == 1} {
	if {[string match $cagey_funcs(maintag) "BDBM_FAILED"] \
		|| !$cagey_funcs(maintag)} {
	    showError "No project loaded, or incomplete compilation"
	    return
	}
    }
    
    # -- read new depfile and rebuild function database
    if { [Cgm_LoadDep  $file "-dumpcg"] == "NOT_OK" } {
	showError "Couldn't read browser information."
	exit
    }
    
    # -- draw new callgraph
    set cagey_funcs($file,[lindex $cagey_funcs(maintag) 1],fold) 1
    setColors 0 0
    set cg_sel_tag NONE
    set cg_sel_cs_obj -1
    
    # -- update "expand to level" menu
    .$myself.bar.b2.m1.1 delete 0 last
    expandMenu
}

#
# displayObj: scroll window so that a specific object is on the same
#             position on the screen after display operations
#
#        obj: object of interest
#

proc displayObj {obj} {

  global myself

  set winpos [.$myself.graph.can canvasy 0]
  set objpos [lindex [.$myself.graph.can coords $obj] 1]
  set maxyscroll [lindex [.$myself.graph.can cget -scrollregion] 3]

  if { [expr $objpos-$winpos] < 0 || [expr $objpos-$winpos] > 600 } {
    # -- object not on screen
    set target [expr $objpos-300]
    .$myself.graph.can yview moveto [expr $target / $maxyscroll]
  }
}

#
# toggleFold: toggle object between "expanded" and "folded" mode
#
#        obj: object to toggle mode
#        tag: Sage++ id of the corresponding function
#

proc toggleFold {file obj tag} {

  global myself
  global cagey_funcs
    
    if [info exists cagey_funcs($file,$tag,fold)] {
	set f $cagey_funcs($file,$tag,fold)
    } else {
	set cagey_funcs($file,$tag,fold) 0
	set f 1
    }
    set cagey_funcs($file,$tag,fold) [expr 1-$f]
    set temp_callsInfo [Cgm_FuncInfo $file $tag "calls"]
    if {[string match $temp_callsInfo "NOT_OK"]} {
	if [DEBUG] {	puts "Cagey:TravGraphX: $file has no calls information"}
	return
    }
    set hasChildren [llength $temp_callsInfo]

  if { $hasChildren } {
    # -- function has children, redisplay new graph layout
    # -- also, scroll window so that the toggled object is on the same
    # -- position on the screen after redisplay
    set oldmyself [lsearch $cagey_funcs($file,$tag,objs) $obj]
    set oldobj [lindex [.$myself.graph.can coords $obj] 1]
    set oldwin [.$myself.graph.can canvasy 0]
    set maxyscroll [lindex [.$myself.graph.can cget -scrollregion] 3]
    redrawGraph
    set newmyself [lindex $cagey_funcs($file,$tag,objs) $oldmyself]
    set newobj [lindex [.$myself.graph.can coords $newmyself] 1]

### Removed by kurtw 1/26/96
#    .$myself.graph.can yview moveto \
#	    [expr ($newobj-$oldobj+$oldwin) / $maxyscroll]
  }
}

#
# toggleObj: toggle object between two display modes (indicated by colors)
#            not yet used for something useful
#
#       tag: Sage++ id of the corresponding function
#

proc toggleObj {file tag} {
    
    global myself
    global cagey_funcs
    global cg_colors
    
    if [info exists $cagey_funcs($file,$tag,state)] {
	set s $cagey_funcs($file,$tag,state)
    } else {
	set cagey_funcs($file,$tag,state) 0
	set s 0
    }

    set s [expr 1-$s]
    foreach o $cagey_funcs($file,$tag,objs) {
	.$myself.graph.can itemconfigure $o -fill [lindex $cg_colors $s]
    }
}

#
# transGraph: traverse callgraph to expand the graph up to a specific
#             level given in cg_exp_level
#             does not work as expected, as the same function can be
#             called on different levels
#             *called recursively *
#
#        tag: function id of root of subtree to traverse
#      level: it's level
#     parent: function id of parent function
#

proc transGraph {file tag level parent} {

  global cagey_funcs
  global cg_exp_level cg_marked

    set temp_callsInfo [Cgm_FuncInfo $file $tag "calls"]
    if {[string match $temp_callsInfo "NOT_OK"]} {
	if [DEBUG] {	puts "Cagey:transGraph: no calls information for tag $tag in $file"}
	return
    }

    if { $tag == $parent } { return }

  if { $level < $cg_exp_level } {
    set cagey_funcs($file,$tag,fold) 0
    lappend cg_marked $tag
  } elseif { $level >= $cg_exp_level } {

      #Hopefully this'll let it get redrawn.
      # I took out this guard, which seemed to have no purpose.
      # Now the graph will both expand and contract properly.
      # SAS 10/21/96
    # if { [lindex $cg_marked $tag] == -1 } {
      set cagey_funcs($file,$tag,fold) 1
    # }
    return
  }

  foreach t $temp_callsInfo {
    transGraph $file $t [expr $level+1] $tag
  }
}

#
# getMaxLevel: compute depth of call graph by traversing it recursively
#
#         tag: function id of root of subtree to traverse
#       level: it's level
#      parent: function id of parent function
#

proc getMaxLevel {file tag level parent} {

  global cagey_funcs
  global cg_max_level

  # -- reached new level? remember
  if { $level > $cg_max_level } { set cg_max_level $level }

  # -- check for cycles
  if [ info exists cagey_funcs($file,$tag,mark) ] { return }
  set cagey_funcs($file,$tag,mark) 1

  # -- walk children
    set temp_callsInfo [Cgm_FuncInfo $file $tag "calls"]
    if {[string match $temp_callsInfo "NOT_OK"]} {
	if [DEBUG] {	puts "Cagey:TravGraphX: $file has no calls information"}
	return
    }
    foreach t $temp_callsInfo {
    getMaxLevel $file $t [expr $level+1] $tag
  }
 
  # -- release mark
  unset cagey_funcs($file,$tag,mark)
}

#
# expandMenu: update "expand" menu, so that it has an entry for each
#             level of the callgraph
#

proc expandMenu {} {

  global depfile cagey_funcs
  global myself cg_exp_level cg_max_level

  set cg_max_level 0
    getMaxLevel [lindex $cagey_funcs(maintag) 0] \
      [lindex $cagey_funcs(maintag) 1] 0 -1

  for {set i 0} {$i<=$cg_max_level} {incr i} {
    .$myself.bar.b2.m1.1 add radiobutton -label "upto level $i" -underline 11 \
                  -variable cg_exp_level -value $i \
                  -command "expandHierarchy"
  }
  set cg_exp_level 0
}

#
# expandHierarchy: expand callgraph after user selected level through
#                  the "expand" menu
#

proc expandHierarchy {} {
   
    global depfile cagey_funcs
    global cg_marked
    
    set cg_marked ""
    transGraph [lindex $cagey_funcs(maintag) 0] \
	[lindex $cagey_funcs(maintag) 1] 0 -1 
    redrawGraph
}

set cg_cx 50;   # start x coordinate
set cg_cy 50;   # start y coordinate
set cg_dx 30;   # delta x to next node
set cg_dy 30;   # delta y to next node

#
# drawGraphExpd: draw "expanded" callgraph by traversing it recursively
#
#          file: filename where this instance of the function is called
#           tag: function id of root of subtree to traverse
#         level: it's level
#        parent: function id of parent function
#          line: linenumber where this instance of the function is called
#

proc drawGraphExpd {file tag level parent {line -1}} {

    global myself
    global cagey_funcs
    global cg_cx cg_cy cg_dx cg_dy
    global cg_colors
    global cg_showlib cg_showclass cg_showtype
    global cg_tagcol cg_showcolor
    
    # -- initialize parameters for current node
    # Ported to CGM interface 7/31/96
    set name  [Cgm_FuncInfo $file $tag "name"]
    if {[string match $name "CGM_FAILED"]} {
	if [DEBUG] {	puts "No information on name. file == $file, tag == $tag"}
	return
    }
    set calls [Cgm_FuncInfo $file $tag "calls"]
    if {[string match $calls "CGM_FAILED"]} {
	if [DEBUG] {	puts "No information on calls."}
	return
    }
   
    set class [lindex [Cgm_FuncInfo $file $tag "class"] 0]
    if {[string match $class "CGM_FAILED"]} {
	if [DEBUG] {	puts "No information on class."}
	return
    }
   
    if [llength $calls] {
	set cfile [lindex [Cgm_FuncInfo $file $tag "file"] 1]
	if {[string match $cfile "CGM_FAILED"]} {
	    if [DEBUG] {	puts "No information on cfile."}
	    return
	}
   
	set cline [Cgm_FuncInfo $file $tag "childline"]
	if {[string match $cline "CGM_FAILED"]} {
	    if [DEBUG] {puts "No information on cline."}
	    return
	}
	
    }
    set ptype [lindex [Cgm_FuncInfo $file $tag "type"] 0]
    if [info exists cagey_funcs($file,$tag,state)] {
	set s $cagey_funcs($file,$tag,state)
    } else { 
	set cagey_funcs($file,$tag,state) 0
	set s 0
    }
    if [info exists cagey_funcs($file,$tag,fold)] {
	set fold $cagey_funcs($file,$tag,fold)
    } else {
	set cagey_funcs($file,$tag,fold) 1
	set fold 1
    }
    set x [expr $cg_cx+$level*$cg_dx]
    set y [expr $cg_cy+10]

    # -- prepend classname if necessary
    if { $cg_showclass && $class != "-" } {
	set name "$class::$name"
    }
    
    # -- show type if necessary
    if { $cg_showtype && ! [lindex [Cgm_FuncInfo $file $tag "file"] 0] } {
	set name "<$name>"
    }
    if { $cg_showtype && $ptype == "par" } {
	set name "|| $name"
    }
    
    # -- mark recursive procedures
    if [ info exists cagey_funcs($file,$tag,mark) ] {
	set name "$name **"
    }
    
    # -- mark nodes which are folded and have children
    if { $fold && [llength $calls] } {
	set setFold 1
	if { ! $cg_showlib } {
	    set setFold 0
	    foreach t $calls {

		# Make sure that we are in the correct depfile and it's loaded
		if [info exists cagey_funcs($file,$t,equiv_func)] {
		    set linked_progfile \
			    [lindex $cagey_funcs($file,$t,equiv_func) 0]
		    set linked_tag      \
			    [lindex $cagey_funcs($file,$t,equiv_func) 1]
		} else {
		    set local_func_name         [Cgm_FuncInfo $file $t "name"]
		    set local_func_mangled_name [Cgm_FuncInfo $file $t "mname"]
		    set temp_bdbInfo [Bdb_FindFuncDefByName \
			    $local_func_name $local_func_mangled_name]
		    if {[llength $temp_bdbInfo] == 0} {
			set linked_progfile $file
			set linked_tag $t
		    } else {
			set linked_progfile [lindex $temp_bdbInfo 0]
			set linked_tag      [lindex $temp_bdbInfo 1]
		    }
		}

		if {![Cgm_IsDepLoaded $linked_progfile]} {
		    if { [Cgm_LoadDep  $linked_progfile "-dumpcg"] \
			    == "NOT_OK" } {
			showError "Couldn't read browser information for $linked_progfile."
			exit
		    }
		}
		set is_defined [lindex [Cgm_FuncInfo \
			$linked_progfile $linked_tag "file"] 0]
		set cagey_funcs($file,$t,equiv_func) \
			[list $linked_progfile $linked_tag]
		
		if { $is_defined } {
		    set setFold 1
		    break
		}
	    }
	}
	if { $setFold } {
	    set name "$name  . . ."
	}
    }

  # -- finally draw node and add bindings
  if { $cg_showcolor } {
    if [info exists cg_tagcol($tag)] {
      set col $cg_tagcol($tag)
    } else {
      set col $cg_tagcol(-1)
    }
  } else {
    set col [lindex $cg_colors $s]
  }
  set obj [.$myself.graph.can create text $x $cg_cy -text $name -anchor w \
                   -fill $col -font -Adobe-Helvetica-Bold-R-Normal--*-120-*]
  .$myself.graph.can bind $obj <Button-1> "toggleFold $file $obj $tag"
  .$myself.graph.can bind $obj <Shift-Button-1> "toggleObj $file $tag"
  .$myself.graph.can bind $obj <Button-2> "PM_GlobalSelect $file global_showFuncTag $tag"
  .$myself.graph.can bind $obj <Button-3> "PM_GlobalSelect $file global_selectFuncTag $tag"
  lappend cagey_funcs($file,$tag,objs) $obj
  lappend cagey_funcs($file,$tag,css) [list $line $file]
  incr cg_cy $cg_dy
#SAS took out  $fold ||
    if { $fold || [info exists cagey_funcs($file,$tag,mark)] } {
    return
  }
  set cagey_funcs($file,$tag,mark) 1

    # -- draw children of current node
    set c 0
    set temp_callsInfo [Cgm_FuncInfo $file $tag "calls"]
    if {[string match $temp_callsInfo "NOT_OK"]} {
	if [DEBUG] {	puts "Cagey:drawGraphExpd: $file has no calls information"}
	return
    }
    # so now that we have a list of function calls made from the target function,
    # chase those calls through the files,and determine how deep the 
    # call graph gets.
    foreach t $temp_callsInfo {

	# Make sure that we are in the correct depfile and it's loaded
	if [info exists cagey_funcs($file,$t,equiv_func)] {
	    set linked_progfile \
		    [lindex $cagey_funcs($file,$t,equiv_func) 0]
	    set linked_tag      \
		    [lindex $cagey_funcs($file,$t,equiv_func) 1]
	} else {
	    set local_func_name         [Cgm_FuncInfo $file $t "name"]
	    set local_func_mangled_name [Cgm_FuncInfo $file $t "mname"]
	    set temp_bdbInfo [Bdb_FindFuncDefByName \
		    $local_func_name $local_func_mangled_name]
	    if {[llength $temp_bdbInfo] == 0} {
		set linked_progfile $file
		set linked_tag $t
	    } else {
		set linked_progfile [lindex $temp_bdbInfo 0]
		set linked_tag      [lindex $temp_bdbInfo 1]
	    }
	}
	set cagey_funcs($file,$t,equiv_func) \
		[list $linked_progfile $linked_tag]
		
	set temp_fileInfo [Cgm_FuncInfo $linked_progfile $linked_tag "file"]
	if {[string match $temp_fileInfo "NOT_OK"]} {
	    if [DEBUG] { puts "Cagey:DrawGraphExpd: $linked_progfile has no file info for tag $linked_tag"}
	    return
	}
	set body [lindex $temp_fileInfo 0]
	
	if { $cg_showlib || $body } {
	    .$myself.graph.can create line [expr $x+5] $y \
		[expr $x+5] $cg_cy \
		[expr $x+$cg_dx-5] $cg_cy -arrow last
	    drawGraphExpd $linked_progfile $linked_tag \
		    [expr $level+1] $tag [lindex $cline $c]
	}
	incr c
    }
    
    unset cagey_funcs($file,$tag,mark)
}

#
# redrawGraph: redraw callgraph by resetting variables, deleting old
#              callgraph, then draw new one
#

proc redrawGraph {} {
   
    global myself
    global cagey_funcs
    global depfile
    global cg_sel_tag
    global cg_sel_cs_obj cg_sel_cs_info
    
    # -- delete obsolete function object information
    
    # Modified 7/22/96 SAS
    # Need to access the file structure of the funcs array as well as the ftags.
    # Also, need modification to access those ftags through the BDBM.
    set filelist [PM_GetFiles]
    foreach file $filelist {
	# the allftags index an the depfile array is used here, but there isn't a 
	# procedural hook to get it out. 
	# Needs fixing SAS 7/31/96
	set cagey_funcs($file,allftags) [Cgm_DepInfo $file "allftags"]
	if {[string match $cagey_funcs($file,allftags) "NOT_OK"]} {
	    if [DEBUG] { puts "Cagey:redrawGraph: allftags failed."}
	    return
	}
	foreach t $cagey_funcs($file,allftags) {
	    if [ info exists cagey_funcs($file,$t,objs) ] { 
		unset cagey_funcs($file,$t,objs) 
	    }
	    if [ info exists cagey_funcs($file,$t,x) ] { 
		unset cagey_funcs($file,$t,x) 
	    }
	    if [ info exists cagey_funcs($file,$t,y) ] { 
		unset cagey_funcs($file,$t,y) 
	    }
	}
    }

  # -- delete graphic objects
  .$myself.graph.can delete all

  # -- redraw graph
  drawGraph
  if { $cg_sel_tag != "NONE" } {
    selectFuncTag $file $cg_sel_tag
  }
  if { $cg_sel_cs_obj != -1 } {
    eval selectLine $file $cg_sel_cs_info
  }
}

set pr_printVar window;  # print mode: "window" or "all"
set pr_psfile "";        # PostScript filename

#
# printGraph: generate PostScript version of callgraph
#

proc printGraph {} {

  global depfile
  global pr_psfile
  global pr_printVar
  global cg_cx cg_cy cg_dx cg_dy cg_drawmode cg_max_x cg_max_y

  toplevel .print
  wm title .print "Print"

  regsub .pmf $depfile(project) .ps pr_psfile
  frame .print.top
  pack .print.top -side top -padx 15 -pady 15

  radiobutton .print.r1 -text "Graph" -variable pr_printVar \
                    -value all -width 10 -anchor w
  radiobutton .print.r2 -text "Window" -variable pr_printVar \
                    -value window -width 10 -anchor w
  frame .print.f1 -width 20 -height 20
  label .print.l1 -text "Filename:"
  entry .print.e1 -textvariable pr_psfile -relief sunken
  pack .print.r1 .print.r2 .print.f1 .print.l1 .print.e1 \
                 -in .print.top -side top -anchor w

  frame .print.bottom -relief sunken -bd 1
  pack .print.bottom -side left -padx 15 -pady 10

  button .print.b1 -text "print" -command {
      if { [info exists pr_psfile] } {
        if { $pr_printVar == "all" } {
          if { $cg_drawmode == "COMPACT" } {
            .$myself.graph.can postscript -colormode color -file $pr_psfile \
                       -x 0 -y 0 -height [expr 60+$cg_max_y*30] \
                       -width [expr 260+$cg_max_x*300]
          } else {
            .$myself.graph.can postscript -colormode color -file $pr_psfile \
                       -x 0 -y 0 -height [expr $cg_cy-$cg_dy+50] -width 600
          }
        } else {
          .$myself.graph.can postscript -colormode color -file $pr_psfile
        }
      }
      destroy .print
    }
  pack .print.b1 -in .print.bottom -side top -padx 5 -pady 5

  button .print.b2 -text "cancel" -command "destroy .print"
  bind .print.e1 <Return> {.print.b1 invoke}
  bind .print <Return> {.print.b1 invoke}
  pack .print.b2 -side right -padx 15

  tkwait visibility .print
  set oldfocus [focus]
  focus .print.e1

  tkwait window .print
  focus $oldfocus
}

#
# traverseGraphX: traverse callgraph to compute X coordinates for
#                 compact callgraph
#
#            tag: function id of root of subtree to traverse
#              x: it's x coordinate (level)
#         parent: function id of parent function
#

proc traverseGraphX {file tag x parent} {
    global cagey_funcs
    global myself
    global cg_max_x
    global cg_showlib
    
    # -- determine maximal level (x coordinate)
    if { $x > $cg_max_x } { set cg_max_x $x}
    
    # -- set x coordinate for node (always use the deepest level of a function)
    if [info exists cagey_funcs($file,$tag,x)] {
	if { $cagey_funcs($file,$tag,x) < $x  \
		 && ![info exists cagey_funcs($file,$tag,mark)] } {
	    set cagey_funcs($file,$tag,x) $x
	}
    } else {
	set cagey_funcs($file,$tag,x) $x

    }
    incr x
    
    # -- traverse children nodes if necessary
    if {![info exists cagey_funcs($file,$tag,fold)]} {
	set cagey_funcs($file,$tag,fold) 1
    }
 
    if { !$cagey_funcs($file,$tag,fold) \
	     && ![info exists cagey_funcs($file,$tag,mark)] } {
	set cagey_funcs($file,$tag,mark) 1
	set temp_callsInfo [Cgm_FuncInfo $file $tag "calls"]
	if {[string match $temp_callsInfo "NOT_OK"]} {
	    if [DEBUG] { puts "Cagey:TravGraphX: $file has no calls information"}
	    return
	}
	# so now that we have a list of function calls made from the target function,
	# chase those calls through the files,and determine how deep the 
	# call graph gets.
	foreach c $temp_callsInfo {
	    # Make sure that we are in the correct depfile and it's loaded
	    set local_func_name         [Cgm_FuncInfo $file $c "name"]
	    set local_func_mangled_name [Cgm_FuncInfo $file $c "mname"]
	    set temp_bdbInfo [Bdb_FindFuncDefByName \
		    $local_func_name $local_func_mangled_name]
	    if {[llength $temp_bdbInfo] == 0} {
		set is_defined 0
		set linked_progfile $file
		set linked_tag $c
	    } else {
		set is_defined 1
		set linked_progfile [lindex $temp_bdbInfo 0]
		set linked_tag      [lindex $temp_bdbInfo 1]
	    }
	    set cage_funcs($file,$c,equiv_func) \
		    [list $linked_progfile $linked_tag]
	    if {$is_defined && ![Cgm_IsDepLoaded $linked_progfile]} {
		if { [Cgm_LoadDep  $linked_progfile "-dumpcg"] == "NOT_OK" } {
		    showError "Couldn't read browser information for $linked_progfile."
		    exit
		}
	    }

	    if { $cg_showlib || $is_defined } {
		traverseGraphX $linked_progfile $linked_tag $x $tag
	    }
	}
	unset cagey_funcs($file,$tag,mark)
    }
}

#
# traverseGraphY: traverse callgraph to compute Y coordinates for
#                 compact callgraph
#
#            tag: function id of root of subtree to traverse
#

proc traverseGraphY {file tag} {
    global cagey_funcs 
    global myself
    global cg_max_y
    global cg_showlib

    if {![info exists cagey_funcs($file,$tag,fold)]} {
	set cagey_funcs($file,$tag,fold) 1
    }
    if { $cagey_funcs($file,$tag,fold) } {
	# -- node folded: just assign next available y coordinate
	incr cg_max_y
	set my_y $cg_max_y
    } else {
	# -- otherwise: first determine number of "real" children
	# -- (children which are displayed)
	set realcalls ""
	set temp_callsInfo [Cgm_FuncInfo $file $tag "calls"]
	if {[string match $temp_callsInfo "NOT_OK"]} {
	    if [DEBUG] { puts "Cagey:TravGraphY: $file has no calls information"}
	    return
	}
	foreach c $temp_callsInfo {
	    set temp_fileInfo [Cgm_FuncInfo $file $c "file"]
	    if {[string match $temp_fileInfo "NOT_OK"]} {
		if [DEBUG] {puts "Cagey:TravGraphY: $file has no file info for tag $c"}
		return
	    }

	    # Make sure that we are in the correct depfile and it's loaded
	    set local_func_name         [Cgm_FuncInfo $file $c "name"]
	    set local_func_mangled_name [Cgm_FuncInfo $file $c "mname"]
	    set temp_bdbInfo [Bdb_FindFuncDefByName \
		    $local_func_name $local_func_mangled_name]
	    if {[llength $temp_bdbInfo] == 0} {
		set is_defined 0
		set linked_progfile $file
		set linked_tag $c
	    } else {
		set is_defined 1
		set linked_progfile [lindex $temp_bdbInfo 0]
		set linked_tag      [lindex $temp_bdbInfo 1]
	    }
	    set cage_funcs($file,$c,equiv_func) \
		    [list $linked_progfile $linked_tag]
	    if {$is_defined && ![Cgm_IsDepLoaded $linked_progfile]} {
		if { [Cgm_LoadDep  $linked_progfile "-dumpcg"] == "NOT_OK" } {
		    showError "Couldn't read browser information for $linked_progfile."
		    exit
		}
	    }

	    if { ![info exists cagey_funcs($linked_progfile,$linked_tag,y)] \
		    && ($cg_showlib || $is_defined) } {
		lappend realcalls [list $linked_progfile $linked_tag]
		set cagey_funcs($linked_progfile,$linked_tag,y) "PRESET"
	    }
	}
	
	switch -exact [llength $realcalls] {
	    0 {
		# -- no children: assign next available y coordinate
		incr cg_max_y
		set my_y $cg_max_y
	    }
	    1 {
		# -- one children: assign y coordinate of child
		set my_y [traverseGraphY \
			[lindex [lindex $realcalls 0] 0] \
			[lindex [lindex $realcalls 0] 1]]
	    }
	    default {
		# -- more than one children: place parent in the middle of all childs
		set start_y $cg_max_y
		foreach c $realcalls {
		    set end_y [traverseGraphY [lindex $c 0] [lindex $c 1]]
		}
		set my_y [expr $start_y+($end_y-$start_y+1)/2]
	    }
	}
    }
    set cagey_funcs($file,$tag,y) $my_y
    return $my_y
}

#
# drawGraphCompact: draw "compact" callgraph by traversing it recursively
#
#              tag: function id of root of subtree to traverse
#           parent: function id of parent function
#

proc drawGraphCompact {file tag parent} {
    global cagey_funcs
    global myself
    global cg_colors
    global cg_showlib cg_showclass cg_showtype
    global cg_showcolor cg_tagcol
    
    # -- initialize parameters for current node
    set x [expr 30+$cagey_funcs($file,$tag,x)*300]
    set y [expr 30+$cagey_funcs($file,$tag,y)*30]
    set name  [Cgm_FuncInfo $file $tag "name"]
    set calls  [Cgm_FuncInfo $file $tag "calls"]
    set class [lindex [Cgm_FuncInfo $file $tag "class"] 0]
    set ptype [lindex  [Cgm_FuncInfo $file $tag "type"] 0]
    if [info exists cagey_funcs($file,$tag,state)] {
	set s $cagey_funcs($file,$tag,state)
    } else {
	set cagey_funcs($file,$tag,state) 0
	set s 0
    }
    if [info exists cagey_funcs($file,$tag,fold)] {
	set fold  $cagey_funcs($file,$tag,fold)
    } else {
	set cagey_funcs($file,$tag,fold) 1
	set fold 1
    }

    # -- prepend classname if necessary
    if { $cg_showclass && $class != "-" } {
	set name "$class::$name"
    }
    
    # -- show type if necessary
    set temp_fileInfo [Cgm_FuncInfo $file $tag "file"]
    if {[string match $temp_fileInfo "NOT_OK"]} {
	if [DEBUG] {	puts "Cagey:drawGC: $file has no file info for tag $tag"}
	return
    }
    
    if { $cg_showtype && ! [lindex $temp_fileInfo 0] } {
	set name "<$name>"
    }
    if { $cg_showtype && $ptype == "par" } {
	set name "|| $name"
    }
    
    # -- mark recursive procedures
    if [info exists cagey_funcs($file,$tag,mark)] {
	set name "$name **"
    }
    
    # -- mark nodes which are folded and have children
    if { $fold && [llength $calls] } {
	set setFold 1
	if { ! $cg_showlib } {
	    set setFold 0
	    foreach t $calls {
		
		# Make sure that we are in the correct depfile and it's loaded
		if [info exists cagey_funcs($file,$t,equiv_func)] {
		    set linked_progfile \
			    [lindex $cagey_funcs($file,$t,equiv_func) 0]
		    set linked_tag      \
			    [lindex $cagey_funcs($file,$t,equiv_func) 1]
		} else {
		    set local_func_name         [Cgm_FuncInfo $file $t "name"]
		    set local_func_mangled_name [Cgm_FuncInfo $file $t "mname"]
		    set temp_bdbInfo [Bdb_FindFuncDefByName \
			    $local_func_name $local_func_mangled_name]
		    if {[llength $temp_bdbInfo] == 0} {
			set linked_progfile $file
			set linked_tag $t
		    } else {
			set linked_progfile [lindex $temp_bdbInfo 0]
			set linked_tag      [lindex $temp_bdbInfo 1]
		    }
		}

		if {![Cgm_IsDepLoaded $linked_progfile]} {
		    if { [Cgm_LoadDep  $linked_progfile "-dumpcg"] \
			    == "NOT_OK" } {
			showError "Couldn't read browser information for $linked_progfile."
			exit
		    }
		}
		set is_defined [lindex [Cgm_FuncInfo \
			$linked_progfile $linked_tag "file"] 0]
		set cagey_funcs($file,$t,equiv_func) \
			[list $linked_progfile $linked_tag]
		
		if { $is_defined } {
		    set setFold 1
		    break
		}
	    }
	}
	if { $setFold } {
	    set name "$name  . . ."
	}
    }
    
    # -- finally draw node and add bindings
    if { $cg_showcolor } {
	if [info exists cg_tagcol($tag)] {
	    set col $cg_tagcol($tag)
	} else {
	    set col $cg_tagcol(-1)
	}
    } else {
	set col [lindex $cg_colors $s]
    }
    set obj [.$myself.graph.can create text $x $y \
	    -text $name -fill $col -anchor w \
	    -font -Adobe-Helvetica-Bold-R-Normal--*-120-*]
    .$myself.graph.can bind $obj <Button-1> "toggleFold $file $obj $tag"
    .$myself.graph.can bind $obj <Shift-Button-1> "toggleObj $file $tag"
    .$myself.graph.can bind $obj <Button-2> "PM_GlobalSelect $file global_showFuncTag $tag"
    .$myself.graph.can bind $obj <Button-3> "PM_GlobalSelect $file global_selectFuncTag $tag"
    lappend cagey_funcs($file,$tag,objs) $obj
    
    # -- compute size of node name
    set bbox [.$myself.graph.can bbox $obj]
    set ex [expr [lindex $bbox 2] - 10]
    set ey [expr [lindex $bbox 1] + ([lindex $bbox 3] - [lindex $bbox 1]) / 2]
    
    # -- draw children nodes and connecting arrows
    set drawn ""
    
    if { ! $fold && ! [info exists cagey_funcs($file,$tag,mark)] } {
	set cagey_funcs($file,$tag,mark) 1
	foreach c $calls {

		# Make sure that we are in the correct depfile and it's loaded
		if [info exists cagey_funcs($file,$c,equiv_func)] {
		    set linked_progfile \
			    [lindex $cagey_funcs($file,$c,equiv_func) 0]
		    set linked_tag      \
			    [lindex $cagey_funcs($file,$c,equiv_func) 1]
		} else {
		    set local_func_name         [Cgm_FuncInfo $file $c "name"]
		    set local_func_mangled_name [Cgm_FuncInfo $file $c "mname"]
		    set temp_bdbInfo [Bdb_FindFuncDefByName \
			    $local_func_name $local_func_mangled_name]
		    if {[llength $temp_bdbInfo] == 0} {
			set linked_progfile $file
			set linked_tag $c
		    } else {
			set linked_progfile [lindex $temp_bdbInfo 0]
			set linked_tag      [lindex $temp_bdbInfo 1]
		    }
		}
		set cagey_funcs($file,$c,equiv_func) \
			[list $linked_progfile $linked_tag]
		
		set temp_fileInfo \
			[Cgm_FuncInfo $linked_progfile $linked_tag "file"]
		if {[string match $temp_fileInfo "NOT_OK"]} {
		    if [DEBUG] { puts "Cagey:drawGC: $file has no file info for tag $c"}
		    return
	        }   
		set body [lindex $temp_fileInfo 0]
		
		if { $cg_showlib || $body } {
		    # -- determine whether child is called more than once
		    if { [set idx [lsearch $drawn "$linked_progfile:$linked_tag *"]] \
			    >= 0 } {
		    # -- child already called from this function
		    # -- (re-)label arrow with number of current call
		    set info [lindex $drawn $idx]
		    set num [lindex $info 4]
		    incr num
		    if { $num == 2 } {
			set t [.$myself.graph.can create text \
				[lindex $info 2] [lindex $info 3] \
				-text $num]
			set info [lreplace $info 1 1 $t]
		    } else {
			set t [lindex $info 1]
            .$myself.graph.can itemconfigure $t -text $num
		    }
		    set info [lreplace $info 4 4 $num]
		    set drawn [lreplace $drawn $idx $idx $info]
		} else {
		    # -- child not yet called
		    # -- draw arrow and remember arrow position
		    set pos [drawGraphCompact $linked_progfile $linked_tag $tag]
		    set nx  [lindex $pos 0]
		    set ny  [lindex $pos 1]
		    if { $ex > $nx } {
			# -- back arrow
			set ty [expr $ny - 15]
			set lobj [.$myself.graph.can create line $ex $ey \
				$ex $ty $nx $ty $nx $ny -arrow last]
		    } else {
			# -- forward arrow
			set lobj [.$myself.graph.can create line $ex $ey \
				$nx $ny -arrow last]
			set ty [expr $ey+($ny-$ey)/2]
		    }
		    set tx [expr $ex+($nx-$ex)/2]
		    .$myself.graph.can lower $lobj
		    lappend drawn [list $linked_progfile:$linked_tag {} $tx $ty 1]
		}
	    }
	}
	unset cagey_funcs($file,$tag,mark)
    }
    
    # -- draw borderless rectangle the size of the text
    # -- to make text standout from all the connecting arrows
    .$myself.graph.can create rectangle \
	    [expr [lindex $bbox 0]-3] [expr [lindex $bbox 1]-3] \
	    [expr [lindex $bbox 2]+3] [expr [lindex $bbox 3]+3] \
	    -fill white -outline ""
    
    # -- raise text above last drawn rectangles
    # -- cannot done the other way round, as the size of the rectangle
    # -- depends on the text
    .$myself.graph.can raise $obj
    
    return [list [expr $x-10] $y]
}

#
# drawGraph: initialize data, display callgraph, and set scrollregion
#

proc drawGraph {} {

    global cagey_funcs
    global myself
    global cg_max_x cg_max_y
    global cg_drawmode
    global cg_cx cg_cy cg_dx cg_dy

    set file [lindex $cagey_funcs(maintag) 0]
    set tag [lindex $cagey_funcs(maintag) 1]
    # the file that has the main() function in it.

  if { $cg_drawmode == "COMPACT" } {
    set cg_max_x 0
    set cg_max_y 0

    traverseGraphX $file $tag 0 -1
    traverseGraphY $file $tag
    drawGraphCompact $file $tag -1
    .$myself.graph.can configure -scrollregion \
                     [list 0 0 [expr 260+$cg_max_x*300] [expr 60+$cg_max_y*30]]
  } else {
      set cg_cx 50
      set cg_cy 50
      
      drawGraphExpd $file $tag 0 -1
      .$myself.graph.can configure -scrollregion \
	  [list 0 0 600 [expr $cg_cy-$cg_dy+50]]
  }
}


proc Tool_AcceptChanges {progfiles flag} {

    global depfile cagey_funcs

    switch $flag {

	d {
	    # Delete a file
	    # Since the file that has the main procedure in it may have
	    # been the file deleted, we need to reinitialize cagey with the 
	    # updated file list. . .
	    
	    loadDep
	    redrawGraph
	}


	a {
	    # Add a file
	    loadDep
	    redrawGraph
	}


	u {
	    # Update a file
	    loadDep
	    redrawGraph
	}
	
	p {
	    #Modify project information. 
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
	    loadDep
	    redrawGraph
	}

	e {
	    #This is a flag for updating during execution. No action is needed here.
	}
	
    }
    if [DEBUG] { puts "in cagey: returning from Tool_AcceptChanges"}
}



#
# exit - TAU exit function that communicates the event to other tau tools.
#
rename exit exit.old
proc exit {{status 0}} {
    # Taken and modified from Spiffy SAS 7/31/96
    global myself

    PM_RemGlobalSelect $myself \
	    { global_selectLine global_selectFuncTag }
    PM_RemTool $myself
    exit.old $status
}



# ------------
# -- main code
# ------------
# SAS : Although you can still summon cagey from the command line, it needs 
# to be specific about which project to read from. The basic unit of information
# for these tools is no longer the depfile; it's the project.
#
# Also, depending on the type of project (ansic or pc++ or fortran or <your lang here>),
# there may be compilation required for a series of files.
#
if {$argc == 0} {
    set parg [pwd]
} elseif {$argc == 1} {
  set parg [lindex $argv 0]
} else {
  puts stderr "usage: $myself \[\[host:\]projFile \| \[host:\]directory\]"
  exit
}

# Init the project manager (taud)
launchTauDaemon -waitfor
PM_AddTool $myself
PM_AddGlobalSelect $myself { \
	global_selectLine \
	global_selectFuncTag}
	

# So. . . where once dfile was set as the depfile, now a series of calls to the BDBM is needed
# for whatever files are in the project.
# SAS 7/22/96

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
	showError "No project opened!  This should work but it doesn't!"
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

loadDep
drawGraph

wm protocol .$myself WM_DELETE_WINDOW exit

removeMessage
