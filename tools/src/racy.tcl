#*********************************************************************#
#*                TAU/pC++/Sage++  Copyright (C) 1994                *#
#*                        Jerry Manic Saftware                       *#
#*  Indiana University  University of Oregon  University of Rennes   *#
#*********************************************************************#

#*********************************************************************#
#* TAU notes
#* April 28, 1997
#* Kurt Windisch (kurtw@cs.uoregon.edu)
#*
#* Currently profiling via pprof supports only projects consisting of
#* a single program file (*.C file).  Therefore, racy is not yet 
#* generalized for multifile projects and the variable racy_progfile
#* can be used to determine the progfile to associate with local tags
#* when interacting with the rest of TAU.
#*
#* Racy gets its profiling information from pprof.  Pprof provides
#* information on functions, collections and aggregates, but
#* collections and aggregates should never be reported for the
#* same project.
#*********************************************************************#

#
# A debugging framework, for installing scaffolding.
#
set DEBUG_SET 0
proc DEBUG {} {
    global DEBUG_SET

    return $DEBUG_SET
}

#
#  ALONE - for standalone racy
#
set ALONE_SET 0
proc ALONE {} {
    global ALONE_SET
    return $ALONE_SET
}

displayMessage hourglass "Loading racy..."

source "$TAUDIR/inc/help.tcl"
source "$TAUDIR/inc/printcan.tcl"
source "$TAUDIR/inc/stack.tcl"

set racy_progfile "";   # the single program file

set tcl_precision 17

#set Colors [list \
#    magenta green red blue yellow purple orange gold brown \
#    tan grey coral forestgreen skyblue seagreen beige steelblue1 royalblue \
#    pink cyan violet snow steelblue white ]

set Colors [list \
    coral  lavender cornflowerblue turquoise yellow palevioletred \
    mediumaquamarine dodgerblue1 salmon cadetblue1 orchid peachpuff1 \
    springgreen papayawhip hotpink royalblue mistyrose powderblue \
    chartreuse plum ] 

set TextColors [list \
    black black black white black white black black white \
    black black black black black black black black black \
    black black black black black black ]

set Stipples [list \
    hline1.bm cross.bm xline1.bm gray1.bm hline22.bm \
    vline1.bm gray3.bm cross0.bm xline2.bm \
    cross1.bm cross3.bm xline3.bm cross2.bm \
    cross5.bm vline2.bm xline4.bm cross4.bm hline2.bm \
    vline22.bm cross6.bm ]


#
# exit - TAU exit function that communicates the event to other tau tools.
#
rename exit exit.old
proc exit {{status 0}} {
    global myself

    PM_RemGlobalSelect $myself \
 	{ global_selectFuncTag }
    PM_RemTool $myself
    exit.old
}


#
# wraplindex: like lindex, but index "wraps around" if bigger
#             than length of list
#

proc wraplindex {list index} {
  lindex $list [expr $index%[llength $list]]
}

#
# min: returns minimum of two values
#

proc min {x1 x2} {
  if { $x1 < $x2 } { return $x1 } else { return $x2 }
}

#
# max: returns maximum of two values
#

proc max {x1 x2} {
  if { $x1 > $x2 } { return $x1 } else { return $x2 }
}

#
# max3: returns maximum of three values
#

proc max3 {x1 x2 x3} {
  if { $x1 >= $x2 && $x1 >= $x3 } {
      return $x1  
  } elseif { $x2 >= $x1 && $x2 >= $x3 } {
      return $x2
  } else {return $x3}
}

#
# vectorsum: returns sum of all elements in a vector
#

proc vectorsum {vector} {
  upvar $vector v
  set r 0.0
  foreach elem $v {
    set r [expr $r+$elem]
  }
  return $r
}

#
# vectormax: returns maximum of all elements in a vector
#

proc vectormax {vector} {
  upvar $vector v
  set r [lindex $v 0]
  foreach elem $v {
    if { $elem > $r } { set r $elem }
  }
  return $r
}

#
# vectorop: apply scalar operation to vector
#

proc vectorop {vector op val} {
  upvar $vector v
  set r ""
  foreach elem $v {
    lappend r [format "%.1f" [expr ${elem}${op}${val}]]
  }
  return $r
}

#
# vectorpercent: scale elements of vector according to max and scale
#

proc vectorpercent {vector max scale} {
  upvar $vector v

  if { $max == 0 } { return $v }
  set r ""
  foreach elem $v {
    lappend r [format "%.2f" [expr (double($elem)/double($max))*$scale]]
  }
  return $r
}

#
# numberInput: widget for input of 1 integer number
#
#         win: widget name
#       label: optional label to display
#     varname: name of global variable which holds input value
#

proc numberInput {win label varname} {
  global $varname

  frame $win
  if { $label != "" } {
    label $win.t -text $label
    pack $win.t -side top -fill x
  }
  button $win.b1 -text "<<" -width 3 -command "incr $varname -5"
  button $win.b2 -text "<"  -width 3 -command "incr $varname -1"
  entry $win.e -textvariable $varname -width 5 -relief sunken
  button $win.b3 -text ">"  -width 3 -command "incr $varname +1"
  button $win.b4 -text ">>" -width 3 -command "incr $varname +5"
  pack $win.b1 $win.b2 $win.e $win.b3 $win.b4 -side left -padx 10 -pady 10
}

set pr_sel_tag -99;  # currently selected function

#
# bargraph: horizontal bargraph widget
#
#         win: pathname for bargraph
#     bgtitle: title for bargraph
# rightlabels: list of labels
#     percent: list of percent values
#      values: list of values
#        tags: list of corresponding Sage++ id of functions
#       nodes: list of corresponding node identifiers
#              (if nonempty used instead of tags)
#        mode: what to show as labels on the right side: per / val / none
#

proc bargraph {win bgtitle rightlabels percent values tags {nodes {}} {mode per}} {
  global tagcol tagstip
  global pr_sel_tag
  global racy_progfile

  if [DEBUG] {
      puts " "
      puts "bargraph: "
      puts "  rightlabels: $rightlabels"
      puts "  percent: $percent"
      puts "  values: $values"
      puts "  tags: $tags"
      puts "  nodes: $nodes"
      puts "  mode: $mode"
  }

  set num [llength $rightlabels];  # number of bars

  # -- create or reset canvas
  if { [winfo exists $win] } {
    $win delete all
  } else {
    canvas $win -background white
  }

  # -- scale percentage vector, so that it uses most of the display
  set pmax [vectormax percent]
  if { $pmax < 0.1 } {
    set pers [vectorop percent * 750]
  } elseif { $pmax < 1 } {
    set pers [vectorop percent * 75]
  } elseif { $pmax < 10 } {
    set pers [vectorop percent * 7.5]
  } elseif { $pmax < 25 } {
    set pers [vectorop percent * 3]
  } elseif { $pmax < 50 } {
    set pers [vectorop percent * 1.5]
  } else {
    set pers $percent
  }

#----------------------------------------------------------------------------
# display function name on the left side of bar graph
#----------------------------------------------------------------------------  
#  set max_llabel_width 0
#  for {set i 0} {$i<$num} {incr i} {
#    # -- create left labels; highlight selected functions in red
#    set t [lindex $tags $i]
#    if { $t == $pr_sel_tag } {
#      set ll_obj($i) [$win create text 0 [expr 40+$i*20] \
#               -text "[lindex $leftlabels $i]" -anchor e -fill red \
#               -font -Adobe-Helvetica-Bold-R-Normal--*-120-*]
#    } else {
#      set ll_obj($i) [$win create text 0 [expr 40+$i*20] \
#               -text "[lindex $leftlabels $i]" -anchor e \
#               -font -Adobe-Helvetica-Bold-R-Normal--*-120-*]
#    }
#    set bbox [$win bbox $ll_obj($i)]
#    set llabel_width [expr [lindex $bbox 2] - [lindex $bbox 0]]
#    if {$llabel_width > $max_llabel_width} {set max_llabel_width $llabel_width}
#
#    # -- setup bindings for clicking on left labels
#    if { $nodes == {} } {
#      $win bind $ll_obj($i) <Button-1> "showFuncgraph $t"
#      # showFuncTag is not implemented by racy
#      # $win bind $ll_obj($i) <Button-2> \
#      #     "PM_GlobalSelect $racy_progfile global_showFuncTag $t"
#      $win bind $ll_obj($i) <Button-3> \
#	  "PM_GlobalSelect $racy_progfile global_selectFuncTag $t"
#    } else {
#      $win bind $ll_obj($i) <Button-1> \
#	  "showBargraph [lindex $nodes $i] {[lindex $leftlabels $i]}"
#      $win bind $ll_obj($i) <Button-2> \
#	  "showText [lindex $nodes $i] {[lindex $leftlabels $i]}"
#    }
#  }
#
#  # -- create title
#  set title [$win create text 0 20 \
#		 -text "$bgtitle" -anchor e \
#		 -font -Adobe-Helvetica-Bold-R-Normal--*-120-*]
#  set bbox [$win bbox $title]
#  set title_width [expr [lindex $bbox 2] - [lindex $bbox 0]]
#  set max_l_width [max $title_width $max_llabel_width]
#  $win coords $title [expr $max_l_width+5] 20
#
#  set max_r_width 0
#  for {set i 0} {$i<$num} {incr i} {
#    # -- position left labels
#    $win coords $ll_obj($i) [expr $max_l_width+5] [expr 40+$i*20]
#
#    # -- display bars and right labels according to mode
#    set p [lindex $pers $i]
#    if { $p > 0 } {
#      set t [lindex $tags $i]
#      $win create rectangle \
#	  [expr $max_l_width+10]\
#	  [expr 31+$i*20] \
#	  [expr $max_l_width+10+$p*2] \
#	  [expr 49+$i*20] \
#	  -fill $tagcol($t) -stipple $tagstip($t)
#      switch -exact $mode {
#	  per   { set rl_obj [$win create text \
#				  [expr $max_l_width+15+$p*2] \
#				  [expr 40+$i*20] \
#				  -text "[lindex $percent $i]%" -anchor w \
#				  -font -Adobe-Helvetica-Bold-R-Normal--*-120-*]}
#
#	  val   { set rl_obj [$win create text \
#				  [expr $max_l_width+15+$p*2] \
#				  [expr 40+$i*20] \
#				  -text "[wraplindex $values $i]" -anchor w \
#				  -font -Adobe-Helvetica-Bold-R-Normal--*-120-*]}
#
#          none  {}
#      }
#      set bbox [$win bbox $rl_obj]
#      set r_width [expr $p*2 + [lindex $bbox 2] - [lindex $bbox 0]]
#      if {$r_width > $max_r_width} {set max_r_width $r_width}
#    }
#  }
#
#  set w [expr $max_l_width + $max_r_width + 20]
#  set h [expr ($num+2)*20];       # height of window
#  $win configure -scrollregion [list 0 0 $w $h]
#  $win xview moveto 1; # to shift view of canvas to right


#----------------------------------------------------------------------------
# display function name on the right side of bar graph
#----------------------------------------------------------------------------
  set max_llabel_width 0
  set max_bar_width 0
  for {set i 0} {$i<$num} {incr i} {
    # -- display leftt labels according to mode
  if [DEBUG] {
      puts "  pers: $pers"
  }
    set p [lindex $pers $i]
    if { $p > 0 } {
      # -- set the width of bar graph
      set bar_width($i) [expr $p*2]
      if {$bar_width($i) > $max_bar_width} {set max_bar_width $bar_width($i)}
      # -- create left label according to its mode
      switch -exact $mode {
	  per   { set ll_obj($i) [$win create text 0 [expr 40+$i*20] \
				  -text "[lindex $percent $i]%" -anchor e \
				  -font -Adobe-Helvetica-Bold-R-Normal--*-120-*]}

	  val   { set ll_obj($i) [$win create text 0 [expr 40+$i*20] \
				  -text "[wraplindex $values $i]" -anchor e \
				  -font -Adobe-Helvetica-Bold-R-Normal--*-120-*]}

          none  {}
      }
      set bbox [$win bbox $ll_obj($i)]
      set llabel_width [expr [lindex $bbox 2] - [lindex $bbox 0]]
      if {$llabel_width > $max_llabel_width} {set max_llabel_width $llabel_width}
    }
  }
  set max_llabel_width [expr $max_llabel_width + 10]
  set max_rlabel_width 0
  for {set i 0} {$i<$num} {incr i} {
    # -- create right labels/function names; highlight selected functions in red
    set t [lindex $tags $i]
    set p [lindex $pers $i]
    if { $t == $pr_sel_tag } {
      set rl_obj($i) [$win create text 0 [expr 40+$i*20] \
                             -text "[lindex $rightlabels $i]" -anchor w -fill red \
                             -font -Adobe-Helvetica-Bold-R-Normal--*-120-*]
    } else {
      set rl_obj($i) [$win create text 0 [expr 40+$i*20] \
                             -text "[lindex $rightlabels $i]" -anchor w \
                             -font -Adobe-Helvetica-Bold-R-Normal--*-120-*]
    }
    set bbox [$win bbox $rl_obj($i)]
    set rlabel_width [expr [lindex $bbox 2] - [lindex $bbox 0]]
    if {$rlabel_width > $max_rlabel_width} {set max_rlabel_width $rlabel_width}

    # -- setup bindings for clicking on right labels
    if { $nodes == {} } {
      $win bind $rl_obj($i) <Button-1> "showFuncgraph $t"
      # showFuncTag is not implemented by racy
      # $win bind $ll_obj($i) <Button-2> \
      #     "PM_GlobalSelect $racy_progfile global_showFuncTag $t"
      $win bind $rl_obj($i) <Button-3> \
	  "PM_GlobalSelect $racy_progfile global_selectFuncTag $t"
    } else {
      $win bind $rl_obj($i) <Button-1> \
	  "showBargraph [lindex $nodes $i] {[lindex $rightlabels $i]}"
      $win bind $rl_obj($i) <Button-2> \
	  "showText [lindex $nodes $i] {[lindex $rightlabels $i]}"
    }
  }

  # -- create title
  set title [$win create text 0 20 \
		 -text "$bgtitle" -anchor w \
		 -font -Adobe-Helvetica-Bold-R-Normal--*-120-*]
  set bbox [$win bbox $title]
  set title_width [expr [lindex $bbox 2] - [lindex $bbox 0]]
  set max_r_width [max $title_width $max_rlabel_width]
  $win coords $title [expr $max_llabel_width + $max_bar_width + 5] 20

  # -- display bars and left labels according to mode
  for {set i 0} {$i<$num} {incr i} {
    set p [lindex $pers $i]
    if { $p > 0 } {
	# -- position left labels
	$win coords $ll_obj($i) [expr $max_llabel_width+$max_bar_width-$bar_width($i)-5] \
                            [expr 40+$i*20]

	# -- display bars and right labels according to mode
    
	set t [lindex $tags $i]
	$win create rectangle \
	  [expr $max_llabel_width+$max_bar_width-1-$bar_width($i)]  \
	  [expr 31+$i*20] \
	  [expr $max_llabel_width+$max_bar_width] \
	  [expr 49+$i*20] \
	  -fill $tagcol($t) -stipple $tagstip($t)
    }
    # -- position right labels
    $win coords $rl_obj($i) [expr $max_llabel_width + $max_bar_width + 5] \
                            [expr 40+$i*20]
  }

  set w [expr $max_llabel_width + $max_bar_width + $max_rlabel_width + 40]
  set h [expr ($num+2)*20];       # height of window
  $win configure -scrollregion [list 0 0 $w $h]
  #$win xview moveto 1; # to shift view of canvas to right

}

#
# selectFuncTag: implementation of global feature "selectFunction" for racy
#                display all instances of selected function in red and
#                show function profile, if possible
#
#           progfile: had better be == racy_progfile
#           tag: id of function to select
#

proc selectFuncTag {progfile tag} {
  global pr_sel_tag \
	  tagname racy_progfile

  if {$progfile != $racy_progfile} {
      return
  }

  set pr_sel_tag $tag
  redraw {} {}
  if [ info exists tagname($tag) ] { showFuncgraph $tag }
}

#
# showText: display text node profile window
#
#     node: id of node
#     name: full nameof node
#

proc showText {node name} {
  global myself
  global data
  global textorder1 textorder2
  global proforder profvalue profmode profunit

  if [DEBUG] {
    puts "\nshowText:"
    puts "  node: $node"
    puts "  name: $name"
  }

  if { ! [winfo exists .text$node] } {
    set textorder1($node) $profvalue(all)
    #if { $proforder(all) == "glob" } {
    #  set textorder2($node) $proforder(all)
    #} else {
    #  set textorder2($node) $node
    #}
    if { $proforder(all) == "all" } {
      set textorder2($node) $node
    } else {
      set textorder2($node) $proforder(all)
    }

    #toplevel .text$node
    #wm title .text$node "$name profile"
    set win2 [ toplevel .text$node]
    wm minsize $win2 1000 250
    wm title .text$node "$name profile"

    frame .text$node.mbar -relief raised -borderwidth 2

    menubutton .text$node.mbar.b1 -text File -menu .text$node.mbar.b1.m1 \
                                  -underline 0
    menu .text$node.mbar.b1.m1
    .text$node.mbar.b1.m1 add command -label "Close"  -underline 0 \
                    -command "destroy .text$node"

    menubutton .text$node.mbar.b2 -text Order -menu .text$node.mbar.b2.m1 \
                                  -underline 0
    menu .text$node.mbar.b2.m1
    submenu2 .text$node.mbar.b2.m1 textorder1 redrawText $node {$name}
    .text$node.mbar.b2.m1 add separator
    submenu3 .text$node.mbar.b2.m1 textorder2 redrawText $node {$name}

    menubutton .text$node.mbar.b3 -text Help -menu .text$node.mbar.b3.m1 \
                                  -underline 0
    menu .text$node.mbar.b3.m1
    if [ALONE] {
      .text$node.mbar.b3.m1 add command -label "on text node profile" \
	  -underline 3 \
	  -command "showHelp $myself 1.2.3-text 1"
    } else {
      .text$node.mbar.b3.m1 add command -label "on text node profile" \
	  -underline 3 \
	  -command "xsend tau \[list showHelp $myself 1.2.3-text 1\]"
    } 

    pack .text$node.mbar.b1 .text$node.mbar.b2 -side left -padx 5
    pack .text$node.mbar.b3 -side right -padx 5

    redrawText $node {$name}

    scrollbar .text$node.sv -orient vert -relief sunken \
	-command ".text$node.text yview"
    .text$node.text configure -yscrollcommand ".text$node.sv set"

    scrollbar .text$node.sh -orient horiz -relief sunken \
	-command ".text$node.text xview"
    .text$node.text configure -xscrollcommand ".text$node.sh set"

    button .text$node.but -text "close" -command "destroy .text$node"

    pack .text$node.mbar -side top -fill x
    pack .text$node.but -side bottom -fill x
    pack .text$node.sh -side bottom -fill x
    pack .text$node.sv -side right -fill y
    #pack .text$node.text -side left -ipadx 10 -ipady 10 -anchor center
    pack .text$node.text -side left -fill both -expand yes
  } else {
    raise .text$node
  }
}

#
# redrawText: create or update contents of text node profile window
#
#       node: id of node
#       name: full nameof node
#

proc redrawText {node name} {
  global data
  global tagname tagcol tagfcol
  global textorder1 textorder2
  global pr_sel_tag
  global racy_progfile alltags
  global newver
  global stddev
  global heading
  global mheading
  global column


  if [DEBUG] {
    puts "\nredrawText:"
    puts "  node: $node"
    puts "  name: $name"
    puts "  newver: $newver"
  }

  set win .text$node.text
  set v $textorder1($node)
  #set tags $data($textorder2($node),${v}tags)
  #set tags $alltags
  if {$textorder2($node) == "glob"} {
    set tags $alltags
  } else {
    set tags $data($textorder2($node),${v}tags)
  }

  # -- create or reset text widget
  if { [winfo exists $win] } {
    $win configure -state normal
    $win delete 1.0 end
    $win configure -height [min 35 [expr 3+[llength $tags]]]
  } else {
      text $win -height [min 35 [expr 3+[llength $tags]]] \
	  -background white -foreground black -padx 5 -pady 5
  }

  # -- redraw text area

  #if { $newver == 1 } {
  #  $win insert end \
  #   "---------------------------------------------------------------------------------------------------------------\n"
  #  foreach line $heading {
  #     $win insert end "$line\n"
  #  }
  #  $win insert end \
  #   "---------------------------------------------------------------------------------------------------------------\n"
  #} else {
  #$win insert end \
  #  "---------------------------------------------------------------------------------------------------\n"
  #$win insert end \
  #  "%time         msec   total msec    #call   #subrs  usec/call name\n"
  #$win insert end \
  #  "---------------------------------------------------------------------------------------------------\n"
  #}
  if { $newver == 1 } {
    $win insert end \
     "---------------------------------------------------------------------------------------------------------------\n"
    if { $node == "m" && $stddev == 1 } {
	foreach line $mheading {
	   $win insert end "$line\n"
	   set column [expr [llength $line]-1]
	}
    } else {
	foreach line $heading {
	   $win insert end "$line\n"
	   set column [expr [llength $line]-1]
	}
    }
    $win insert end \
      "---------------------------------------------------------------------------------------------------------------\n"
  } else {
    $win insert end \
     "---------------------------------------------------------------------------------------------------\n"
    $win insert end \
     "%time         msec   total msec    #call   #subrs  usec/call name\n"
    $win insert end \
     "---------------------------------------------------------------------------------------------------\n"
  } 


  set n 3
  set txt [order $textorder2($node) $node $tags $v text {}]

  foreach line $txt {
    incr n

    set t [lindex $tags [expr $n-4]]
    if { $line == "" } {
	if { $column == 8 } {
	    $win insert end \
		  "                                                                        $tagname($t)\n"
	} else {
	    $win insert end \
	       "                                                             $tagname($t)\n"
	}
    } else {
      $win insert end "$line\n"


      if { $t == $pr_sel_tag } {
	  if { $column == 8} {
	      $win tag add seltag $n.0 $n.71
	  } else { 
	      $win tag add seltag $n.0 $n.61
	  }
        $win tag configure seltag -foreground red
      }
    }
    #$win tag add t$n $n.61 "$n.61 lineend"
    if { $newver == 1 } {
	if { $column == 8} {
	    $win tag add t$n $n.72 "$n.72 lineend"
	} else {
	    $win tag add t$n $n.61 "$n.61 lineend"
	}
    } else {
        $win tag add t$n $n.61 "$n.61 lineend"   
    }
    if { [winfo depth .] == 2 } {
      $win tag configure t$n -background white -foreground black
    } else {
      $win tag configure t$n -background $tagcol($t) -foreground $tagfcol($t)
    }
    $win tag bind t$n <Button-1> "showFuncgraph $t"
    # showFuncTag is not implemented by racy
    # $win tag bind t$n <Button-2> "PM_GlobalSelect $racy_progfile global_showFuncTag $t"
    $win tag bind t$n <Button-3> "PM_GlobalSelect $racy_progfile global_selectFuncTag $t"
  }

  $win configure -wrap none
  $win configure -state disabled
}

#
# order: return field of a data item in the same order it appear in
#        other data item. order is determined by "tags" field
#
# source: id of data which values determine sorting
# target: id of data to order
#  value: excl or incl
#  field: field of data to be sorted
#     NA: NULL value for this field
#

proc order {source target tags value field NA} {
  global data alltags
  if [DEBUG] {
    puts "\norder:"
    puts "  source: $source"
    puts "  target: $target"
    puts "  tags: $tags"
    puts "  value: $value"
    puts "  field: $field"
    puts "  NA: $NA"
  }

  if { $source == $target } {
    return $data($target,$value$field)
  } else {
    set result ""
    #foreach t $data($source,${value}tags)
    #foreach t $alltags
    foreach t $tags {
      set i [lsearch -exact $data($target,${value}tags) $t]
      if { $i == -1 } {
        lappend result $NA
      } else {
        lappend result [lindex $data($target,$value$field) $i]
      }
    }
    return $result
  }
}

#
# specialOrder: basically like order above, but only for numeric fields
#               corrects "other" entry in field
#

proc specialOrder {source target tags value field} {
  global data alltags

  if { $source == $target } {
    return $data($target,$value$field)
  } else {
    set result ""
    set na [vectorsum data($target,$value$field)]; #sum of all values
    #foreach t $data($source,${value}tags) 
    #foreach t $alltags 
    foreach t $tags {
      set i [lsearch -exact $data($target,${value}tags) $t]
      if { $i == -1 } {
        lappend result 0.0
      } else {
        set x [lindex $data($target,$value$field) $i]
        if { $t == -1 } {
          # -- tag -1 is the "others" element; always last one
          # -- use "notused" values instead of precalculated ones
          lappend result [format "%4.2f" $na]
        } else {
          set na [expr $na-$x]; #value used; subtract it from "notused" value
          lappend result $x
        }
      }
      }
    return $result
  }
}

#
# submenu2: create "Value" menu
#
#      menu: pathname of menu
#       var: basename of global variable to store selection
#      func: function to call for redraw
# node,name: arguments to func
#

proc submenu2 {menu var func node name} {
  $menu add radiobutton -label "microseconds" \
                 -underline 0 -variable ${var}($node) -value excl \
                 -command "$func $node {$name}"
  $menu add radiobutton -label "total microseconds" \
                 -underline 0 -variable ${var}($node) -value incl \
                 -command "$func $node {$name}"
}

#
# submenu3: create "Order" menu
#
#      menu: pathname of menu
#       var: basename of global variable to store selection
#      func: function to call for redraw
# node,name: arguments to func
#

proc submenu3 {menu var func node name} {
  global funodes fulabels

  $menu add radiobutton -label "global" \
            -underline 0 -variable ${var}($node) -value glob \
            -command "$func $node {$name}"
  $menu add radiobutton -label "decreasing" \
            -underline 0 -variable ${var}($node) -value $node \
            -command "$func $node {$name}"

  $menu add cascade -label "node" \
            -underline 0 -menu $menu.1

  # -- create "node" submenu
  menu $menu.1
  set i 0
  foreach n $funodes {
    if { $n != {} } {
      $menu.1 add radiobutton -label [lindex $fulabels $i] \
                  -variable ${var}($node) -value $n \
	          -command "$func {$node} {$name}"
    } else {
      $menu.1 add separator
    }
    incr i
  }
}

#
# submenu4: create function profile "Mode" menu
#
#      menu: pathname of menu
#       var: basename of global variable to store selection
#      func: function to call for redraw
# node,name: arguments to func
#

proc submenu4 {menu var func node name} {
  $menu add radiobutton -label "percent" \
            -underline 0 -variable ${var}($node) -value per \
            -command "$func $node {$name}"
  $menu add radiobutton -label "value" \
            -underline 0 -variable ${var}($node) -value val \
            -command "$func $node {$name}"
}

#
# submenu5: create "Units" menu
#
#      menu: pathname of menu
#       var: basename of global variable to store selection
#      func: function to call for redraw
# node,name: arguments to func
#

proc submenu5 {menu var func node name} {
  $menu add radiobutton -label "microseconds" \
            -underline 0 -variable ${var}($node) -value 1.0 \
            -command "$func $node {$name}"
  $menu add radiobutton -label "milliseconds" \
            -underline 0 -variable ${var}($node) -value 1.0e3 \
            -command "$func $node {$name}"
  $menu add radiobutton -label "seconds" \
            -underline 0 -variable ${var}($node) -value 1.0e6 \
            -command "$func $node {$name}"
}

#
# submenu6: create collection profile "Mode" menu
#
#      menu: pathname of menu
#       var: basename of global variable to store selection
#      func: function to call for redraw
# node,name: arguments to func
#

proc submenu6 {menu var func node name} {
  $menu add radiobutton -label "local percent" \
            -underline 0 -variable ${var}($node) -value lper \
            -command "$func $node {$name}"
  $menu add radiobutton -label "global percent" \
            -underline 0 -variable ${var}($node) -value gper \
            -command "$func $node {$name}"
  $menu add radiobutton -label "value" \
            -underline 0 -variable ${var}($node) -value val \
            -command "$func $node {$name}"
}

#
# submenu6a: create aggregate profile "Mode" menu
#
#      menu: pathname of menu
#       var: basename of global variable to store selection
#      func: function to call for redraw
# node,name: arguments to func
#

proc submenu6a {menu var func node name} {
  $menu add radiobutton -label "percent" \
            -underline 0 -variable ${var}($node) -value lper \
            -command "$func $node {$name}"
  $menu add radiobutton -label "value" \
            -underline 0 -variable ${var}($node) -value val \
            -command "$func $node {$name}"
}

#
# bargraphmenu: create menubar for node profile windows
#
#       parent: pathname for parent of menubar
#       prefix: prefix for global variables to store selections
#         func: function to call for redraw
#    node,name: arguments to func
#

proc bargraphmenu {parent prefix func node name} {
  global myself

  set menubar $parent.mbar
  frame $menubar -relief raised -borderwidth 2
  menubutton $menubar.b1 -text File -menu $menubar.b1.m1 -underline 0
  menu $menubar.b1.m1
  $menubar.b1.m1 add command -label "Print" -underline 0 \
                    -command "printCanvas .bar${node}.bar node.$node"
  $menubar.b1.m1 add separator
  $menubar.b1.m1 add command -label "Close" -underline 0 \
                    -command "destroy .bar$node"

  menubutton $menubar.b2 -text Value -menu $menubar.b2.m1 -underline 0
  menu $menubar.b2.m1
  submenu2 $menubar.b2.m1 ${prefix}value $func $node {$name}

  menubutton $menubar.b3 -text Order -menu $menubar.b3.m1 -underline 0
  menu $menubar.b3.m1
  submenu3 $menubar.b3.m1 ${prefix}order $func $node {$name}

  menubutton $menubar.b4 -text Mode -menu $menubar.b4.m1 -underline 0
  menu $menubar.b4.m1
  submenu4 $menubar.b4.m1 ${prefix}mode $func $node {$name}

  menubutton $menubar.b5 -text Units -menu $menubar.b5.m1 -underline 0
  menu $menubar.b5.m1
  submenu5 $menubar.b5.m1 ${prefix}unit $func $node {$name}

  menubutton $menubar.b6 -text Help -menu $menubar.b6.m1 -underline 0
  menu $menubar.b6.m1
  if [ALONE] {
    $menubar.b6.m1 add command -label "on node profile" -underline 3 \
	              -command "showHelp $myself 1.2.2-node 1"
  } else {
    $menubar.b6.m1 add command -label "on node profile" -underline 3 \
	              -command "xsend tau \[list showHelp $myself 1.2.2-node 1\]"
  }

  pack $menubar.b1 $menubar.b2 $menubar.b3 $menubar.b4 $menubar.b5\
       -side left -fill x -padx 5
  pack $menubar.b6 -side right -fill x -padx 5
}

#
# showBargraph: display node profile window
#
#         node: id of node
#         name: full nameof node
#

proc showBargraph {node name} {
  global barorder barmode barvalue barunit
  global proforder profvalue profmode profunit

  if { ! [winfo exists .bar$node] } {
    #if { $proforder(all) == "glob" } {
    #  set barorder($node) $proforder(all)
    #} else {
    #  set barorder($node) $node
    #}
    if { $proforder(all) == "all" } {
      set barorder($node) $node
    } else {
      set barorder($node) $proforder(all)
    }
    set barmode($node) $profmode(all)
    set barvalue($node) $profvalue(all)
    set barunit($node) $profunit(all)

    set win [toplevel .bar$node]
    wm title $win "$name profile"
    #wm minsize $win 250 250
    wm minsize $win 500 250

    bargraphmenu .bar$node bar redrawBargraph $node {$name}
    redrawBargraph $node {$name}

    scrollbar .bar$node.sv -orient vert -relief sunken \
	-command ".bar$node.bar yview"
    .bar$node.bar configure -yscrollcommand ".bar$node.sv set"

    scrollbar .bar$node.sh -orient horiz -relief sunken \
	-command ".bar$node.bar xview"
    .bar$node.bar configure -xscrollcommand ".bar$node.sh set"

    button .bar$node.but -text "close" -command "destroy $win"

    pack .bar$node.mbar -side top -fill x
    pack .bar$node.but -side bottom -fill x
    pack .bar$node.sh -side bottom -fill x
    pack .bar$node.sv -side right -fill y
    pack .bar$node.bar -side left -fill both -expand yes
  } else {
    raise .bar$node
  }
}

#
# redrawBargraph: create or update contents of node profile window
#
#           node: id of node
#           name: full nameof node
#

proc redrawBargraph {node name} {
  global data
  global barorder barmode barvalue barunit
  global tagname alltags
    
  set v $barvalue($node)
  #foreach t $data($barorder($node),${v}tags) 
  #foreach t $alltags {
  #  lappend n $tagname($t)
  #}
  if {$barorder($node) == "glob"} {
      set currtags $alltags
  } else {
      set currtags $data($barorder($node),${v}tags)
  }
  foreach t $currtags {
      lappend n $tagname($t)
  }
  set u [specialOrder $barorder($node) $node $currtags $v usecs]

  # -- add title within window for bargraph
  # -- $name is not always available, so must use $node
  switch $node {
      t       {set title "total"}
      m       {set title "mean"}
#      <       {set title "min"}
#      >       {set title "max"}
      default {
	  set title ""
	  set title [append $title "n,c,t " $node]
      }
  }
  bargraph .bar$node.bar $title $n \
                         [specialOrder $barorder($node) $node $currtags $v percent] \
                         [vectorop u / $barunit($node)] \
                         [order $barorder($node) $node $currtags $v tags -2] \
                         {} $barmode($node)
  if { $barmode($node) == "val" } {
    .bar$node.mbar.b5 configure -state normal
  } else {
    .bar$node.mbar.b5 configure -state disabled
  }
}

#
# multiFuncgraph: create summary function bargraphs
#
#            win: pathname for canvas
#     leftlabels: list of labels for left side of bargraphs
#          nodes: list of node ids
#       percents: list of list of percentage values
#           tags: list of tags 
#

proc multiFuncgraph {win leftlabels nodes percents tags} {
  global tagcol tagstip
  global minheight
  global racy_progfile

  if [DEBUG] {
	puts "multiFuncgraph:"
	puts "  leftlabels: $leftlabels"
	puts "  nodes: $nodes"
	puts "  percents: $percents"
	puts "  tags: $tags"
  }

  set num [llength $leftlabels];  #number of bars
  set h [expr ($num+1)*20];       #height of the display

  # -- create or reset canvas
  if { [winfo exists $win] } {
    $win delete all
    $win configure -height [min 420 $minheight]
  } else {
    canvas $win -width 300 -height [min 420 $minheight] -background white
  }
  bind $win <2> "$win scan mark %x %y"
  bind $win <B2-Motion> "$win scan dragto %x %y"
  $win configure -scrollregion [list 0 0 300 $h]

  set max_rlabel_width 0 ;# previously constant at 60
  for {set i 0} {$i<$num} {incr i} {
      # -- draw left labels
      if {[lindex $leftlabels $i] != {}} {
	  set rl_obj($i) [$win create text 0 [expr 20+$i*20] \
		  -text "[lindex $leftlabels $i]" -anchor e \
		  -font -Adobe-Helvetica-Bold-R-Normal--*-120-*]
	  set bbox [$win bbox $rl_obj($i)]
	  set lwidth [expr [lindex $bbox 2] - [lindex $bbox 0]]
	  if {$lwidth > $max_rlabel_width} {set max_rlabel_width $lwidth}
	  
	  # -- setup bindings for left labels, if necessary
 	  $win bind $rl_obj($i) <Button-1> \
		  "showBargraph [lindex $nodes $i] {[lindex $leftlabels $i]}"
	  $win bind $rl_obj($i) <Button-2> \
		  "showText [lindex $nodes $i] {[lindex $leftlabels $i]}"
      }
  }

  for {set i 0} {$i<$num} {incr i} {
    
    # -- move the right labels into place
    if {[lindex $leftlabels $i] != {}} {
      $win coords $rl_obj($i) \
	      [expr $max_rlabel_width + 5] [expr 20+$i*20]
    }
      
    # -- display multibar, and create bindings for each subbar
    set p [lindex $percents $i]
    set sum 0.0
    for {set j 0} {$j<[llength $p]} {incr j} {
      #if { [lindex $p $j] > 0 } {
      #  set obj [$win create rectangle \
      #		[expr $max_rlabel_width+10+$sum*2] \
      #		[expr 29+$i*20] \
      #		[expr $max_rlabel_width+10+($sum+[lindex $p $j])*2] \
      #		[expr 11+$i*20]\
      #		-fill $tagcol([lindex $tags $j]) \
      #		-stipple $tagstip([lindex $tags $j])]
      #  set t [lindex $tags $j]
      #  $win bind $obj <Button-1> "showFuncgraph $t"
      #	# showFuncTag is not implemented by racy
      #  # $win bind $obj <Button-2> "PM_GlobalSelect $racy_progfile global_showFuncTag $t"
      #  $win bind $obj <Button-3> "PM_GlobalSelect $racy_progfile global_selectFuncTag $t"
      #  set sum [expr $sum+[lindex $p $j]]
      # }

      #fix the jagged edge of multiFuncgraph
      if { [lindex $p $j] > 0 } {
        set obj [$win create rectangle \
		[expr $max_rlabel_width+10+$sum*2] \
		[expr 29+$i*20] \
		[expr $max_rlabel_width+11+($sum+[lindex $p $j])*2] \
		[expr 11+$i*20]\
		-fill $tagcol([lindex $tags $j]) \
		-stipple $tagstip([lindex $tags $j])]
        set t [lindex $tags $j]
        $win bind $obj <Button-1> "showFuncgraph $t"
	# showFuncTag is not implemented by racy
        # $win bind $obj <Button-2> "PM_GlobalSelect $racy_progfile global_showFuncTag $t"
        $win bind $obj <Button-3> "PM_GlobalSelect $racy_progfile global_selectFuncTag $t"
        set sum [expr $sum+[lindex $p $j]]
      }
    }
  } 

  set w [expr $max_rlabel_width + 230]
  $win configure -width $w
  $win configure -scrollregion [list 0 0 $w $h]
}

#
# multiCollgraph: create collection profile bargraphs
#
#            win: pathname for canvas
#     leftlabels: list of labels for left side of bargraphs
#         idents: list of collection ids
#       percents: list of list of percentage values
#         values: list of values
#           mode: display mode: none or lper or gper
#          scale: scale factor for bars
#

proc multiCollgraph {win leftlabels idents percents \
	{values {}} {mode none} {scale 1.0}} {
  global minheight
  global TAUDIR

  if [DEBUG] {
	puts "multiCollgraph:"
	puts "  leftlabels: $leftlabels"
	puts "  idents: $idents"
	puts "  percents: $percents"
	puts "  values: $values"
	puts "  mode: $mode"
	puts "  scale: $scale"
  }

  if { [winfo depth .] == 2 } {
    set cols {black black}
    set stipps [list @$TAUDIR/xbm/gray3.bm {}]
  } else {
    set cols {green red}
    set stipps [list {} {}]
  }
  set num [llength $leftlabels];           #number of bars (lines)
  set h [expr ($num+1)*20];                #screen height
  set nbar [llength [lindex $percents 0]]; #number of subbars per bar

  # -- screen width different with and without right side labels
  if { $mode == "none" } {
    set w 300
  } else {
    set w [expr 300+$nbar*60]
  }

  # -- create or update drawing canvas
  if { [winfo exists $win] } {
    $win delete all
    $win configure -height [min 420 $minheight]
  } else {
    canvas $win -width $w -height [min 420 $minheight] -background white
    bind $win <2> "$win scan mark %x %y"
    bind $win <B2-Motion> "$win scan dragto %x %y"
  }
  #$win configure -scrollregion [list 0 0 $w $h]

    set max_rlabel_width 0 ;# previously constant at 60
    for {set i 0} {$i<$num} {incr i} {
	# -- draw left labels
	if {[lindex $leftlabels $i] != {}} {
	    set rl_obj($i) [$win create text 0 [expr 20+$i*20] \
		    -text "[lindex $leftlabels $i]" -anchor e \
		    -font -Adobe-Helvetica-Bold-R-Normal--*-120-*]
	    set bbox [$win bbox $rl_obj($i)]
	    set lwidth [expr [lindex $bbox 2] - [lindex $bbox 0]]
	    if {$lwidth > $max_rlabel_width} {set max_rlabel_width $lwidth}
	    
	    # -- setup bindings for left labels, if necessary
	    if { $idents != "" } {
		$win bind $rl_obj($i) <Button-1> \
			"showCollgraph [lindex $idents $i]"
	    }
	}
    }

  for {set i 0} {$i<$num} {incr i} {
      
    # -- move the right labels into place
    if {[lindex $leftlabels $i] != {}} {
      $win coords $rl_obj($i) \
	      [expr $max_rlabel_width + 5] [expr 20+$i*20]
    }

    # -- draw bars and right labels
    set p [lindex $percents $i]
    set sum 0.0
    for {set j 0} {$j<$nbar} {incr j} {
      # -- draw subbars, if value > 0
      set sc [expr [lindex $p $j]*$scale]
      if { $sc > 0 } {
        # -- don't draw first bar (total) if mode is global percent, as this
        # -- bar would be *very* big
        if { $i != 0 || $mode == "lper" || $mode == "none" } {
          set obj [$win create rectangle \
		  [expr $max_rlabel_width+10+$sum*2] \
		  [expr 29+$i*20] \
		  [expr $max_rlabel_width+10+($sum+$sc)*2] \
		  [expr 11+$i*20] \
		  -fill [lindex $cols $j] -stipple [lindex $stipps $j]]
	  set sum [expr $sum+$sc]
        }

        # -- draw right labels according to mode
        switch -glob $mode {
          none {}
          [lg]per { $win create text \
		  [expr $max_rlabel_width+280+$j*60] \
		  [expr 20+$i*20] \
                 -text [format "%.2f%%" [lindex $p $j]] -anchor e \
                 -font -Adobe-Helvetica-Bold-R-Normal--*-120-*}
	   val  { $win create text \
		 [expr $max_rlabel_width+280+$j*60] \
		 [expr 20+$i*20] \
                 -text [lindex [lindex $values $i] $j] -anchor e \
                 -font -Adobe-Helvetica-Bold-R-Normal--*-120-*}
        }
      }
    }
  }

  if {$mode == "none"} {
      set w [expr $max_rlabel_width + 280]
  } else {
      set w [expr $max_rlabel_width + 290 + 60*$nbar]
  }      
  $win configure -width $w
  $win configure -scrollregion [list 0 0 $w $h]

}

#
# multiAggrgraph: create aggregate profile bargraphs
#
#            win: pathname for canvas
#     leftlabels: list of labels for left side of bargraphs
#         idents: list of aggregate ids
#       percents: list of list of percentage values
#         values: list of values
#           mode: display mode: none or lper or gper
#          scale: list of scale factor for bars
#        bar_for: bars drawn for "aggr" or "node"
#

proc multiAggrgraph {win leftlabels idents percents \
	{values {}} {mode none} {scale {}}  {bar_for "aggr"}} {
    global minheight evcol evstip evname aggr
    global TAUDIR
    
    if [DEBUG] {
	puts "multiAggrgraph:"
	puts "  leftlabels: $leftlabels"
	puts "  idents: $idents"
	puts "  percents: $percents"
	puts "  values: $values"
	puts "  mode: $mode"
	puts "  scale: $scale"
	puts "  bar_for: $bar_for"
    }

    if {$scale == {}} {
	set fill_scales 1
    } else {
	set fill_scales 0
    }

    set head 25
    set num [llength $leftlabels];           #number of bars (lines)
    set h [expr $head+($num+1)*20];             #screen height

    set maxnbar 0
    for {set i 0} {$i < $num} {incr i} {
	if {$fill_scales} { lappend scale 1.0}
	if {[llength [lindex $percents $i]] > $maxnbar} {
	    set maxnbar [llength [lindex $percents $i]]
	}
    }
    
    # -- screen width different with and without right side labels
    if { $mode == "none" } {
	set w 300
    } else {
	set w 500
    }
    
    # -- create or update drawing canvas
    if { [winfo exists $win] } {
	$win delete all
	$win configure -height [min 420 $minheight]
    } else {
	canvas $win -width $w -height [min 420 $minheight] -background white
	bind $win <2> "$win scan mark %x %y"
	bind $win <B2-Motion> "$win scan dragto %x %y"
    }
    #$win configure -scrollregion [list 0 0 $w $h]
    
    set max_rlabel_width 0 ;# previously constant at 60
    for {set i 0} {$i<$num} {incr i} {
	# -- draw left labels
	if {[lindex $leftlabels $i] != {}} {
	    set rl_obj($i) [$win create text 0 [expr $head+20+$i*20] \
		    -text "[lindex $leftlabels $i]" -anchor e \
		    -font -Adobe-Helvetica-Bold-R-Normal--*-120-*]
	    set bbox [$win bbox $rl_obj($i)]
	    set lwidth [expr [lindex $bbox 2] - [lindex $bbox 0]]
	    if {$lwidth > $max_rlabel_width} {set max_rlabel_width $lwidth}
	    
	    # -- setup bindings for left labels, if necessary
	    if { $bar_for == "aggr" } {
		$win bind $rl_obj($i) <Button-1> \
			"showAggrgraph [lindex $idents $i]"
	    }
	}
    }

    # -- draw the event titles
    if {$mode != "none"} {
	set xpos [expr 220 + $max_rlabel_width] ;#280
	for {set j 0} {$j<$maxnbar} {incr j} {
	    set obj [$win create text -100 5 \
		    -text $evname([lindex $aggr([lindex $idents 0],counter_set) $j]) \
		    -font -Adobe-Helvetica-Bold-R-Normal--*-120-* \
		    -anchor ne]
	    set objrect [$win bbox $obj]
	    set label_width \
		    [expr [lindex $objrect 2] - [lindex $objrect 0]]
	    if {$label_width < 55} {set label_width 55}
	    set label_pos($j) [expr $xpos + $label_width]
	    $win coords $obj [expr $xpos + $label_width] 5
	    set xpos [expr $xpos + $label_width + 5]
	}
    } else {
	set xpos $w
    }

    for {set i 0} {$i<$num} {incr i} {
	set nbar [llength [lindex $percents $i]]; #number of subbars per bar
	# -- move the right labels into place
	if {[lindex $leftlabels $i] != {}} {
	    $win coords $rl_obj($i) \
		    [expr $max_rlabel_width + 5] [expr $head+20+$i*20]
	}

	# -- draw bars and right labels
	set p [lindex $percents $i]
	set sum 0.0
	for {set j 0} {$j<$nbar} {incr j} {
	    # -- draw subbars, if value > 0
	    set sc [lindex $p $j]
	    if {$i != 0} { set sc [expr $sc * [lindex $scale $i]] }
	    if { $sc >= 0 } {
		if {$bar_for == "aggr" } {
		    set col $evcol([lindex $aggr([lindex $idents $i],counter_set) $j])
		    set stip $evstip([lindex $aggr([lindex $idents $i],counter_set) $j])
		} else {
		    # bar_for == "node"
		    set col $evcol([lindex $aggr([lindex $idents 0],counter_set) $j])
		    set stip $evstip([lindex $aggr([lindex $idents 0],counter_set) $j])
		}
		set obj [$win create rectangle \
			[expr $max_rlabel_width+10+$sum*2] \
			[expr $head+29+$i*20] \
			[expr $max_rlabel_width+10+($sum+$sc)*2] \
			[expr $head+11+$i*20] \
                        -fill $col -stipple $stip]
		$win bind $obj <Button-1> "showEventLegend"
		set sum [expr $sum+$sc]
		
		# -- draw right labels according to mode
		switch -glob $mode {
		    none {}
		    [lg]per { $win create text \
			    $label_pos($j) [expr $head+20+$i*20] \
			    -text [format "%.2f%%" [lindex $p $j]] -anchor e \
			    -font -Adobe-Helvetica-Bold-R-Normal--*-120-*}
		    val  { $win create text \
			    $label_pos($j) [expr $head+20+$i*20] \
			    -text [lindex [lindex $values $i] $j] -anchor e \
			    -font -Adobe-Helvetica-Bold-R-Normal--*-120-*}
		}
	    }
	}
    }

    if {$mode == "none"} {
	set w [expr $max_rlabel_width + 10 + 215]
	$win configure -width $w
	$win configure -scrollregion [list 0 0 $w $h]
    } else {
	$win configure -scrollregion [list 0 0 $xpos $h]
    }
}

#
# readProfile: read in profile data by using the "pprof" tool
#
#         dir: directory in which profile data should be looked for
#

set wantnum 100

proc readProfile {} {
    global depfile \
	    alltags \
	    data \
	    nodes \
	    numfunc wantnum \
	    tagname tagcol tagfcol tagstip Colors TextColors Stipples \
	    evcol evstip evname \
	    coll colls \
	    aggr aggrs \
	    newver \
	    stddev \
	    heading \
	    mheading \
	    column \
	    BINDIR REMBINDIR REMSH TAUDIR

    set dir $depfile(dir)
    if {$depfile(host) == "localhost"} {
	#set dir "."
	set file "$dir/profile"
	
	# -- open connection to pprof to read profile data
	#if { ! [file readable $file.ftab] } {
	#    return NOT_OK
	#}
	#set in [open "|$BINDIR/pprof -d -m -n $wantnum -f $file" r]
	if [catch {open "|$BINDIR/pprof -d -m -n $wantnum -f $file" r} in ] {
	    showError "pprof missing from directory $BINDIR"
	    return NOT_OK
	}
    } else {
	set REMBINDIR "$depfile(root)/bin/$depfile(arch)"
	#set in [open "|$REMSH $depfile(host) \
	#       -n \"cd $dir; $REMBINDIR/pprof -d -m -n $wantnum\"" r]
	if [catch {open "|$REMSH $depfile(host) \
	        -n \"cd $dir; $REMBINDIR/pprof -d -m -n $wantnum\"" r} in ] {
	    showError "pprof missing from directory $REMBINDIR"
            return NOT_OK
        }
    }
    set n NA

    # -- read header: 1) name of corresponding depfile
    # --              2) number of profiled functions
    # --              3) column headings, #-line
    if { [gets $in line] <= 0 } {
	showError "environment variables not configured correctly in racy"
	return NOT_OK
    }
    scan $line "%s" depfile(file)
    gets $in line
    scan $line "%d" numfunc
    set f [lindex $line 1]
    if {$f == "templated_functions" || $f == "templated_functions_hw_counters" } {
	set newver 1
	#read line 3, the heading & count number of columns
#	gets $in line
#	set column [expr [llength $line]-1]
#	lappend heading $line

	#check if the profile includes standard deviation
	set two [lindex $line 2]
	set three [lindex $line 3]
	if {$two == "-stddev" || $three == "-stddev"} {
	    set stddev 1
	    gets $in line
	    set column [expr [llength $line]-1]
	    #lappend heading $line
	    #gets $in line
	    #lappend mheading $line
	    if { $column == 0 } {
		lappend heading $line
	    } else {
		#if there is old heading, delete old heading
		set heading {}
		lappend heading $line
	    }
	    gets $in line
	    set length [expr [llength $line]-1] 
	    if { $length == 0 } {
		lappend mheading $line
	    } else {
		set mheading {}
		lappend mheading $line
	    }
	} else {
	    set stddev 0
	    gets $in line
	    set column [expr [llength $line]-1]
	    if { $column == 0 } {
		lappend heading $line
	    } else {
		#if there is old heading, delete old heading
		set heading {}
		lappend heading $line
	    }
	}
    } else {
	set newver 0
	#check if the profile includes standard deviation
	#set two [lindex $line 2]
	#set three [lindex $line 3]
	#if {$two == "-stddev" || $three == "-stddev"} {
	#    set stddev 1
	#} else {
	#    set stddev 0
	#}
    }
   

    # -- init
    set depfile(path) $dir/$depfile(file)
    set tagcol(-1) black
    set evcol(-1) black
    set tagfcol(-1) white
    set tagstip(-1) {}
    set evstip(-1) {}
    set tagname(-1) "-others-"
    set evname(-1) "-others-"
    set tagcol(-2) white
    set evcol(-2) white
    set tagfcol(-2) black
    set tagstip(-2) {}
    set evstip(-2) {}
    set tagname(-2) ""
    set i 0
    set j 0
    set colls ""
    set aggrs ""
    
    # -- read line by line
    while {[gets $in line] >= 0} {
	set node [lindex $line 0]
	if { $node == "coll" } {
	    #scan $line "%s %s %d %d %f %d %f" dummy node no local lper remote rper
	    set node [lindex $line 1]
	    if {![string match "*,*,*" $node]} {
		if {$node != "t" && $node != "m"} {
		    set node "$node,0,0"
		}
	    }
	    set no [lindex $line 2]
	    set coll($no,$node,val) [list [lindex $line 3] [lindex $line 5]]
	    set coll($no,$node,per) [list [lindex $line 4] [lindex $line 6]]
	} elseif { $node == "cinfo" } {
	    #scan $line "%s %s %s %s %d %d %d %d %f %d %f" \
	    #     dummy no type name elem size dim local lper remote rper
	    set no [lindex $line 1]
	    lappend colls $no
	    set coll($no,name) [lindex $line 3]
	    set coll($no,type) [lindex $line 2]
	    set coll($no,info) [lrange $line 4 6]
	    set coll($no,t,val) [list [lindex $line 7] [lindex $line 9]]
	    set coll($no,t,per) [list [lindex $line 8] [lindex $line 10]]
	} elseif {$node == "aggregates"} {
	    # HPC++ Aggregates
	    set node [lindex $line 1]
	    if {![string match "*,*,*" $node]} {
		if {$node != "t" && $node != "m"} {
		    set node "$node,0,0"
		}
	    }
	    set no [lindex $line 2]
	    set num_counters [lindex $line 3]
	    set aggr($no,$node,num_counters) $num_counters
	    set index 4
	    for {set counter 0} \
		    {$counter < $num_counters} \
		    {incr counter; incr index 4} {
		set eid [lindex $line $index]
		set aggr($no,$node,counter,$eid,name) \
			[lindex $line [expr $index + 1]]
		set aggr($no,$node,counter,$eid,value) \
			[lindex $line [expr $index + 2]]
		set aggr($no,$node,counter,$eid,percent) \
			[lindex $line [expr $index + 3]]

		if { ! [info exists evcol($eid)] } {
		    if { [winfo depth .] == 2 } {
			set evcol($eid) black
			set evstip($eid) @$TAUDIR/xbm/[wraplindex $Stipples $j]
		    } else {
			set evcol($eid) [wraplindex $Colors $j]
			set evstip($eid) {}
		    }
		    set evname($eid) [lindex $line [expr $index + 1]]
		    incr j
		}
	    }
	    set aggr($no,$node,container_name) [lindex $line $index]
	    set aggr($no,$node,container_type) [lindex $line [expr $index + 1]]
	    set aggr($no,$node,variable_name)  [lindex $line [expr $index + 2]]
	} elseif {$node == "ainfo"} {
	    # HPC++ Aggregates - Total info
	    set no [lindex $line 1]
	    lappend aggrs $no
	    set aggr($no,container_name) [lindex $line 2]
	    set aggr($no,container_type) [lindex $line 3]
	    set aggr($no,variable_name)  [lindex $line 4]
	    set aggr($no,elem)           [lindex $line 5]
	    set aggr($no,size)           [lindex $line 6]
	    set aggr($no,dim)            [lindex $line 7]
	    if {![info exists aggr($no,counter_set)]} {
		set aggr($no,counter_set)    [list]
		set accumulate_counters 1
	    } else {
		set accumulate_counters 0
	    }

	    set num_counters [lindex $line 8]
	    set aggr($no,num_counters) $num_counters
	    set index 9
	    for {set counter 0} \
		    {$counter < $num_counters} \
		    {incr counter; incr index 4} {
		set eid [lindex $line $index]
		set aggr($no,counter_t,$eid,name) \
			[lindex $line [expr $index + 1]]
		set aggr($no,counter_t,$eid,value) \
			[lindex $line [expr $index + 2]]
		set aggr($no,counter_t,$eid,percent) \
			[lindex $line [expr $index + 3]]
		if {$accumulate_counters} {
		    lappend aggr($no,counter_set) $eid
		}
	    }
	} else {
	    # -- function profile data
	    #scan $line "%s %d %s %s %g %g %f" node tag name mode usec per
	    set tag   [lindex $line 1]
	    set name  [lindex $line 2]
	    set mode  [lindex $line 3]
	    set usec  [lindex $line 4]
	    set per   [lindex $line 5]
	    if {![info exists alltags_tmp($tag)]} {
		set alltags_tmp($tag) 1
		lappend alltags $tag
	    }
	    if {![string match "*,*,*" $node]} {
		if {$node != "t" && $node != "m"} {
		    set node "$node,0,0"
		}
	    }
	    if { $n != $node && $mode == "incl"} {
		set n $node
		lappend nodes $node
	    }
	    
	    lappend data($node,${mode}tags) $tag
	    lappend data($node,${mode}usecs) [format "%8.4G" $usec]
#	    lappend data($node,${mode}usecs) $usec
	    lappend data($node,${mode}percent) [format "%4.2f" $per]
	    
	    if { ! [info exists tagcol($tag)] } {
		if { [winfo depth .] == 2 } {
		    set tagcol($tag) black
		    set tagfcol($tag) white
		    set tagstip($tag) @$TAUDIR/xbm/[wraplindex $Stipples $i]
		} else {
		    set tagcol($tag) [wraplindex $Colors $i]
		    set tagfcol($tag) [wraplindex $TextColors $i]
		    set tagstip($tag) {}
		}
		set tagname($tag) $name
		incr i
	    }

	    gets $in line
	    lappend data($node,${mode}text) $line
	}
    }
#    if [catch {close $in} errmsg] {
#	showError "$readcom: `$errmsg'."
#	return NOT_OK
#    }
    
    if { ! [info exists nodes] } {
	return NOT_OK
    }

    return OK
}

#
# funcgraphmenu: create menubar for function profile windows
#
#        parent: pathname for parent of menubar
#        prefix: prefix for global variables to store selections
#          func: function to call for redraw
#     node,name: arguments to func
#

proc funcgraphmenu {parent prefix func node name} {
  global myself

  set menubar $parent.mbar
  frame $menubar -relief raised -borderwidth 2
  menubutton $menubar.b1 -text File -menu $menubar.b1.m1 -underline 0
  menu $menubar.b1.m1
  $menubar.b1.m1 add command -label "Print" -underline 0 \
                    -command "printCanvas .func${node}.bar func.$node"
  $menubar.b1.m1 add separator
  $menubar.b1.m1 add command -label "Close" -underline 0 \
                    -command "destroy .func$node"

  menubutton $menubar.b2 -text Value -menu $menubar.b2.m1 -underline 0
  menu $menubar.b2.m1
  submenu2 $menubar.b2.m1 ${prefix}value $func $node {$name}

  menubutton $menubar.b4 -text Mode -menu $menubar.b4.m1 -underline 0
  menu $menubar.b4.m1
  submenu4 $menubar.b4.m1 ${prefix}mode $func $node {$name}

  menubutton $menubar.b5 -text Units -menu $menubar.b5.m1 -underline 0
  menu $menubar.b5.m1
  submenu5 $menubar.b5.m1 ${prefix}unit $func $node {$name}

  menubutton $menubar.b6 -text Help -menu $menubar.b6.m1 -underline 0
  menu $menubar.b6.m1
  if [ALONE] {
    $menubar.b6.m1 add command -label "on function profile" -underline 3 \
	              -command "showHelp $myself 1.2.4-func 1"
  } else {
    $menubar.b6.m1 add command -label "on function profile" -underline 3 \
	              -command "xsend tau \[list showHelp $myself 1.2.4-func 1\]"
  }

  pack $menubar.b1 $menubar.b2 $menubar.b4 $menubar.b5\
       -side left -fill x -padx 5
  pack $menubar.b6 -side right -fill x -padx 5
}

#
# showFuncgraph: display function profile window
#
#           tag: Sage++ id of function
#

proc showFuncgraph {tag} {
  global data
  global funcvalue funcmode funcunit
  global proforder profvalue profmode profunit
  global tagname

  

  if { ! [winfo exists .func$tag] } {
    set funcvalue($tag) $profvalue(all)
    set funcmode($tag) $profmode(all)
    set funcunit($tag) $profunit(all)

    set win [toplevel .func$tag]
    set name $tagname($tag)
    wm title $win "$name profile"
    wm minsize $win 250 250

    funcgraphmenu .func$tag func redrawFuncgraph $tag {}
    redrawFuncgraph $tag {}

    scrollbar .func$tag.sv -orient vert -relief sunken \
	-command ".func$tag.bar yview"
    .func$tag.bar configure -yscrollcommand ".func$tag.sv set"

    scrollbar .func$tag.sh -orient horiz -relief sunken \
	-command ".func$tag.bar xview"
    .func$tag.bar configure -xscrollcommand ".func$tag.sh set"

    button .func$tag.but -text "close" -command "destroy .func$tag"

    bind .func$tag.bar <2> ".func$tag.bar scan mark %x %y"
    bind .func$tag.bar <B2-Motion> ".func$tag.bar scan dragto %x %y"

    pack .func$tag.mbar -side top -fill x
    pack .func$tag.but -side bottom -fill x
    pack .func$tag.sh -side bottom -fill x
    pack .func$tag.sv -side right -fill y
    pack .func$tag.bar -side left -fill both -expand yes
  } else {
    raise .func$tag
  }
}

#
# redrawFuncgraph: create or update contents of function profile window
#
#             tag: Sage++ id of function
#           dummy: unused parameter so it can be used with the generic
#                  menu construction functions
#

proc redrawFuncgraph {tag dummy} {
  global data
  global funodes fulabels
  global funcvalue funcmode funcunit
  global tagname

  # -- compute arguments for bargraph widget
  set value $funcvalue($tag)
  foreach n $funodes {
    if { $n == "" } {
      lappend pers 0.0
      lappend vals 0.0
      lappend tags -2
    } else {
      set i [lsearch -exact $data($n,${value}tags) $tag]
      if { $i == -1 } {
        lappend pers 0.0
        lappend vals 0.0
      } else {
        lappend pers [lindex $data($n,${value}percent) $i]
        lappend vals [lindex $data($n,${value}usecs) $i]
      }
      lappend tags $tag
    }
  }
  if { $funcmode($tag) == "val" } {
    set vmax [vectormax vals]
    set pers [vectorpercent vals $vmax 80.0]
  }

  # -- create display using bargraph widget
  bargraph .func$tag.bar $tagname($tag) $fulabels $pers \
                         [vectorop vals / $funcunit($tag)] \
                         $tags $funodes $funcmode($tag)
  if { $funcmode($tag) == "val" } {
    .func$tag.mbar.b5 configure -state normal
  } else {
    .func$tag.mbar.b5 configure -state disabled
  }
}

#
# collgraphmenu: create menubar for collection profile windows
#
#        parent: pathname for parent of menubar
#        prefix: prefix for global variables to store selections
#          func: function to call for redraw
#     node,name: arguments to func
#

proc collgraphmenu {parent prefix func node name} {
  global myself

  set menubar $parent.mbar
  frame $menubar -relief raised -borderwidth 2
  menubutton $menubar.b1 -text File -menu $menubar.b1.m1 -underline 0
  menu $menubar.b1.m1
  $menubar.b1.m1 add command -label "Print" -underline 0 \
                    -command "printCanvas .coll${node}.bar coll.$node"
  $menubar.b1.m1 add separator
  $menubar.b1.m1 add command -label "Close" -underline 0 \
                    -command "destroy .coll$node"

  menubutton $menubar.b6 -text Mode -menu $menubar.b6.m1 -underline 0
  menu $menubar.b6.m1
  submenu6 $menubar.b6.m1 ${prefix}mode $func $node {$name}

  menubutton $menubar.b7 -text Help -menu $menubar.b7.m1 -underline 0
  menu $menubar.b7.m1
  if [ALONE] {
    $menubar.b7.m1 add command -label "on collection profile" -underline 3 \
	              -command "showHelp $myself 1.3.1-coll 1"
  } else {
    $menubar.b7.m1 add command -label "on collection profile" -underline 3 \
	              -command "xsend tau \[list showHelp $myself 1.3.1-coll 1\]"
  }

  pack $menubar.b1 $menubar.b6 -side left -fill x -padx 5
  pack $menubar.b7 -side right -fill x -padx 5
}

#
# aggrgraphmenu: create menubar for the aggregate event profile windows
#
#        parent: pathname for parent of menubar
#        prefix: prefix for global variables to store selections
#          func: function to call for redraw
#     node,name: arguments to func
#

proc aggrgraphmenu {parent prefix func node name} {
  global myself

  set menubar $parent.mbar
  frame $menubar -relief raised -borderwidth 2
  menubutton $menubar.b1 -text File -menu $menubar.b1.m1 -underline 0
  menu $menubar.b1.m1
  $menubar.b1.m1 add command -label "show Event Legend" -underline 5 \
	            -command showEventLegend
  $menubar.b1.m1 add command -label "Print" -underline 0 \
                    -command "printCanvas .aggr${node}.bar aggr.$node"
  $menubar.b1.m1 add separator
  $menubar.b1.m1 add command -label "Close" -underline 0 \
                    -command "destroy .aggr$node"

  menubutton $menubar.b6 -text Mode -menu $menubar.b6.m1 -underline 0
  menu $menubar.b6.m1
  submenu6a $menubar.b6.m1 ${prefix}mode $func $node {$name}

  menubutton $menubar.b7 -text Help -menu $menubar.b7.m1 -underline 0
  menu $menubar.b7.m1
  if [ALONE] {
    $menubar.b7.m1 add command -label "on aggregate event profile" -underline 3 \
                      -command "showHelp $myself 1.4.1-aggr 1"
  } else {
    $menubar.b7.m1 add command -label "on aggregate event profile" -underline 3 \
                      -command "xsend tau \[list showHelp $myself 1.4.1-aggr 1\]"
  }

  pack $menubar.b1 $menubar.b6 -side left -fill x -padx 5
  pack $menubar.b7 -side right -fill x -padx 5
}

#
# showCollgraph: display collection profile window
#
#         ident: collection id
#

proc showCollgraph {ident} {
  global coll
  global collmode profmode

  if { ! [winfo exists .coll$ident] } {
    set win [toplevel .coll$ident]

    set name $coll($ident,name)
    wm title $win "$name profile"
    wm minsize $win 300 50

    if { $profmode(all) == "per" } {
      set collmode($ident) lper
    } else {
      set collmode($ident) $profmode(all)
    }

    collgraphmenu .coll$ident coll redrawCollgraph $ident {}
    pack .coll$ident.mbar -side top -fill x

    set e [lindex $coll($ident,info) 0]
    set s [lindex $coll($ident,info) 1]
    set d [lindex $coll($ident,info) 2]
    set t [expr $e*$s]
    if { $t > 1000000 } {
      set t "[format {%.1f MB} [expr $t/1e6]]"
    } elseif { $t > 1000 } {
      set t "[format {%.1f kB} [expr $t/1e3]]"
    } else {
      set t "$t B"
    }
    frame .coll$ident.info -background white -relief raised -borderwidth 2
    label .coll$ident.l1 -text "$coll($ident,type) $name" \
                         -background white -foreground black
    label .coll$ident.l2 -text "$e elements of size $s \[$t\]" \
                         -background white -foreground black
    label .coll$ident.l3 -text "$d-dimensional shape" \
                         -background white -foreground black
    pack .coll$ident.l1 .coll$ident.l2 .coll$ident.l3 \
                        -side top -anchor w -in .coll$ident.info
    pack .coll$ident.info -side top -fill x

    redrawCollgraph $ident {}

    bind .coll$ident.bar <2> ".coll$ident.bar scan mark %x %y"
    bind .coll$ident.bar <B2-Motion> ".coll$ident.bar scan dragto %x %y"

    scrollbar .coll$ident.s1 -orient vert -relief sunken \
                           -command ".coll$ident.bar yview"
    .coll$ident.bar configure -yscrollcommand ".coll$ident.s1 set"

    button .coll$ident.but -text "close" -command "destroy .coll$ident"
    pack .coll$ident.but -side bottom -fill x
    pack .coll$ident.s1 -side right -fill y
    pack .coll$ident.bar -side left -fill both -expand yes
  } else {
    raise .coll$ident
  }
}


#
# showAggrgraph: display aggregate profile window
#
#         ident: aggregate id
#

proc showAggrgraph {ident} {
  global aggr
  global aggrmode profmode

  if { ! [winfo exists .aggr$ident] } {
    set win [toplevel .aggr$ident]

    set name $aggr($ident,container_name)
    set var_name $aggr($ident,variable_name)
    set cont_type $aggr($ident,container_type)

    wm title $win "$name event profile"
    wm minsize $win 300 50

    if { $profmode(all) == "per" } {
	set aggrmode($ident) lper
    } elseif {$profmode(all) == "gper" } {
	set aggrmode($ident) lper
    } else {
      set aggrmode($ident) $profmode(all)
    }

    aggrgraphmenu .aggr$ident aggr redrawAggrgraph $ident {}
    pack .aggr$ident.mbar -side top -fill x

    set e $aggr($ident,elem)
    set s $aggr($ident,size)
    set d $aggr($ident,dim)
    set t [expr $e*$s]
    if { $t > 1000000 } {
      set t "[format {%.1f MB} [expr $t/1e6]]"
    } elseif { $t > 1000 } {
      set t "[format {%.1f kB} [expr $t/1e3]]"
    } else {
      set t "$t B"
    }
    frame .aggr$ident.info -background white -relief raised -borderwidth 2
    if {$cont_type == "NULL"} {
	label .aggr$ident.l1 -text "$name $var_name" \
		-background white -foreground black
    } else {
	label .aggr$ident.l1 -text "$name<$cont_type> $var_name" \
		-background white -foreground black
    }
    label .aggr$ident.l2 -text "$e elements of size $s \[$t\]" \
                         -background white -foreground black
    label .aggr$ident.l3 -text "$d-dimensional shape" \
                         -background white -foreground black
    pack .aggr$ident.l1 .aggr$ident.l2 .aggr$ident.l3 \
                        -side top -anchor w -in .aggr$ident.info
    pack .aggr$ident.info -side top -fill x

    redrawAggrgraph $ident {}

    bind .aggr$ident.bar <2> ".aggr$ident.bar scan mark %x %y"
    bind .aggr$ident.bar <B2-Motion> ".aggr$ident.bar scan dragto %x %y"

    scrollbar .aggr$ident.s1 -orient vert -relief sunken \
                           -command ".aggr$ident.bar yview"
    .aggr$ident.bar configure -yscrollcommand ".aggr$ident.s1 set"
    if {$aggrmode($ident) != "none"} {
	scrollbar .aggr$ident.s2 -orient horiz -relief sunken \
		-command ".aggr$ident.bar xview"
	.aggr$ident.bar configure -xscrollcommand ".aggr$ident.s2 set"
    }

    button .aggr$ident.but -text "close" -command "destroy .aggr$ident"
    pack .aggr$ident.but -side bottom -fill x
    pack .aggr$ident.s1 -side right -fill y
    if {$aggrmode($ident) != "none"} {
	pack .aggr$ident.s2 -side bottom -fill x
    }
    pack .aggr$ident.bar -side right -fill both -expand yes
  } else {
    raise .aggr$ident
  }
}


#
# redrawCollgraph: create or update contents of collection profile window
#
#           ident: collection id
#           dummy: unused parameter so it can be used with the generic
#                  menu construction functions
#

proc redrawCollgraph {ident dummy} {
  global conodes colabels
  global coll
  global collmode

  set mode $collmode($ident)
  if { $mode == "gper" || $mode == "val" } {
    # -- global percent / value display mode
    # -- compute data from local percent available from pprof
    set max   0.0
    foreach n $conodes {
      if [ info exists coll($ident,$n,val) ] {
        set v $coll($ident,$n,val)
        lappend value $v
        if { $n != "t" } {
          set s [expr [lindex $v 0]+[lindex $v 1]]
          if { $s > $max } { set max $s }
        }
      } else {
        lappend value [list 0.0 0.0]
      }
    }
    set total [expr \
          double([lindex $coll($ident,t,val) 0]+[lindex $coll($ident,t,val) 1])]
    set scale [expr $total/$max]

    foreach n $conodes {
      if [ info exists coll($ident,$n,val) ] {
        set v $coll($ident,$n,val)
        lappend percent [list [expr [lindex $v 0]/$total*100.0] \
                              [expr [lindex $v 1]/$total*100.0]]
      } else {
        lappend percent [list 0.0 0.0]
      }
    }
  } elseif { $mode == "lper" } {
    # -- local percent display mode
    foreach n $conodes {
      if [ info exists coll($ident,$n,per) ] {
        lappend percent $coll($ident,$n,per)
      } else {
        lappend percent [list 0.0 0.0]
      }
    }
    set value {}
    set scale 1.0
  }
  multiCollgraph .coll$ident.bar $colabels {} $percent $value $mode $scale
}

#
# redrawAggrgraph: create or update contents of aggregate profile window
#
#           ident: aggregate id
#           dummy: unused parameter so it can be used with the generic
#                  menu construction functions
#

proc redrawAggrgraph {ident dummy} {
    global agnodes aglabels
    global aggr
    global aggrmode
    
    if [DEBUG] {
	puts "redrawAggrgraph:"
	puts "  ident: $ident"
    }

    set mode $aggrmode($ident)
    if { $mode == "val" } {
	# -- value display mode
	# -- compute data from local percent available from pprof
	set max   0.0
      
	set value [list]
	foreach n $agnodes {
	    # create a list of all the value sets
	    set tmp_values {}
	    foreach event $aggr($ident,counter_set) {
		if {$n == "t"} {
		    if [info exists aggr($ident,counter_t,$event,value)] {
			lappend tmp_values \
				$aggr($ident,counter_t,$event,value)
		    } else {
			lappend tmp_values -1.0
		    }
		} else {
		    if [info exists aggr($ident,$n,counter,$event,value)] {
			lappend tmp_values \
				$aggr($ident,$n,counter,$event,value)
		    } else {
			lappend tmp_values -1.0
		    }
		}
	    }
	    set s [vectorsum tmp_values]
	    if { $n != "t" && $n != {}} {
		lappend sums $s
		if { $s > $max } { set max $s }
	    } else {
		lappend sums 1.0
	    }
	    lappend value $tmp_values
	}
	    
	# Total the counter totals
	set total 0.0
	foreach event $aggr($ident,counter_set) {
	    set total [expr $total + \
		    double($aggr($ident,counter_t,$event,value))]
	}
	#set scale [expr $total/$max]
	for {set i 0} {$i < [llength $agnodes]} {incr i} {
	    if {[lindex $sums $i] == 0} {
		lappend scales 1.0
	    } else {
		lappend scales [expr $total / [lindex $sums $i]]
	    }
	}
	
	set percent [list]
	foreach n $agnodes {
	    # create a list of all the value sets
	    set tmp_percents {}
	    foreach event $aggr($ident,counter_set) {
		if {$n == "t"} {
		    if [ info exists aggr($ident,counter_t,$event,value) ] {
			lappend tmp_percents \
				[expr $aggr($ident,counter_t,$event,value) \
				/$total*100.0]
		    } else {
			lappend tmp_percents -1.0
		    }
		} else {
		    if [ info exists aggr($ident,$n,counter,$event,value) ] {
			lappend tmp_percents \
				[expr $aggr($ident,$n,counter,$event,value) \
				/$total*100.0]
		    } else {
			lappend tmp_percents -1.0
		    }
		}
	    }
	    lappend percent $tmp_percents
	}

    } elseif { $mode == "lper" } {
	# -- local percent display mode
	set percent [list]
	foreach n $agnodes {
	    # create a list of all the value sets
	    set tmp_percents {}
	    foreach event $aggr($ident,counter_set) {
		if {$n == "t"} {
		    if [ info exists aggr($ident,counter_t,$event,percent) ] {
			lappend tmp_percents \
				$aggr($ident,counter_t,$event,percent)
		    } else {
			lappend tmp_percents -1.0
		    }
		} else {
		    if [ info exists aggr($ident,$n,counter,$event,percent) ] {
			lappend tmp_percents \
				$aggr($ident,$n,counter,$event,percent)
		    } else {
			lappend tmp_percents -1.0
		    }
		}
	    }
	    lappend percent $tmp_percents
	}
	set value {}
	#set scale 1.0
	for {set i 0} {$i < [llength $agnodes]} {incr i} {
	    lappend scales 1.0
	}
    }
    multiAggrgraph .aggr$ident.bar $aglabels [list $ident] $percent $value $mode $scales "node"
}

#
# computeMultiBars: compute data for summary function profile
#                   as well as often used lists of labels
#

proc computeMultiBars {} {
  global data alltags
  global nodes funodes conodes agnodes
  global fulabels colabels clabels aglabels alabels
  global exclpercent collpercent aggrpercent
  global coll colls
  global aggr aggrs
  global minheight

  foreach n $nodes {
    set p ""
    set na [vectorsum data($n,exclpercent)]
    foreach t $alltags {
      set i [lsearch -exact $data($n,excltags) $t]
      if { $i == -1 } {
        lappend p 0.0
      } else {
        set x [lindex $data($n,exclpercent) $i]
        if { $t == -1 } {
          lappend p $na
        } else {
          set na [expr $na-$x]
          lappend p $x
        }
      }
    }
    set data($n,meanper) $p
  }

  #set fulabels [list "mean" "min" "max" {}]
  set fulabels [list "mean" {}]
  set colabels [list "total" {}]
  set aglabels [list "total" {}]
  #set exclpercent \
  #    [list $data(m,exclpercent) $data(<,meanper) $data(>,meanper) {}]
  #  set exclpercent [list $data(m,exclpercent) {}]
  set exclpercent [list $data(m,meanper) {}]
  #set funodes "m < > {} [lrange $nodes 0 [expr [llength $nodes]-5]]"
  set funodes "m {} [lrange $nodes 0 [expr [llength $nodes]-3]]"
  #set conodes "t {} [lrange $nodes 0 [expr [llength $nodes]-5]]"
  set conodes "t {} [lrange $nodes 0 [expr [llength $nodes]-3]]"
  #set agnodes "t {} [lrange $nodes 0 [expr [llength $nodes]-5]]"
  set agnodes "t {} [lrange $nodes 0 [expr [llength $nodes]-3]]"

  #foreach n [lrange $nodes 0 [expr [llength $nodes]-5]] 
  foreach n [lrange $nodes 0 [expr [llength $nodes]-3]] {
    lappend fulabels [format "n,c,t %5s" $n]
    lappend colabels [format "n,c,t %5s" $n]
    lappend aglabels [format "n,c,t %5s" $n]
    lappend exclpercent $data($n,meanper)
  }

  set clabels ""
  set collpercent ""
  foreach c $colls {
    lappend clabels $coll($c,name)
    lappend collpercent $coll($c,t,per)
  }

  set alabels ""
  set aggrpercent ""
  foreach c $aggrs {
      lappend alabels $aggr($c,variable_name)

      set tmp_percents [list]
      foreach event $aggr($c,counter_set) {
	  lappend tmp_percents $aggr($c,counter_t,$event,percent)
      }
      lappend aggrpercent $tmp_percents
  }

    set minheight [expr ([max3 [llength $fulabels] \
			       [llength $clabels] \
			       [llength $alabels] ]+1)*20]
}


#
# redraw: redraw all windows currently displayed, after user selected
#         global order from Configure main menu
#
# dummy1:
# dummy2: unused parameters so it can be used with the generic
#         menu construction functions
#

proc redraw {dummy1 dummy2} {
  global myself
  global data
  global funodes fulabels
  global barorder barmode barvalue barunit
  global textorder1 textorder2
  global funcmode funcvalue funcunit
  global proforder profvalue profmode profunit
  global collmode colls aggrmode aggrs



  # -- redraw node and text node profile windows
  set i 0
  foreach n $funodes {
    if [winfo exists .bar$n] {
      set barvalue($n) $profvalue(all)
      #if { $proforder(all) == "glob" } {
      #  set barorder($n) $proforder(all)
      #} else {
      #  set barorder($n) $n
      #}
      if { $proforder(all) == "all" } {
        set barorder($n) $n
      } else {
        set barorder($n) $proforder(all)
      }
      set barmode($n) $profmode(all)
      set barunit($n) $profunit(all)
      redrawBargraph $n [lindex $fulabels $i]
    }
    if [winfo exists .text$n] {
      set textorder1($n) $profvalue(all)
      #if { $proforder(all) == "glob" } {
      #  set textorder2($n) $proforder(all)
      #} else {
      #  set textorder2($n) $n
      #}
      if { $proforder(all) == "all" } {
        set textorder2($n) $n
      } else {
        set textorder2($n) $proforder(all)
      }

      redrawText $n [lindex $fulabels $i]
    }
    incr i
  }

  # -- redraw function profile windows
  foreach t $data(m,excltags) {
    if [winfo exists .func$t] {
      set funcvalue($t) $profvalue(all)
      set funcmode($t) $profmode(all)
      set funcunit($t) $profunit(all)
      redrawFuncgraph $t {}
    }
  }

  # -- redraw collection profile windows
  foreach c $colls {
    if [winfo exists .coll$c] {
      if { $profmode(all) == "per" } {
        set collmode($c) lper
      } else {
        set collmode($c) $profmode(all)
      }
      redrawCollgraph $c {}
    }
  }

  # -- redraw aggregate profile windows
  foreach a $aggrs {
    if [winfo exists .aggr$a] {
      if { $profmode(all) == "per" } {
        set aggrmode($a) lper
      } else {
        set aggrmode($a) $profmode(all)
      }
      redrawAggrgraph $a {}
    }
  }

  # -- redraw function/event legends
  if [winfo exists .$myself.led.can] {
    redrawFuncLegend .$myself.led.can
  }
  if [winfo exists .$myself.eventled.can] {
    redrawEventLegend .$myself.eventled.can
  }

  if { $profmode(all) == "val" } {
    .$myself.mbar.b2.m1 entryconfigure 4 -state normal
  } else {
    .$myself.mbar.b2.m1 entryconfigure 4 -state disabled
  }
}

#
# showFuncLegend: display function legend
#

proc showFuncLegend {} {
  global myself

  if { ! [winfo exists .$myself.led] } {
    toplevel .$myself.led
    wm title .$myself.led "Function Legend"
    wm minsize .$myself.led 250 250

    # -- create canvas, scrollbars, "close" button
    canvas .$myself.led.can -background white

    scrollbar .$myself.led.sv -orient vert -relief sunken \
	                      -command ".$myself.led.can yview"
    .$myself.led.can configure -yscrollcommand ".$myself.led.sv set"

    scrollbar .$myself.led.sh -orient horiz -relief sunken \
                              -command ".$myself.led.can xview"
    .$myself.led.can configure -xscrollcommand ".$myself.led.sh set"

    button .$myself.led.b -text close -command "destroy .$myself.led"

    pack .$myself.led.b -side bottom -fill x
    pack .$myself.led.sh -side bottom -fill x
    pack .$myself.led.sv -side right -fill y
    pack .$myself.led.can -side left -fill both -expand yes

    # -- draw color legend 
    redrawFuncLegend .$myself.led.can
  } else {
    raise .$myself.led
  }
}

#
# showEventLegend: display event legend
#

proc showEventLegend {} {
  global myself

  if { ! [winfo exists .$myself.eventled] } {
    toplevel .$myself.eventled
    wm title .$myself.eventled "Event Legend"

    # -- create canvas and "close" button
    canvas .$myself.eventled.can -width 250 -background white
    pack .$myself.eventled.can

    button .$myself.eventled.b -text close -command "destroy .$myself.eventled"
    pack .$myself.eventled.b -fill x

    # -- draw color legend
    redrawEventLegend .$myself.eventled.can
  } else {
    raise .$myself.eventled
  }
}

#
# redrawFuncLegend: draw or update function legend canvas
#
#          can: canvas id
#

proc redrawFuncLegend {can} {
  global tagcol tagstip tagname alltags
  global pr_sel_tag
  global racy_progfile

  # -- generate alphabetical list of functions
  #foreach t [array names tagcol] 
  foreach t $alltags {
    if { $t != -2 && $t != -1 } {
      lappend map [list $tagname($t) $t $tagcol($t) $tagstip($t)]
    }
  }
  set map [lsort $map]
  lappend map [list $tagname(-1) -1 $tagcol(-1) {}]

  # -- set height of map
  set num [llength $map];    # number of funcs
  set h [expr ($num+1)*20];  # height of canvas

  # -- draw color legend
  $can delete all
  set i 11
  set max_lwidth 0
  foreach m $map {
    set t [lindex $m 1]
    $can create rectangle 10 [expr $i] 28 [expr 18+$i] \
	-fill [lindex $m 2] -tag fl$t -stipple [lindex $m 3]
    if { $t == $pr_sel_tag } {
	set label [$can create text 40 [expr 9+$i] -text [lindex $m 0] \
		       -anchor w -tag fl$t -fill red \
		       -font -Adobe-Helvetica-Bold-R-Normal--*-120-*]
    } else {
	set label [$can create text 40 [expr 9+$i] -text [lindex $m 0] \
		       -anchor w -tag fl$t \
		       -font -Adobe-Helvetica-Bold-R-Normal--*-120-*]
    }
    set bbox [$can bbox $label]
    set lwidth [expr [lindex $bbox 2] - [lindex $bbox 0]]
    if {$lwidth > $max_lwidth} {set max_lwidth $lwidth}
    $can bind fl$t <Button-1> "showFuncgraph $t"
    # showFuncTag is not implemented by racy
    # $can bind fl$t <Button-2> "PM_GlobalSelect $racy_progfile global_showFuncTag $t"
    $can bind fl$t <Button-3> "PM_GlobalSelect $racy_progfile global_selectFuncTag $t"
    incr i 20
  }

  set w [expr $max_lwidth + 50]
  $can configure -scrollregion [list 0 0 $w $h]
}

#
# redrawEventLegend: draw or update event legend canvas
#
#          can: canvas id
#

proc redrawEventLegend {can} {
  global evcol evstip evname
  global pr_sel_tag
  global racy_progfile

  # -- generate alphabetical list of functions
  foreach t [array names evcol] {
    if { $t != -2 && $t != -1 } {
      lappend map [list $evname($t) $t $evcol($t) $evstip($t)]
    }
  }
  set map [lsort $map]

  # -- set height of map
  set num [llength $map];    # number of funcs
  set h [expr ($num+1)*20];  # height of canvas
  $can configure -height $h

  # -- draw color legend
  $can delete all
  set i 11
  foreach m $map {
    set t [lindex $m 1]
    $can create rectangle 10 [expr $i] 28 [expr 18+$i] \
         -fill [lindex $m 2] -tag el$t -stipple [lindex $m 3]
    if { $t == $pr_sel_tag } {
      $can create text 40 [expr 9+$i] -text [lindex $m 0] \
           -anchor w -tag el$t -fill red \
           -font -Adobe-Helvetica-Bold-R-Normal--*-120-*
    } else {
      $can create text 40 [expr 9+$i] -text [lindex $m 0] -anchor w -tag fl$t \
           -font -Adobe-Helvetica-Bold-R-Normal--*-120-*
    }
    incr i 20
  }
}

#
# setNumFunc: set maximal number of functions included in profile data displays
#

proc setNumFunc {} {
  global myself
  global depfile
  global wantnum

  set copy $wantnum

  if { ! [winfo exists .$myself.num] } {
    toplevel .$myself.num
    wm title .$myself.num "Set #Functions"

    numberInput .$myself.num.i "Number of functions to display:" wantnum
    pack .$myself.num.i -side top

    frame .$myself.num.b -relief sunken -bd 1
    pack .$myself.num.b -side left -padx 15 -pady 10
    button .$myself.num.b.b1 -text "set" -command {
      destroy .$myself.num
      loadProfile
    }
    pack .$myself.num.b.b1 -side top -padx 5 -pady 5

    button .$myself.num.b2 -text "cancel" -command "
      set wantnum $copy
      destroy .$myself.num
    "
    pack .$myself.num.b2 -side right -padx 15

    bind .$myself.num.i.e <Return> {.$myself.num.b.b1 invoke}
    bind .$myself.num <Return> {.$myself.num.b.b1 invoke}
  } else {
    raise .$myself.num
  }
}

#
#  mainmenu: create menubar for main window
#
#    parent: pathname for parent of menubar
#    prefix: prefix for global variables to store selections
#      func: function to call for redraw
# node,name: arguments to func
#

proc mainmenu {parent prefix func node name} {
  global myself colls aggrs

  set menubar $parent.mbar
  frame $menubar -relief raised -borderwidth 2

  menubutton $menubar.b1 -text File -menu $menubar.b1.m1 -underline 0
  menu $menubar.b1.m1
  $menubar.b1.m1 add command -label "set #Functions" -underline 0 \
                    -command setNumFunc
  $menubar.b1.m1 add command -label "show Function Legend" -underline 5 \
                    -command showFuncLegend
  if [llength $aggrs] {
      $menubar.b1.m1 add command -label "show Event Legend" -underline 5 \
	      -command showEventLegend
  }

  $menubar.b1.m1 add command -label "print Functions" -underline 0 \
                    -command "printCanvas .$myself.fu.bar profile.fu"
  if [llength $colls] {
      $menubar.b1.m1 add command -label "print Collections" -underline 6 \
	      -command "printCanvas .$myself.co.bar profile.co"
  } elseif [llength $aggrs] {
      $menubar.b1.m1 add command -label "print Aggregates" -underline 6 \
	      -command "printCanvas .$myself.ag.bar profile.ag"
  }
  $menubar.b1.m1 add separator
  $menubar.b1.m1 add command -label "Exit" -underline 0 -command "exit"

  menubutton $menubar.b2 -text Configure -menu $menubar.b2.m1 -underline 0
  menu $menubar.b2.m1

  $menubar.b2.m1 add cascade -label "Value" -menu $menubar.b2.m1.1 -underline 0
  menu $menubar.b2.m1.1
  submenu2 $menubar.b2.m1.1 ${prefix}value $func $node {$name}

  $menubar.b2.m1 add cascade -label "Order" -menu $menubar.b2.m1.2 -underline 0
  menu $menubar.b2.m1.2
  submenu3 $menubar.b2.m1.2 ${prefix}order $func $node {$name}

  $menubar.b2.m1 add cascade -label "Mode" -menu $menubar.b2.m1.3 -underline 0
  menu $menubar.b2.m1.3
  submenu4 $menubar.b2.m1.3 ${prefix}mode $func $node {$name}

  $menubar.b2.m1 add cascade -label "Units" -menu $menubar.b2.m1.4 -underline 0
  menu $menubar.b2.m1.4
  submenu5 $menubar.b2.m1.4 ${prefix}unit $func $node {$name}
  $menubar.b2.m1 entryconfigure 4  -state disabled

  menubutton $menubar.b3 -text Help -menu $menubar.b3.m1 -underline 0
  menu $menubar.b3.m1
  if [ALONE] {
    $menubar.b3.m1 add command -label "on $myself" -underline 3 \
	-command "showHelp $myself 1-$myself 1"
    $menubar.b3.m1 add separator
    $menubar.b3.m1 add command -label "on menubar" -underline 3 \
	-command "showHelp $myself 1.1-menu 1"
    $menubar.b3.m1 add command -label "on function summary" -underline 3 \
	-command "showHelp $myself 1.2-funcsum 1"
    if [llength $colls] {
      $menubar.b3.m1 add command -label "on collection summary" -underline 3 \
	  -command "showHelp $myself 1.3-collsum 1"
    } elseif [llength $aggrs] {
      $menubar.b3.m1 add command -label "on aggregate summary" -underline 3 \
	  -command "showHelp $myself 1.4-aggrsum 1"
    }
    $menubar.b3.m1 add separator
    $menubar.b3.m1 add command -label "on using help" -underline 3 \
	-command "showHelp general 1-help 1"
  } else {
    $menubar.b3.m1 add command -label "on $myself" -underline 3 \
	-command "xsend tau \[list showHelp $myself 1-$myself 1\]"
    $menubar.b3.m1 add separator
    $menubar.b3.m1 add command -label "on menubar" -underline 3 \
	-command "xsend tau \[list showHelp $myself 1.1-menu 1\]"
    $menubar.b3.m1 add command -label "on function summary" -underline 3 \
	-command "xsend tau \[list showHelp $myself 1.2-funcsum 1\]"
    if [llength $colls] {
      $menubar.b3.m1 add command -label "on collection summary" -underline 3 \
	  -command "xsend tau \[list showHelp $myself 1.3-collsum 1\]"
    } elseif [llength $aggrs] {
      $menubar.b3.m1 add command -label "on aggregate summary" -underline 3 \
	  -command "xsend tau \[list showHelp $myself 1.4-aggrsum 1\]"
    }
    $menubar.b3.m1 add separator
    $menubar.b3.m1 add command -label "on using help" -underline 3 \
	-command "xsend tau \[list showHelp general 1-help 1\]"
  }

  if {![ALONE]} {
    createToolMenu $menubar.b4
  }

  pack $menubar.b1 $menubar.b2 -side left -padx 5
  if [ALONE] {
    pack $menubar.b3  -side right -padx 5
  } else {
    pack $menubar.b3 $menubar.b4 -side right -padx 5
  }
}

#
# loadProfile: switch to another application
#              this is invoked from the TAU master control window
#

proc loadProfile {} {
  global myself
  global data
  global numfunc
  global nodes funodes fulabels conodes colabels exclpercent agnodes aglabels
  global barorder barmode barvalue barunit
  global textorder1 textorder2
  global funcmode funcvalue funcunit
  global collmode aggrmode
  global proforder profvalue profmode profunit
  global tagname tagcol tagfcol tagstip
  global evname evcol evstip
  global colls coll clabels collpercent
  global aggrs aggr alabels aggrpercent
  global depfile alltags

  set dir  $depfile(dir)
  set root $depfile(root)
  set arch $depfile(arch)

  foreach n $funodes {
    if [winfo exists .bar$n]  { destroy .bar$n  }
    if [winfo exists .text$n] { destroy .text$n }
  }
  foreach t $data(m,excltags) {
    if [winfo exists .func$t] { destroy .func$t }
  }
  foreach c $colls {
    if [winfo exists .coll$c] { destroy .coll$c }
  }
  foreach a $aggrs {
    if [winfo exists .aggr$a] { destroy .aggr$a }
  }
  if [winfo exists .$myself.num] { destroy .$myself.num }
  if [winfo exists .$myself.led] { destroy .$myself.led }
  if [winfo exists .$myself.eventled] { destroy .$myself.eventled }

  # -- destroy old profile data database
  unset data
  unset alltags
  unset nodes funodes fulabels conodes colabels exclpercent
  unset agnodes aglabels evname evcol evstip alabels aggrpercent
  unset tagname tagcol tagfcol tagstip
  unset clabels collpercent
  if { [info exists barorder]   } { unset barorder barmode barvalue barunit }
  if { [info exists textorder1] } { unset textorder1 textorder2 }
  if { [info exists funcmode]   } { unset funcmode funcvalue funcunit }
  if { [info exists collmode]   } { unset collmode }
  if { [info exists aggrmode]   } { unset aggrmode }
  if { [info exists colls]      } { unset colls }
  if { [info exists coll]       } { unset coll }
  if { [info exists aggrs]      } { unset aggrs }
  if { [info exists aggr]       } { unset aggr }

  # -- reset global configuration parameters
  set profvalue(all) excl
  #set proforder(all) glob
  set proforder(all) all
  set profmode(all) per
  set profunit(all) 1.0

  # -- read new profile data
  if { [readProfile] == "NOT_OK" } {
    #showError "No profile data available in directory $dir."
    after 1 exit
    return
  }
  if {![ALONE]} {
    xsend tau "set depfile(numprof) $numfunc"
  }

  # -- redraw summary bargraphs
  computeMultiBars
  # multiFuncgraph .$myself.fu.bar $fulabels $funodes \
                 $exclpercent $data(m,excltags)
  multiFuncgraph .$myself.fu.bar $fulabels $funodes \
                 $exclpercent $alltags
  if [llength $colls] {
      multiCollgraph .$myself.co.bar $clabels $colls $collpercent
  } elseif [llength $aggrs] {
      multiAggrgraph .$myself.ag.bar $alabels $aggrs $aggrpercent
  }

  # -- rebuild configure-order-node submenu
  .$myself.mbar.b2.m1.2.1 delete 0 last
  set i 0
  foreach n $funodes {
    if { $n != {} } {
      .$myself.mbar.b2.m1.2.1 add radiobutton -label [lindex $fulabels $i] \
	          -variable proforder(all) -value $n \
                  -command "redraw all {}"
    } else {
      .$myself.mbar.b2.m1.2.1 add separator
    }
    incr i
  }
}

#
# createWindow: create and display main window of racy
#

proc createWindow {} {
  global myself
  global TAUDIR
  global data
  global fulabels funodes clabels colls collpercent exclpercent \
	  aggrs alabels aggrpercent alltags

  toplevel .$myself
  wm title .$myself "RACY"
  wm minsize .$myself 300 50
  wm iconbitmap .$myself @$TAUDIR/xbm/$myself.xbm

  mainmenu .$myself prof redraw all {}
  pack .$myself.mbar -side top -fill x

  frame .$myself.fu
  label .$myself.fu.l1 -text Functions -relief raised
  multiFuncgraph .$myself.fu.bar $fulabels $funodes \
                 $exclpercent $alltags 
  scrollbar .$myself.fu.s1 -orient vert -relief sunken \
                        -command ".$myself.fu.bar yview"
  .$myself.fu.bar configure -yscrollcommand ".$myself.fu.s1 set"
  pack .$myself.fu.l1 -side top -fill x
  pack .$myself.fu.s1 -side right -fill y
  pack .$myself.fu.bar -side left -fill both -expand yes

  if [llength $colls] {
      frame .$myself.co
      label .$myself.co.l1 -text Collections -relief raised
      multiCollgraph .$myself.co.bar $clabels $colls $collpercent
      scrollbar .$myself.co.s1 -orient vert -relief sunken \
	      -command ".$myself.co.bar yview"
      .$myself.co.bar configure -yscrollcommand ".$myself.co.s1 set"
      pack .$myself.co.l1 -side top -fill x
      pack .$myself.co.s1 -side right -fill y
      pack .$myself.co.bar -side left -fill both -expand yes
      pack .$myself.fu .$myself.co -side left -padx 15 -pady 15 \
	      -fill both -expand yes
  } elseif [llength $aggrs] {
      frame .$myself.ag
      label .$myself.ag.l1 -text Aggregates -relief raised
      multiAggrgraph .$myself.ag.bar $alabels $aggrs $aggrpercent
      scrollbar .$myself.ag.s1 -orient vert -relief sunken \
	      -command ".$myself.ag.bar yview"
      .$myself.ag.bar configure -yscrollcommand ".$myself.ag.s1 set"
      pack .$myself.ag.l1 -side top -fill x
      pack .$myself.ag.s1 -side right -fill y
      pack .$myself.ag.bar -side left -fill both -expand yes
      pack .$myself.fu .$myself.ag -side left -padx 15 -pady 15 \
	      -fill both -expand yes
  } else {
      pack .$myself.fu -side left -padx 15 -pady 15 \
	      -fill both -expand yes
  }
}

#
# getColorCode: return list of actual function name <-> color mapping
#

proc getColorCode {} {
  global tagcol

  foreach t [array names tagcol] {
    if { $t != -2 } { lappend result [list $t $tagcol($t)] }
  }
  return $result
}


proc Tool_AcceptChanges {progfiles flag} {
    global myself depfile \
	    showFile selectBox racy_progfile

    switch $flag {

        d {
        }


        a {
        }


        u {
        }
	
	e {
	    loadProfile
	}

	p {
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
	    
	    #reset
	    global racy_progfile
	    set racy_progfile "";
	    
	    destroy .$myself
	    initRacy
	    if { [readProfile] == "NOT_OK" } {
		#showError "No profile data available in current directory."
		exit
	    }
	    computeMultiBars
	    createWindow
	}
    }
}



#  initRacy - Racy only works with simple pc++ projects, ie, pc++ projects
#             with only one program file.
proc initRacy {} {
    global racy_progfile alltags

    set alltags [list]
    
    set file [Bdb_GetMaintag]
    set racy_progfile [lindex $file 0]

    if {0} {
    set files [PM_GetFiles]
    if {[llength $files] != 1} {
	showError "A Racy-compatible project only must have one program (.C) file!"
	exit
    }
    set racy_progfile [lindex $files 0]
    }

    if { [readProfile] == "NOT_OK" } {
	#showError "No profile data available in current directory."
	exit
    }
}


# ------------
# -- main code
# ------------


set profvalue(all) excl
#set proforder(all) glob
set proforder(all) all
set profmode(all) per
set profunit(all) 1.0

# racy is currently a standalone tool
set ALONE_SET 1
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

# if want other than standalone tool, comment out above
# and uncomment the following
#set ALONE_SET 0
#switch $argc {
#    0   {
#	set parg [pwd]
#    }
#    1   {
#	if {[lindex $argv 0] == "-sa" } {
#	    set ALONE_SET 1
#	    set parg [pwd]
#        } else {
#  	    set parg [lindex $argv 0]
#	    if {[file extension $parg] != ".pmf"} {
#	        set parg "$parg.pmf"
#	    }
#	}
#    }
#    2   {
#	if {[lindex $argv 0] == "-sa" } {
#	    set ALONE_SET 1
#	    set parg [lindex $argv 1]
#	    if {[file extension $parg] != ".pmf"} {
#		set parg "$parg.pmf"
#	    }
#	} else {
#	    puts stderr "usage: $myself \[-sa\] \[\[host:\]projFile \| \[host:\]directory\]"
#	    exit
#	}
#    }
#    default {
#	puts stderr "usage: $myself \[-sa\] \[\[host:\]projFile \| \[host:\]directory\]"
#	exit
#    }
#}

# Init the project manager (taud)
launchTauDaemon -waitfor
PM_AddTool $myself
PM_AddGlobalSelect $myself {global_selectFuncTag}

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
initRacy

# -- create new toplevel window
computeMultiBars
createWindow
if {![ALONE]} {
    launchTAU
}

# -- notify tau about number of functions profiled
if {![ALONE]} {
    xsend tau "set depfile(numprof) $numfunc"
}

wm protocol .$myself WM_DELETE_WINDOW exit

removeMessage

