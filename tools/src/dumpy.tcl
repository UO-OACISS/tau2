#*********************************************************************#
#*                TAU/pC++/Sage++  Copyright (C) 1994                *#
#*                        Jerry Manic Saftware                       *#
#*  Indiana University  University of Oregon  University of Rennes   *#
#*********************************************************************#

source "$TAUDIR/inc/printcan.tcl"
source "$TAUDIR/inc/stack.tcl"
source "$TAUDIR/inc/help.tcl"

#
# breakup: parse input line into array of words
#          also replace string of dashes with $na
#
#    line: input line
#    word: array name for list of words
#      na: replacement string for NA's (dash strings)
#

proc breakup {line word {na -}} {
  upvar $word w

  regsub -all {[-]-+} $line $na newline
  regsub -all {[ 	]+} [string trim $newline] { } line
  set w [split $line]
}

#
# readDump: read output of dumpdep utility
#           has to be adapted to dumpdep output format changes
#
#  depfile: depfile to read by dumpdep
#

proc readDump {depfile} {
  global BINDIR
  global nodes
  global txtwin

  set lnum 1; #input linenumber

  # -- read output of dumpdep into text window
  set in [open "|$BINDIR/dumpdep $depfile" r]
  $txtwin insert end [read $in]
  if [catch "close $in" errmsg] {
    puts "$depfile: $errmsg"
    exit
  }
  $txtwin configure -state disabled

  # -- process this output
  while { [set line [$txtwin get $lnum.0 [incr lnum].0]] != "" } {
    scan $line "%s" node
    if { [regexp {([0-9]+)-([BSTEFL])} $node dummy num nodetype] == 1 } {
      # -- line contains BIF, SYMBOL, TYPE, EXPRESSION, LABEL, or FILE node
      # -- other lines are ignored
      set start [expr $lnum-1];  #remember start line
      breakup $line word
      switch $nodetype {
        B  {  # -- BIF node
              #set nodes($node,cp)   [lindex $word 2]
              set nodes($node,symb) [lindex $word 5]
              set nodes($node,e1)   [lindex $word 6]
              set nodes($node,e2)   [lindex $word 7]
              set nodes($node,e3)   [lindex $word 8]
              set nodes($node,labl) [lindex $word 12]
              set nodes($node,line) [lindex $word 13]
              set nodes($node,file) [lindex $word 15]
              set nodes($node,next) [lindex $word 16]
              set nodes($node,var)  [lindex $word 17]

              set line [$txtwin get $lnum.0 [incr lnum].0]
              breakup $line nodes($node,true) ""
              set line [$txtwin get $lnum.0 [incr lnum].0]
              breakup $line nodes($node,false) ""
              set line [$txtwin get $lnum.0 [incr lnum].0]
              breakup $line word
              set nodes($node,flag) [lrange $word 2 end]
           }
        E  {  # -- EXPRESSION node
              set nodes($node,type)  [lindex $word 2]
              set ll [expr [llength $word] - 1]
              set var [set nodes($node,var) [lindex $word $ll]]

              if [regexp {_VAL$} $var] {
                # -- constant value expression
                set nodes($node,symb)  -
                set nodes($node,left)  -
                set nodes($node,right) -
                set nodes($node,val)   [join [lrange $word 3 [expr $ll-1]]]
              } else {
                # -- all other expressions
                set nodes($node,symb)  [lindex $word 3]
                set nodes($node,left)  [lindex $word 4]
                set nodes($node,right) [lindex $word 5]
                set nodes($node,val)   -
              }
           }
        T  {  # -- TYPE node
              set name [lindex $word 2]

              # -- some TYPE nodes have 2 lines
              # -- last word (variant) should be DEFAULT or start with T_
              set ll [expr [llength $word] - 1]
              set var [lindex $word $ll]
              if { $var != "DEFAULT" && [regexp {^T_} $var] == 0 } {
                set line [$txtwin get $lnum.0 [incr lnum].0]
                breakup $line word2
                set ll [expr [llength $word2] - 1]
                set var [lindex $word2 $ll]
              }

              set nodes($node,flag)  ""
              set nodes($node,atype) "-"
              set nodes($node,dtype) "-"
              set nodes($node,base)  "-"
              set nodes($node,rngs)  "-"
              set nodes($node,chld)  "-"
              set nodes($node,parnt) "-"

              switch -exact $var {
                T_DERIVED_TEMPLATE { set nodes($node,dtype) [lindex $word 7]
                                     set nodes($node,atype) [lindex $word 10] }
                T_DERIVED_COLLECTION { set nodes($node,atype) [lindex $word 10]}
                T_DERIVED_TYPE   { set nodes($node,dtype) [lindex $word  5] }
                T_DESCRIPT       { set nodes($node,flag)  [lrange $word 10 end]
                                   set nodes($node,base)  [lindex $word2 5] }
                T_REFERENCE      -
                T_POINTER        { set nodes($node,base)  [lindex $word  5] }
                T_ARRAY          { set nodes($node,rngs)  [lindex $word  9]
                                   set nodes($node,base)  [lindex $word  7] }
                T_CLASS          -
                T_STRUCT         -
                T_ENUM           -
                T_UNION          { set nodes($node,chld)  [lindex $word  7]
                                   set nodes($node,parnt) [lindex $word  9] }
              }
              set nodes($node,var) $var
           }
        S  {  # -- SYMBOL node
              #set nodes($node,scope) [lindex $word 5]

              set nodes($node,type)  [lindex $word 2]
              set nodes($node,attr)  [lindex $word 3]
              set nodes($node,name)  [lindex $word 6]

              # -- some SYMBOL nodes have 2 or 3 lines
              # -- last word (variant) should be DEFAULT or end with _NAME
              # -- or _FUNC
              set ll [expr [llength $word] - 1]
              set var [set nodes($node,var) [lindex $word $ll]]
              if { $var != "DEFAULT" && [regexp {_NAME$|_FUNC$} $var] == 0 } {
                set line [$txtwin get $lnum.0 [incr lnum].0]
                breakup $line word2
                set ll [expr [llength $word2] - 1]
                set var [set nodes($node,var) [lindex $word2 $ll]]
              }
              if { $var != "DEFAULT" && [regexp {_NAME$|_FUNC$} $var] == 0 } {
                set line [$txtwin get $lnum.0 [incr lnum].0]
                breakup $line word3
                set ll [expr [llength $word3] - 1]
                set var [set nodes($node,var) [lindex $word3 $ll]]
              }

              set nodes($node,nchld) "-"
              switch -exact $var {
                FIELD_NAME   { set nodes($node,nchld) [lindex $word 10] }
                MEMBER_FUNC  { set nodes($node,nchld) [lindex $word3 1] }
              }
           }
        L  {  # -- LABEL node
              set nodes($node,symb) [lindex $word 4]
           }
        F  {  # -- FILE node
              set nodes($node) [lindex $word 1]
           }
      }
      # -- mark nodes with tag in text window
      $txtwin tag add node$node $start.0 $lnum.0
    }
    # -- reset so blank lines are processed correctly
    set line ""
    set node ""
  }
}

#
# printSymb: draw SYMBOL node and add bindings
#
#      symb: symbol node id
#       win: window pathname
#       x,y: window coordinates
#  showtype: draw type of symbol too?

proc printSymb {symb win x y {showtype true}} {
  global nodes

  # -- draw type, if necessary
  if { $showtype } {
    set type $nodes($symb,type)
    if { $type != "-" } {
      set obj [$win create text $x $y -text [printType $type] \
                                      -anchor w -tags t$type]
      set x [expr [lindex [$win bbox $obj] 2]+10]
    }
  }

  # -- draw symbol identifier and setup symbol bindings
  set obj [$win create text $x $y -text "$nodes($symb,name)  " \
           -tags s$symb -anchor w]
  $win bind $obj <ButtonPress-1> "showSymb $win $obj $symb 1"
  $win bind $obj <ButtonPress-2> "showSymb $win $obj $symb 2"
  $win bind $obj <ButtonRelease-1> \
                 "$win delete symb_obj; selectItem s$symb black"
  $win bind $obj <Button-3> "showText $symb"
}

#
# showSymbol: popup small symbol information window below symbol string
#
#        win: window pathname
#        obj: canvas item id which represents symbol string
#       symb: symbol node id
#     button: button pressed on symbol string
#

proc showSymb {win obj symb button} {
  global nodes

  # -- computes coordinates for window from symbol string object position
  set co [$win coords $obj]
  set x [expr [lindex $co 0]+10]
  set y [expr [lindex $co 1]+10]
  if { $button == "1" } {
    set tag "symb_obj"
  } else {
    set tag "symb$symb"
  }

  # -- draw double border rectangle containing variant, identifier, and
  # -- attributes
  $win create rectangle $x $y [expr $x+140] [expr $y+60] \
              -fill white -tags $tag
  $win create rectangle [expr $x+2] [expr $y+2] [expr $x+138] [expr $y+58] \
              -fill white -tags $tag
  $win create text [expr $x+10] [expr $y+15] -text $nodes($symb,var) \
              -anchor w -tags $tag
  $win create text [expr $x+10] [expr $y+30] -text $nodes($symb,name) \
              -anchor w -tags [list $tag s$symb]
  $win create text [expr $x+10] [expr $y+45] \
              -text [printAttr $nodes($symb,attr)] -anchor w -tags $tag

  if { $button == "1" } {
    selectItem s$symb red
  } elseif { $button == "2" } {
    $win bind symb$symb <Button-1> "$win delete symb$symb"
    $win bind symb$symb <Button-2> "$win delete symb$symb"
  }
}

set sym_nums {32768 16384 8192 4096 2048 1024 512 256 128 64 32 16 8 4 2 1}
set sym_attr(1) "att_global"
set sym_attr(2) ""
set sym_attr(4) ""
set sym_attr(8) "pure"
set sym_attr(16) "private"
set sym_attr(32) "protected"
set sym_attr(64) "public"
set sym_attr(128) "element"
set sym_attr(256) "collection"
set sym_attr(512) "ctor"
set sym_attr(1024) "dtor"
set sym_aattr(2048) "pcplusplus_dosubset"
set sym_ttr(4096) "invalid"
set sym_attr(8192) "subcollection"
set sym_attr(16384) "ovoperator"
set sym_attr(32768) "virtual_dtor"

#
# printAttr: convert attribute number to text representation
#
#      attr: attribute number
#

proc printAttr {attr} {
  global sym_nums sym_attr

  set r ""
  foreach n $sym_nums {
    if { $attr >= $n } {
      lappend r $sym_attr($n)
      incr attr -$n
      if { $attr == 0 } break
    }
  }
  return $r
}

#
# printType: return text representation of type
#            if type is complex return type node id, otherwise variant
#
#      type: type node id
#

proc printType {type} {
  global nodes

  if { $type == "-" } {
    return ""
  } elseif { $nodes($type,base)  != "-" || $nodes($type,dtype) != "-" || \
       $nodes($type,atype) != "-" || $nodes($type,chld)  != "-" } {
    return  $type
  } else {
    return $nodes($type,var)
  }
}

set txt_node "";  #currently selected node in text window

#
# showText: highlight node in text window
#
#     node: node to highlight
#

proc showText {node} {
  global myself
  global txtwin
  global txt_node

  # -- make sure text window is displayed on screen
  wm deiconify .${myself}txt

  # -- un-select previous node, if necessary
  if { $txt_node != "" } {
    $txtwin tag configure $txt_node  -background white -relief flat
  }

  # -- select node, make sure it is visible  in the text window
  $txtwin tag configure node$node  -background yellow -relief raised
  set txt_node node$node
  $txtwin yview -pickplace [lindex [$txtwin tag nextrange node$node 1.0] 0]
}

set bif_x 0;      #(logical) coordinates for current BIF node to draw
set bif_y -1;
set bif_lastx -1; #coordinates for last one (so we can draw the arrows)
set bif_lasty -1;

#
# genBifPos: layout BIF node graph (recursive)
#
#      bwin: pathname of BIF node graph window canvas
#      node: current node
#

proc genBifPos {bwin node} {
  global myself
  global nodes
  global bif_x bif_y bif_lastx bif_lasty
  global txtwin
  global typ_flags

  # -- draw current BIF node: rectangle containing variant and
  # -- if existing, symbol and expression fields
  incr bif_y
  $bwin create rectangle [expr $bif_x*40+5] [expr $bif_y*95+5] \
                         [expr $bif_x*40+165] [expr $bif_y*95+70] \
                         -fill gray95 -tags b$node
  $bwin create text [expr $bif_x*40+10] [expr $bif_y*95+15] \
                    -text "$nodes($node,var)" -anchor w -tags b$node

  set tx 10
  foreach f $nodes($node,flag) {
    set obj [$bwin create text [expr $bif_x*40+$tx] [expr $bif_y*95+30] \
                          -text $typ_flags($f) -anchor w]
    set tx [expr [lindex [$bwin bbox $obj] 2]+5-$bif_x*40]
  }

  if { [set symb $nodes($node,symb)] != "-" } {
    printSymb $symb $bwin [expr $bif_x*40+10] [expr $bif_y*95+45]
  }
  if { [set labl $nodes($node,labl)] != "-" } {
    printSymb $nodes($labl-L,symb) $bwin \
              [expr $bif_x*40+10] [expr $bif_y*95+45] false
  }
  foreach i {1 2 3} {
    if { [set exp $nodes($node,e$i)] != "-" } {
      $bwin create text [expr $bif_x*40-45+$i*55] \
                        [expr $bif_y*95+60] -text $exp -anchor w -tags e$exp
    }
  }

  # -- draw arrow form last node, and remember current coordinates for next time
  if { $bif_lastx != -1 } {
    $bwin create line [expr $bif_lastx*40+85] [expr $bif_lasty*95+70] \
                      [expr $bif_lastx*40+85] [expr $bif_y*95-10] \
                      [expr $bif_x*40+85]     [expr $bif_y*95-10] \
                      [expr $bif_x*40+85]     [expr $bif_y*95+5] -arrow last
  }
  set bif_lastx $bif_x
  set bif_lasty $bif_y

  # -- remember positions before "true" branch
  set start_y $bif_y
  set start_lastx $bif_lastx
  set start_lasty $bif_lasty

  # -- draw "true" branch intended by one relative to current
  # -- compute maxmimal (in max_x) reached width within "true" branch, so we
  # -- can place the "false" branch right to it
  set max_x $bif_x
  set true $nodes($node,true)
  if [llength $true] {
    incr bif_x
    incr max_x
    set max 0
    foreach n $true {
      if { [set m [genBifPos $bwin $n]] > $max_x } { set max_x $m }
    }
    incr bif_x -1
  }

  # -- draw "false" branch, if existent
  set false $nodes($node,false)
  if [llength $false] {
    # -- remember end position of "true" branch
    set end_y $bif_y
    set end_lastx $bif_lastx
    set end_lasty $bif_lasty

    # -- compute start position for "false" branch from remembered "true"
    # -- start position and maximum width reached within it
    set bif_y $start_y
    set bif_x [expr $max_x+5]
    set bif_lastx $start_lastx
    set bif_lasty $start_lasty

    # -- label "true" and "false" branch
    $bwin create text [expr $start_lastx*40+135] [expr $bif_y*95+93] -text "T"
    $bwin create text [expr $bif_x*40+95] [expr $bif_y*95+93] -text "F"

    # -- now finally draw it
    foreach n $false {
      if { [set m [genBifPos $bwin $n]] > $max_x } { set max_x $m }
    }

    # -- reset positions
    set bif_x $start_lastx
    if { $end_y > $bif_y } { set bif_y $end_y }

    # -- draw connecting line for "true" branch
    # -- rest will be drawn by next node
    $bwin create line [expr $end_lastx*40+85] [expr $end_lasty*95+70] \
                      [expr $end_lastx*40+85] [expr $bif_y*95+85]
  }
  # -- return maximal width reached within this subgraph
  return $max_x
}

set exp_x 0;  #(logical) x coordinate for current expression node

#
# genExpPos: layout expression tree (recursive)
#
#      ewin: pathname of expression tree window canvas
#      node: current node
#         y: cuurent level within tree (y coordinate)
#

proc genExpPos {ewin node y} {
  global myself
  global nodes
  global exp_x

  # -- myself is 1 level deeper in the tree than my parent
  incr y

  # -- determine lefy and right child
  set lexp $nodes($node,left)
  set rexp $nodes($node,right)

  if { $lexp != "-" && $rexp != "-" } {
    # -- current has left and right child; draw them first
    # -- then place me above them in the middle
    set l [genExpPos $ewin $lexp $y]
    set r [genExpPos $ewin $rexp $y]
    set x [expr ($l+$r)/2.0]

    # -- draw the two connecting arrows
    $ewin create line [expr $x*140+65] [expr $y*80+55] \
                      [expr $x*140+65] [expr $y*80+70] \
                      [expr $r*140+65] [expr $y*80+70] \
                      [expr $r*140+65] [expr $y*80+85] -arrow last
    $ewin create line [expr $x*140+65] [expr $y*80+70] \
                      [expr $l*140+65] [expr $y*80+70] \
                      [expr $l*140+65] [expr $y*80+85] -arrow last
  } elseif { $lexp != "-" } {
    # -- current has only left child; draw it first
    # -- then place me directly above
    set x [genExpPos $ewin $lexp $y]

    # -- draw connecting arrow marked with "L"
    $ewin create line [expr $x*140+65] [expr $y*80+55] \
                      [expr $x*140+65] [expr $y*80+85] -arrow last
    $ewin create text [expr $x*140+70] [expr $y*80+70] -text L -anchor w
  } elseif { $rexp != "-" } {
    # -- current has only right child; draw it first
    # -- then place me directly above
    set x [genExpPos $ewin $rexp $y]

    # -- draw connecting arrow marked with "R"
    $ewin create line [expr $x*140+65] [expr $y*80+55] \
                      [expr $x*140+65] [expr $y*80+85] -arrow last
    $ewin create text [expr $x*140+70] [expr $y*80+70] -text R -anchor w
  } else {
    # -- current has no children; pick next free x position for myself
    set x $exp_x
    incr exp_x
  }

  # -- now draw expression node: rectangle with variant and
  # -- symbol and type, if existent
  $ewin create rectangle [expr $x*140+5]   [expr $y*80+5] \
                         [expr $x*140+125] [expr $y*80+55] \
                         -fill gray95 -tags E$node
  $ewin create text [expr $x*140+10] [expr $y*80+15] \
                    -text "$nodes($node,var)" -anchor w -tags e$node

  if { [set t $nodes($node,symb)] != "-" } {
    printSymb $t $ewin [expr $x*140+10] [expr $y*80+30]
  } elseif { [set t $nodes($node,val)] != "-" } {
    $ewin create text [expr $x*140+10] [expr $y*80+30] -text $t -anchor w
  }
  if { [set t $nodes($node,type)] != "-" } {
    $ewin create text [expr $x*140+10] [expr $y*80+45] \
                      -text [printType $t] -anchor w -tags t$t
  }

  # -- return my own x coordinate
  return $x
}

#
# selectItem: select node in all canvases by changing its color
#
#       item: tag id of node to select
#      color: color used for selection
#

proc selectItem {item color} {
  global all_canvas

  foreach c $all_canvas {$c itemconfigure $item -fill $color}
}

#
# ldelete: remove element from list
#
#    list: name of list
#    elem: element to remove
#

proc ldelete {list elem} {
  upvar $list l

  if { [set i [lsearch -exact $l $elem]] != -1 } {
    set l [lreplace $l $i $i]
  }
}

set exp_id "";  #currently selected expression node

#
# showExpr: display selected expression tree in expression window
#
#     expr: expression node id
#

proc showExpr {expr} {
  global ewin
  global myself
  global exp_x
  global exp_id typ_id

  # -- reset or create expression tree window
  set exp_x 0
  if [winfo exists .${myself}exp] {
    $ewin delete all
  } else {
    set ewin [createWindow exp 600 600 cloneExpr]
    bind .${myself}exp <Destroy> {
      ldelete all_canvas $ewin; selectItem e$exp_id black; set exp_id ""
    }
  }
  wm title .${myself}exp "expression $expr"

  # -- draw expression tree into it
  genExpPos $ewin $expr -1
  setScrollregion $ewin

  bind $ewin c "cloneExpr $expr"

  # -- highlight selected expression in all windows (including my own one)
  if { $exp_id != "" } {selectItem e$exp_id black}
  selectItem e$expr red
  set exp_id $expr

  # -- highlight currently selected type in my own window
  $ewin itemconfigure t$typ_id -fill red
}

#
# cloneExpr: draw expression tree in extra clone window
#
#      expr: expression node id
#      

proc cloneExpr {expr} {
  global ewin
  global myself
  global exp_x
  global exp_id typ_id

  # -- draw a clone window for a expression only once
  if [winfo exists .${myself}exp-$expr] {
    raise .${myself}exp-$expr
  } else {
    # -- create clone expression window
    set exp_x 0
    set mywin [createWindow exp-$expr 600 600]
    bind .${myself}exp-$expr <Destroy> \
      "ldelete all_canvas .${myself}exp-$expr.can"
    wm title .${myself}exp-$expr "expr $expr"

    # -- draw expression tree into it
    genExpPos $mywin $expr -1
    setScrollregion $mywin
    setSize $mywin 600 600

    # -- highlight currently selected type and expression in my own window
    $mywin itemconfigure t$typ_id -fill red
    $mywin itemconfigure e$exp_id -fill red
  }
}

set typ_flags(0) "syn/protected"
set typ_flags(1) "shared/public"
set typ_flags(2) "private"
set typ_flags(3) "future"
set typ_flags(4) "virtual"
set typ_flags(5) "inline"
set typ_flags(6) "unsigned"
set typ_flags(7) "signed"
set typ_flags(8) "short"
set typ_flags(9) "long"
set typ_flags(10) "volatile"
set typ_flags(11) "const"
set typ_flags(12) "typedef"
set typ_flags(13) "extern"
set typ_flags(14) "friend"
set typ_flags(15) "static"
set typ_flags(16) "register"
set typ_flags(17) "auto"
set typ_flags(18) "global"
set typ_flags(19) "sync"
set typ_flags(20) "atomic"
set typ_flags(21) "__private"
set typ_flags(22) "restrict"

#
# genTypPos: layout type graph
#
#      twin: pathname of type graph window canvas
#      node: current node
#

proc genTypPos {twin type} {
  global nodes
  global typ_flags

  set x 0
  while { $type != "-" } {
    # -- draw type node right of last one:
    # -- rectangle with variant in it
    if { $x } {
      $twin create line [expr $x*200-55] 30 [expr $x*200+5] 30 -arrow last
      $twin create text [expr $x*200-50] 20 -text $arrowname -anchor w
    }
    set var $nodes($type,var)
    $twin create rectangle [expr $x*200+5] 5 [expr $x*200+145] 55 \
                           -fill gray95 -tags T$type
    $twin create text [expr $x*200+10] 15 -text $var -anchor w -tags t$type

    # -- depending on variant, draw more information
    # -- if type has "subtype" point variable "type" to it
    switch -exact $var {
      T_DERIVED_TEMPLATE { printSymb $nodes($type,dtype) $twin \
                                 [expr $x*200+10] 30 true
                       set exp $nodes($type,atype)
                       $twin create text [expr $x*200+10] 45 \
                                    -text $exp -anchor w -tags e$exp
                       set type "-" }
      T_DERIVED_TYPE { printSymb $nodes($type,dtype) $twin \
                                 [expr $x*200+10] 30 true
                       set type "-" }
      T_DERIVED_COLLECTION { set type $nodes($type,atype)
                       set arrowname "arg" }
      T_DESCRIPT     { # -- descriptor type, show flags set
                       set tx 0
                       set ty 0
                       foreach f $nodes($type,flag) {
                         $twin create text [expr $x*200+$tx*70+10] \
                                           [expr $ty/2*15+30] \
                               -text $typ_flags($f) -anchor w
                         set tx [expr 1-$ty]
                         incr ty
                       }
                       set type $nodes($type,base)
                       set arrowname "base"  }
      T_REFERENCE    -
      T_POINTER      { set type $nodes($type,base)
                       set arrowname "base"  }
      T_ARRAY        { set exp $nodes($type,rngs)
                       $twin create text [expr $x*200+10] 30 \
                                    -text $exp -anchor w -tags e$exp
                       set type $nodes($type,base)
                       set arrowname "base"  }
      T_CLASS        -
      T_STRUCT       -
      T_ENUM         -
      T_UNION        { # -- "structured" type: show name of structure
                       # -- plus rectangle containing all members
                       printSymb $nodes($nodes($type,parnt),symb) $twin \
                                 [expr $x*200+10] 30 false
                       set i 0
                       set next $nodes($type,chld)
                       while { $next != "-" } {
                         printSymb $next $twin [expr $x*200+20] \
                                               [expr $i*15+90]
                         incr i
                         set next $nodes($next,nchld)
                       }
                       set obj [$twin create rectangle [expr $x*200+15] 80 \
                                [expr $x*200+135] [expr $i*15+85] -fill gray95]
                       $twin lower $obj
                       $twin create line [expr $x*200+75] 55 \
                                         [expr $x*200+75] 80 -arrow last
                       set type "-"
                       set arrowname "" }
      default        { set type "-"
                       set arrowname "" }
    }
    incr x
  }
}

set typ_id ""; #currently selected type

#
# showType: display selected type graph in graph window
#
#     type: type node id
#

proc showType {type} {
  global twin
  global myself
  global typ_id exp_id

  # -- create or reset type graph window
  if [winfo exists .${myself}typ] {
    $twin delete all
  } else {
    set twin [createWindow typ 600 150 "cloneType"]
    bind .${myself}typ <Destroy> {
      ldelete all_canvas $twin; selectItem t$typ_id black; set typ_id ""
    }
  }
  wm title .${myself}typ "type $type"

  # -- draw type graph onto it
  genTypPos $twin $type
  setScrollregion $twin

  bind $twin c "cloneType $type"

  # -- highlight selected type in all windows (including my own one)
  if { $typ_id != "" } {selectItem t$typ_id black}
  selectItem t$type red
  set typ_id $type

  # -- highlight currently selected expression in my own window
  $twin itemconfigure e$exp_id -fill red
}

#
# cloneType: draw type graph in extra clone window
#
#      type: type node id
#      

proc cloneType {type} {
  global myself
  global typ_id exp_id

  # -- draw a clone window for a type only once
  if [winfo exists .${myself}typ-$type] {
    raise .${myself}typ-$type
  } else {
    # -- create clone type window
    set mywin [createWindow typ-$type 600 150]
    bind .${myself}typ-$type <Destroy> \
      "ldelete all_canvas .${myself}typ-$type.can"
    wm title .${myself}typ-$type "type $type"

    # -- draw type graph into it
    genTypPos $mywin $type
    setScrollregion $mywin
    setSize $mywin 600 150

    # -- highlight currently selected type and expression in my own window
    $mywin itemconfigure e$exp_id -fill red
    $mywin itemconfigure t$typ_id -fill red
  }
}

#
# createWindow: create new toplevel window
#
#          win: prefix to be used for pathname of new window
# width,height: size of new window
#     clonecom: command for cloning the window, also used for
#               determining window setup (clone windows are different)
#

proc createWindow {win width height {clonecom {}}} {
  global TAUDIR
  global myself
  global all_canvas

  toplevel .$myself$win
  wm minsize .$myself$win 50 50
  wm iconbitmap .$myself$win @$TAUDIR/xbm/$myself.xbm


  # -- create main drawing canvas
  canvas .$myself$win.can -width $width -height $height -background white
  bind .$myself$win.can <2> ".$myself$win.can scan mark %x %y"
  bind .$myself$win.can <B2-Motion> ".$myself$win.can scan dragto %x %y"

  if { $clonecom != {} } {
    # -- NO CLONE WINDOW: create menubar on top and add scrollbars
    frame .$myself$win.mbar -relief raised -borderwidth 2
    menubutton .$myself$win.mbar.b1 -text File -menu .$myself$win.mbar.b1.m1 \
                                    -underline 0
    menu .$myself$win.mbar.b1.m1
    if { $clonecom != {} && $clonecom != "noclone" } {
      .$myself$win.mbar.b1.m1 add command -label "Clone" -underline 0 \
                     -command "$clonecom \[lindex \[wm title .$myself$win\] 1\]"
    }
    .$myself$win.mbar.b1.m1 add command -label "Print" -underline 0 -command \
           "printCanvas .$myself$win.can \[lindex \[wm title .$myself$win\] 1\]"
    .$myself$win.mbar.b1.m1 add command -label "Quit" -underline 0 \
                            -command "destroy .$myself$win"

    menubutton .$myself$win.mbar.b2 -text Help -menu .$myself$win.mbar.b2.m1 \
                                -underline 0
    menu .$myself$win.mbar.b2.m1
    .$myself$win.mbar.b2.m1 add command -label "on $myself" -underline 3 \
                     -command "showHelp $myself 1-$myself 1"
    .$myself$win.mbar.b2.m1 add command -label "on bindings" -underline 3 \
                     -command "showHelp $myself 1.6-bind 1"
    .$myself$win.mbar.b2.m1 add command -label "on commands" -underline 3 \
                     -command "showHelp $myself 1.7-com 1"
    .$myself$win.mbar.b2.m1 add separator
    .$myself$win.mbar.b2.m1 add command -label "on using help" -underline 3 \
                     -command "showHelp general 1-help 1"

    pack .$myself$win.mbar.b1 -side left -padx 5
    pack .$myself$win.mbar.b2 -side right -padx 5
    pack .$myself$win.mbar    -side top -fill x

    frame .$myself$win.f1
    scrollbar .$myself$win.hscroll -orient horiz -relief sunken \
              -command ".$myself$win.can xview"
    .$myself$win.can configure -xscroll ".$myself$win.hscroll set"

    frame .$myself$win.f2 -width 20 -height 20

    pack .$myself$win.hscroll -in .$myself$win.f1 -side left -expand yes -fill x
    pack .$myself$win.f2      -in .$myself$win.f1 -side right

    scrollbar .$myself$win.vscroll -orient vert -relief sunken \
              -command ".$myself$win.can yview"
    .$myself$win.can configure -yscroll ".$myself$win.vscroll set"

    pack .$myself$win.f1      -side bottom -fill x
    pack .$myself$win.vscroll -side right  -fill y
    pack .$myself$win.can     -side top -padx 15 -pady 15 -fill both -expand yes

    tk_menuBar .$myself$win.mbar .$myself$win.mbar.b1 .$myself$win.mbar.b2
  } else {
    # -- CLONE WINDOW: just add two buttons on the bottom
    button .$myself$win.b1 -text print -command \
           "printCanvas .$myself$win.can \[lindex \[wm title .$myself$win\] 1\]"
    button .$myself$win.b2 -text close -command "destroy .$myself$win"
    pack .$myself$win.can -side top -fill both -expand yes
    pack .$myself$win.b1 .$myself$win.b2 -side left -fill x -expand yes
  }

  # -- add all necessary bindings
  bind .$myself$win.can <1> {getTag %W 1}
  bind .$myself$win.can <2> {+getTag %W 2}
  bind .$myself$win.can <3> {getTag %W 3}
  bind .$myself$win.can <Shift-3> {getTag %W S3}

  bind .$myself$win.can <Enter> "focus .$myself$win.can"
  bind .$myself$win.can <Leave> "focus ."
  bind .$myself$win.can q "destroy .$myself$win"
  bind .$myself$win.can Q exit
  bind .$myself$win.can p \
       "printCanvas .$myself$win.can \[lindex \[wm title .$myself$win\] 1\]"

  lappend all_canvas .$myself$win.can
  return .$myself$win.can
}

#
# getTag: process click of mouse button (used in createWindow bindings)
#
#    win: pathname of window where the mouse click occurred
# button: button pressed
#

proc getTag {win button} {
  # -- test whether object nearest to mouse click (available through
  # -- special Tk tag "current") represents a BIF, TYPE, Expression
  # -- or SYMBOL node. If yes, call corresponding show/clone routine
  if [regexp {[0-9]+-[BEST]} [$win gettags current] node] {
    if { $button == 3 } {
      showText $node
    } elseif { $button == "S3" } {
      if [string match {*B} $node] { showFile $node }
    } else {
      if { $button == 1 } {set com show} else {set com clone}

      if [string match {*E} $node] {
        ${com}Expr $node
      } elseif [string match {*T} $node] {
        ${com}Type $node
      }
    }
  }
}

set file_id "";  #currently displayed file node id

#
# showFile: show source text of BIF node in source text window
#
#     node: BIF node id
#

proc showFile {node} {
  global TAUDIR
  global myself
  global nodes
  global depdir
  global file_id

  if { [set start $nodes($node,line)] == 0 } return
  set fid $nodes($node,file)

  if [regexp '^/' $nodes($fid)] {
    set file $nodes($fid)
  } else {
    set file $depdir/$nodes($fid)
  }

  if { ! [file readable $file] } {
    showError "Cannot open file: `$file'."
    return
  }

  if { ! [winfo exists .viewer] } {
    # -- text window does not yet exist, create and display one
    toplevel .viewer
    wm title .viewer [file tail $file]
    wm minsize .viewer 50 50
    wm iconbitmap .viewer @$TAUDIR/xbm/$myself.xbm

    text .viewer.t1 -width 80 -height 32 -background white -foreground black
    scrollbar .viewer.s1 -orient vert -relief sunken \
                    -command ".viewer.t1 yview"
    .viewer.t1 configure -yscroll ".viewer.s1 set"
    button .viewer.bu -text "close" -command "destroy .viewer"

    pack .viewer.bu -side bottom -fill x
    pack .viewer.s1 -side right  -fill y
    pack .viewer.t1 -side top -expand yes -fill both

    bind .viewer.t1 <Enter> "focus .viewer.t1"
    bind .viewer.t1 <Leave> "focus ."
    bind .viewer.t1 q "destroy .viewer"
    bind .viewer.t1 Q exit

    # -- read necessay file
    readFile .viewer.t1 $file
  } else {
    # -- there is already a text window
    # -- un-highlight old selection
    .viewer.t1 tag delete statement

    if { $fid != $file_id } {
      # -- but different file; update title
      wm title .viewer [file tail $file]
      .viewer.t1 configure -state normal
      .viewer.t1 delete 1.0 end

      # -- read new file
      readFile .viewer.t1 $file
    }
  }
  # -- highlight select BIF node
  .viewer.t1 tag add statement "$start.0" "$start.0 lineend"
  .viewer.t1 tag configure statement -background yellow -relief raised
  .viewer.t1 yview -pickplace $start.0

  # -- remember currently selected item and its file
  set file_id $fid
}

#
# readFile: read text of file into text window
#
#      win: pathname of text window
#     file: UNIX pathname of file to read
#

proc readFile {win file} {
  set in [open $file r]
  $win insert end [read $in]
  close $in
  $win configure -state disabled
}

#
# createTxtWindow: create text window for dumpdep output
#
#            file: name of depfile
#

proc createTxtWindow {file} {
  global TAUDIR
  global myself
  global txt_node

  toplevel .${myself}txt
  wm title .${myself}txt "dumpdep $file"
  wm minsize .${myself}txt 50 50
  wm iconbitmap .${myself}txt @$TAUDIR/xbm/$myself.xbm

  text .${myself}txt.t1 -width 120 -height 8 \
                        -background white -foreground black \
                        -exportselection false -selectbackground white \
                        -selectborderwidth 0

  scrollbar .${myself}txt.s1 -orient vert -relief sunken \
                  -command ".${myself}txt.t1 yview"
  .${myself}txt.t1 configure -yscroll ".${myself}txt.s1 set"
  button .${myself}txt.bu -text "close" -command "wm withdraw .${myself}txt"

  pack .${myself}txt.bu -side bottom -fill x
  pack .${myself}txt.s1 -side right  -fill y
  pack .${myself}txt.t1 -side top -expand yes -fill both

  # -- don't show in the beginning. window will be mapped
  # -- if showText is called the first time
  wm withdraw .${myself}txt

  # -- setup bindings for text window
  bind .${myself}txt.t1 <Button-1> {getNode %x %y 1}
  bind .${myself}txt.t1 <Button-2> {getNode %x %y 2}
  bind .${myself}txt.t1 <Button-3> {getNode %x %y 3}

  bind .${myself}txt.t1 <Enter> "focus .${myself}txt.t1"
  bind .${myself}txt.t1 <Leave> "focus ."
  bind .${myself}txt.t1 q "wm withdraw .${myself}txt"
  bind .${myself}txt.t1 Q exit

  return .${myself}txt.t1
}

#
# getNode: process click of mouse button (used in createTxtWindow bindings)
#
#    x,y: position within text window where the mouse click occurred
# button: button pressed
#

proc getNode {x y button} {
  global txtwin

  # -- get text near mouse click (6 characters before and afterwards)
  # -- and check whether it represents a BIF, TYPE, Expression
  # -- or SYMBOL node. If yes, call corresponding show/clone routine
  if [regexp {[0-9]+-[BESTF]} [$txtwin get "@$x,$y-6c" "@$x,$y+7c"] node] {
    if { $button == 2 } {
      if [string match {*E} $node] {
        cloneExpr $node
      } elseif [string match {*T} $node] {
        cloneType $node
      }
    } else {
      showText $node
      if { $button == 3 } {
        if [string match {*E} $node] {
          showExpr $node
        } elseif [string match {*T} $node] {
          showType $node
        }
      }
    }
  }
}

#
# setSize: change size of canvas after drawing, so that the canvas is
#          just big enough to hold the graph, at least up to a specific
#          maximum size given by (maxw, maxh)
#
#     win: pathname of canvas
#

proc setSize {win maxw maxh} {
  set bbox [$win bbox all]
  set x [lindex $bbox 0]
  set w [expr [lindex $bbox 2]+$x]
  set h [expr [lindex $bbox 3]+$x]
  $win configure -width [expr $w>$maxw ? $maxw : $w]
  $win configure -height [expr $h>$maxh ? $maxh : $h]
}

#
# setScrollregion: set scrollregion of a window, so that it is
#                  just big enough to hold the graph
#
#             win: pathname of canvas
#

proc setScrollregion {win} {
  set bbox [$win bbox all]
  set x [lindex $bbox 0]
  $win configure -scrollregion \
       [list 0 0 [expr [lindex $bbox 2]+$x] [expr [lindex $bbox 3]+$x]]
}

# ------------
# -- main code
# ------------

if {$argc != 1} {
  puts "usage: $myself depfile"
  puts ""
  puts "Use the \"Help\" menu of the main window"
  puts "to learn more on how to use dumpy."
  exit
} else {
  # -- create invisible text window and read output of dumpdep into it
  set depfile [lindex $argv 0]
  set depdir [file dirname $depfile]
  set txtwin [createTxtWindow $depfile]
  readDump $depfile

  # -- create main BIF graph window
  set bwin [createWindow bif 550 750 noclone]
  bind .${myself}bif <Destroy> {ldelete all_canvas $bwin}
  wm title .${myself}bif "DUMPY $depfile"
  genBifPos $bwin 1-B
  setScrollregion $bwin

  # -- wait for destroy of BIF window, then exit
  tkwait window .${myself}bif
  exit
}
