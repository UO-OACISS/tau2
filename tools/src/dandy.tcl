#!/local/hp700-ux/bin/wish -f

set Colors(0)  #00000000FFFF
set Colors(1)  #00003333FFFF
set Colors(2)  #00006666FFFF
set Colors(3)  #00009999FFFF
set Colors(4)  #0000CCCCFFFF
set Colors(5)  #0000FFFFFFFF
set Colors(6)  #0000FFFFCCCC
set Colors(7)  #0000FFFF9999
set Colors(8)  #0000FFFF6666
set Colors(9)  #0000FFFF0000
set Colors(10) #6666FFFF0000
set Colors(11) #9999FFFF0000
set Colors(12) #CCCCFFFF0000
set Colors(13) #FFFFFFFF0000
set Colors(14) #FFFFCCCC0000
set Colors(15) #FFFF99990000
set Colors(16) #FFFF66660000
set Colors(17) #FFFF33330000
set Colors(18) #FFFF00000000

proc ReadBreezyHeader {} {
  global mat_data

  gets stdin line  ;# num dimension collection
  scan $line "%d" cdim

  gets stdin line  ;# x dimension
  scan $line "%d" cx

  if { $cdim > 1 } {
    gets stdin line  ;# y dimension
    scan $line "%d" cy
    if { $cy == 1 } { incr cdim -1 }
  } else {
    set cy 1
  }
  if { $cx == 1 } { incr cdim -1 }

  gets stdin line  ;# type

  gets stdin line  ;# num dimension element
  scan $line "%d" edim

  if { $edim == 0 } {
    set ex 1
    set ey 1
  } else {
    gets stdin line  ;# x dimension
    scan $line "%d" ex
    if { $edim > 1 } {
      gets stdin line  ;# y dimension
      scan $line "%d" ey
      if { $ey == 1 } { incr edim -1 }
    } else {
      set ey 1
    }
    if { $ex == 1 } { incr edim -1 }
  }

  set dim [expr $cdim+$edim]
  if { $dim > 2 } {
    puts stderr "Error: cannot process more than 2 dimensions!"
    exit
  }

  set mat_data(x) 1
  set mat_data(y) 1
  set mat_data(dimx) ""
  set mat_data(dimy) ""
  set next x
  set dimlist { cx cy ex ey }
  foreach d $dimlist {
    if { [expr $$d] > 1 } {
      set mat_data($next) [expr $$d]
      set mat_data(dim$next) "$d "
      set next y
    }
  }
  set mat_data(parts) [expr $cx*$cy]
}

proc InitMat {} {
  global myself TAUDIR
  global mat_data mat_obj Colors

  set mat_data(msg) " Init ..."

  if {$mat_data(x)>$mat_data(y)} {set m $mat_data(x)} else {set m $mat_data(y)}
  set mat_data(size) [expr 500/$m]
  set mat_data(slice) -1
  set mat_data(last) -1

  set x $mat_data(x)
  set y $mat_data(y)
  set s $mat_data(size)

  toplevel .$myself
  wm title .$myself "DANDY"
  wm iconbitmap .$myself @$TAUDIR/xbm/$myself.xbm

  label .$myself.message -textvariable mat_data(msg) -relief raised -anchor w
  pack .$myself.message -side top -ipadx 5 -ipady 5 -fill x

  canvas .$myself.mat -height [expr $y*$s] -width [expr $x*$s]
  pack .$myself.mat -side left -padx 10 -pady 10

  label .$myself.lab -textvariable mat_data(v) -width 20
  pack .$myself.lab -side top -padx 10 -pady 10

  frame .$myself.range
  canvas .$myself.range.scale -height 190 -width 10
  for {set i 0} {$i<19} {incr i} {
    .$myself.range.scale create rectangle 0 [expr $i*10] 10 [expr $i*10+10] \
                        -fill $Colors([expr 18-$i]) -outline ""
  }

  entry .$myself.range.cmax -textvariable mat_data(max) -width 14
  label .$myself.range.lmax -textvariable mat_data(labmax) -width 14 -anchor w
  pack .$myself.range.lmax .$myself.range.cmax -side top -anchor w \
       -in .$myself.range -padx 30

  entry .$myself.range.cmin -textvariable mat_data(min) -width 14
  label .$myself.range.lmin -textvariable mat_data(labmin) -width 14 -anchor w
  pack .$myself.range.lmin .$myself.range.cmin -side bottom -anchor w \
       -in .$myself.range -padx 30
  pack .$myself.range.scale -side left -in .$myself.range
  update

  button .$myself.b1 -text "quit" -command exit -width 12
  button .$myself.b2 -text "redraw" -command DrawMat -width 12
  button .$myself.b3 -text "next" -command ReadData -width 12
  button .$myself.b3a -text "prev" -command "ReadData -1" -width 12 \
         -state disabled
  button .$myself.b4 -text "step off" -command StepToggle -width 12
  button .$myself.b5 -text "block mode" -command ModeToggle -width 12
  pack .$myself.b1 .$myself.b2 .$myself.b3a .$myself.b3 .$myself.b4 \
       .$myself.b5 -side bottom -padx 5 -pady 10
  pack .$myself.range -side bottom -pady 50

  for {set i 0} {$i<$x} {incr i} {
    for {set j 0} {$j<$y} {incr j} {
      set mat_obj($i,$j) [.$myself.mat create rectangle \
                  [expr $i*$s] [expr $j*$s] \
                  [expr $i*$s+$s] [expr $j*$s+$s] \
                   -fill white -outline ""]
    }
    update
  }
  update

  bind .$myself.mat <B1-ButtonPress> {ShowVal %x %y}
  bind .$myself.mat <B1-Motion> {ShowVal %x %y}
  bind .$myself.mat <B1-ButtonRelease> {set show_val(on) 0}
  set mat_data(msg) " Init ... done"
}

set show_val(on) 0

proc ShowVal {x y} {
  global mat_data show_val

  set show_val(x) $x
  set show_val(y) $y
  set show_val(on) 1

  set lx $mat_data(dimx)
  set ly $mat_data(dimy)
  set i [expr $x/$mat_data(size)]
  set j [expr $y/$mat_data(size)]
  if { $i>=0 && $i<$mat_data(x) && $j>=0 && $j<$mat_data(y) } {
    set mat_data(v) \
            "($lx$i,$ly$j) = [lindex $mat_data(vals) [expr $i*$mat_data(x)+$j]]"
  }
}

set step_mode 1

proc StepToggle {} {
  global myself
  global step_mode

  set step_mode [expr 1-$step_mode]
  if { $step_mode } {
    .$myself.b3 configure -text "next" -command ReadData
    .$myself.b4 configure -text "step: on"
    .$myself.b3a configure -state normal
  } else {
    .$myself.b3 configure -text "start" -command StartReadData
    .$myself.b4 configure -text "step: off"
    .$myself.b3a configure -state disabled
  }
}

set line_mode 1

proc ModeToggle {} {
  global myself
  global line_mode

  set line_mode [expr 1-$line_mode]
  if { $line_mode } {
    .$myself.b5 configure -text "block mode"
  } else {
    .$myself.b5 configure -text "line mode"
  }
}

proc DrawMat {} {
  global myself
  global mat_data mat_obj Colors line_mode show_val

  set mat_data(msg) " slice $mat_data(slice): Reading ... Drawing ..."
  set x $mat_data(x)
  set y $mat_data(y)
  set m $mat_data(min)
  set r [expr $mat_data(max) - $m]

  set i 0
  set j 0
  foreach v $mat_data(vals) {
    if { $v < $mat_data(min) } {
      set col "white"
    } elseif { $v > $mat_data(max) } {
      set col "black"
    } elseif { $r } {
      set col $Colors([expr int(double($v-$m)/$r*18)])
    } else {
      set col $Colors(0)
    }
    .$myself.mat itemconfigure $mat_obj($i,$j) -fill $col
    incr j
    if { $j == $y } {
      set j 0
      incr i
      if { $line_mode } { update }
    }
  }
  update
  set mat_data(msg) " slice $mat_data(slice): Reading ... Drawing ... done"
  if { $show_val(on) } { ShowVal $show_val(x) $show_val(y) }
}

proc ReadLine {buf} {
  global mat_data
  upvar $buf b

  set b ""
  set p $mat_data(parts)

  for {set i 0} {$i<$p} {incr i} {
    if { [gets stdin line] < 0 } {return -1}
    append b " " $line
  }
  return 1
}

proc ReadData {{next 1}} {
  global myself
  global mat_data old_data
  global go_ahead
  global errorInfo

  set WINDOWLEN 5

  incr mat_data(slice) $next

  if { $mat_data(slice) <= 0 ||
       $mat_data(slice) <= [expr $mat_data(last) - $WINDOWLEN] } {
    .$myself.b3a configure -state disabled
  } else {
    .$myself.b3a configure -state normal
  }

  if [info exists old_data($mat_data(slice))] {
    set mat_data(vals) $old_data($mat_data(slice))
  } elseif { [ReadLine line] >= 0 } {
    set mat_data(vals) $line
    set old_data($mat_data(slice)) $line

    if { $mat_data(slice) > $WINDOWLEN } {
      unset old_data([expr $mat_data(slice) - $WINDOWLEN - 1])
    }
    set mat_data(last) $mat_data(slice)
  } else {
    set mat_data(msg) " NO MORE DATA!!!"
    set go_ahead 0
    return
  }

  set mat_data(msg) " slice $mat_data(slice): Reading ..."

  set should [expr $mat_data(x)*$mat_data(y)]
  set is [llength $mat_data(vals)]
  if { $is > $should } {
    set errorInfo $mat_data(vals)
    tkerror "Too many data items ($is instead of $should)"
    return
  } elseif { $is < $should } {
    set errorInfo $mat_data(vals)
    tkerror "Too few data items ($is instead of $should)"
    return
  }

  set min [lindex $mat_data(vals) 0]
  set max [lindex $mat_data(vals) 0]

  if { $mat_data(rmin) == " " } {
    set mat_data(rmin) $min
    set mat_data(rmax) $max
  }

  foreach v $mat_data(vals) {
    if { $v < $min } { set min $v }
    if { $v > $max } { set max $v }
  }

  if { $min < $mat_data(rmin) } { set mat_data(rmin) $min }
  if { $max > $mat_data(rmax) } { set mat_data(rmax) $max }


  if { $mat_data(min) == " " } {
    set mat_data(min) $min
    set mat_data(max) $max
  }

  set mat_data(labmin) [format "\[%.2f/%.2f\]" $min $mat_data(rmin)]
  set mat_data(labmax) [format "\[%.2f/%.2f\]" $max $mat_data(rmax)]

  DrawMat
}

set go_ahead 1

proc StartReadData {} {
  global myself
  global go_ahead

  set go_ahead 1
  .$myself.b3 configure -text "stop" -command StopReadData
  while { $go_ahead } { ReadData }
}

proc StopReadData {} {
  global myself
  global go_ahead

  set go_ahead 0
  .$myself.b3 configure -text "start" -command StartReadData
}

set mat_data(min)  " "
set mat_data(rmin) " "
set mat_data(parts) 1
set mat_data(dimx) "x "
set mat_data(dimy) "y "

switch -exact $argc {
  1       { if { [lindex $argv 0] == "-b" || [lindex $argv 0] == "-bheader" } {
              ReadBreezyHeader
            } else {
              puts stderr "usage: $argv0 (-bheader | x y) \[min max\]"
              exit
            }
          }
  2       { set mat_data(x)   [lindex $argv 0]
            set mat_data(y)   [lindex $argv 1]
          }
  3       { ReadBreezyHeader
            set mat_data(min) [lindex $argv 1]
            set mat_data(max) [lindex $argv 2]
          }
  4       { set mat_data(x)   [lindex $argv 0]
            set mat_data(y)   [lindex $argv 1]
            set mat_data(min) [lindex $argv 2]
            set mat_data(max) [lindex $argv 3]
          }
  default { puts stderr "usage: $argv0 (-bheader | x y) \[min max\]"
            exit
          }
}
InitMat
