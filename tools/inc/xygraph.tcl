#*********************************************************************#
#*                TAU/pC++/Sage++  Copyright (C) 1994                *#
#*                        Jerry Manic Saftware                       *#
#*  Indiana University  University of Oregon  University of Rennes   *#
#*********************************************************************#

#
# NewXYGraph:
#
#        win:
#      sizeX:
#      sizeY:
#

proc NewXYGraph {win sizeX sizeY} {
  global XYGraph

  canvas $win -width [expr $sizeX+300] -height [expr $sizeY+100]
  $win create rectangle 100 50 [expr $sizeX+100] [expr $sizeY+50] -tag xy0
  $win configure -scrollregion [list 0 0 [expr $sizeX+300] [expr $sizeY+100]]

  set XYGraph($win,sizeX)   $sizeX
  set XYGraph($win,sizeY)   $sizeY
  set XYGraph($win,tag)     2
  set XYGraph($win,bg)      [lindex [$win configure -bg] 4]

  return $win
}

proc ResetGraph {win} {
  global XYGraph

  set sizeX $XYGraph($win,sizeX)
  set sizeY $XYGraph($win,sizeY)

  $win delete all
  $win create rectangle 100 50 [expr $sizeX+100] [expr $sizeY+50] -tag xy0
  set XYGraph($win,tag)     2
}

#
# SetXAxis:
#
#      win:
#     minX:
#     maxX:
#

proc SetXAxis {win {minX 0.0} {maxX 1.0}} {
  global XYGraph

  set XYGraph($win,minX)    $minX
  set XYGraph($win,maxX)    $maxX
  set XYGraph($win,rangeX)  [expr double($maxX-$minX)]
}

#
# SetYAxis:
#
#      win:
#     minY:
#     maxY:
#

proc SetYAxis {win {minY 0.0} {maxY 1.0}} {
  global XYGraph

  set XYGraph($win,minY)    $minY
  set XYGraph($win,maxY)    $maxY
  set XYGraph($win,rangeY)  [expr double($maxY-$minY)]
}

#
# SetTitles:
#
#       win:
#     title:
#      subX:
#      subY:
#

proc SetTitles {win {title {}} {subX {}} {subY {}}} {
  global XYGraph

  set sizeX $XYGraph($win,sizeX)
  set sizeY $XYGraph($win,sizeY)

  $win create text [expr 100+$sizeX/2.0] 10 -text $title -anchor n -tag xy1 \
                   -font -Adobe-Helvetica-Bold-R-Normal--*-120-*
  $win create text [expr 100+$sizeX/2.0] [expr 95+$sizeY] -text $subX \
                   -anchor s -tag xy1 \
                   -font -Adobe-Helvetica-Bold-R-Normal--*-120-*
  $win create text 5 [expr 50+$sizeY/2.0] -text $subY -anchor w -tag xy1 \
                   -font -Adobe-Helvetica-Bold-R-Normal--*-120-*
}

#
# SetLegendTitle:
#
#            win:
#          title:
#

proc SetLegendTitle {win title} {
  global XYGraph

  set sizeX $XYGraph($win,sizeX)
  set sizeY $XYGraph($win,sizeY)

  $win create text [expr 150+$sizeX] 50 -text $title -anchor w -tag xy1 \
              -font -Adobe-Helvetica-Bold-R-Normal--*-120-*
}

#
# X2graph:
#
#     win:
#       x:
#

proc X2graph {win x} {
  global XYGraph

  if { $x < $XYGraph($win,minX) } {
    return 100
  } elseif { $x > $XYGraph($win,maxX) } {
    return [expr 100+$XYGraph($win,sizeX)]
  } else {
    return [expr (($x - $XYGraph($win,minX)) / $XYGraph($win,rangeX) \
            * $XYGraph($win,sizeX)) + 100]
  }
}

#
# Y2graph:
#
#     win:
#       y:
#

proc Y2graph {win y} {
  global XYGraph

  if { $y < $XYGraph($win,minY) } {
    return [expr 50+$XYGraph($win,sizeY)]
  } elseif { $y > $XYGraph($win,maxY) } {
    return 50
  } else {
    return [expr 50 + $XYGraph($win,sizeY) - \
            (($y - $XYGraph($win,minY)) / $XYGraph($win,rangeY) \
            * $XYGraph($win,sizeY))]
  }
}

#
# AddXLabels:
#
#        win:
#       args:
#

proc AddXLabels {win args} {
  global XYGraph

  set t $XYGraph($win,tag)
  incr XYGraph($win,tag)
  set y $XYGraph($win,sizeY)
  foreach a $args {
    set x [X2graph $win $a]
    $win create line $x [expr 50+$y] $x [expr 60+$y] -tag xy$t
    $win create text $x [expr 65+$y] -text $a -anchor n -tag xy$t \
                -font -Adobe-Helvetica-Bold-R-Normal--*-120-*
  }
  return xy$t
}

#
# AddYLabels:
#
#        win:
#       args:
#

proc AddYLabels {win args} {
  global XYGraph

  set t $XYGraph($win,tag)
  incr XYGraph($win,tag)
  foreach a $args {
    set y [Y2graph $win $a]
    $win create line 90 $y 100 $y -tag xy$t
    $win create text 85 $y -text $a -anchor e -tag xy$t \
                -font -Adobe-Helvetica-Bold-R-Normal--*-120-*
  }
  return xy$t
}

#
# AddLegendItem:
#
#           win:
#             n:
#         value:
#           obj:
#           col:
#        symbol:
#

proc AddLegendItem {win n value obj col {symbol {}}} {
  global XYGraph

  set sizeX $XYGraph($win,sizeX)
  set sizeY $XYGraph($win,sizeY)

  set x0 [expr 170+$sizeX]
  set x1 [expr 190+$sizeX]
  set y  [expr 70+$n*20]

  set o [$win create line $x0 $y $x1 $y -fill $col -tag xy1]
  $win bind $o <1> "$win raise $obj"
  if { $symbol != "" } {
    $win create text $x0 $y -text $symbol -anchor c -fill $col -tag xy1 \
                -font -Adobe-Helvetica-Bold-R-Normal--*-120-*
    $win create text $x1 $y -text $symbol -anchor c -fill $col -tag xy1 \
                -font -Adobe-Helvetica-Bold-R-Normal--*-120-*
  }
  set o [$win create text [expr 200+$sizeX] $y -text $value -anchor w -tag xy1 \
                     -font -Adobe-Helvetica-Bold-R-Normal--*-120-*]
  $win bind $o <1> "$win raise $obj"
}

#
# AddLine:
#
#     win:
#     col:
#    type:
#    xvec:
#    yvec:
#  symbol:
#    etag:
#

proc AddLine {win col type xvec yvec {symbol {}} {etag {}}} {
  global XYGraph

  set tg $XYGraph($win,tag)
  incr XYGraph($win,tag)

  if { $etag == "" } {
    set tt xy$tg
  } else {
    set tt [list xy$tg $etag]
  }

  set line 0
  set pnt  0
  set fill 0
  set hgth 0

  foreach t [split $type {}] {
    switch $t {
      L  { set line 1 }
      P  { set pnt  1 }
      F  { set fill 1 }
      H  { set hgth 1 }
    }
  }

  if { $fill } {
    lappend fco [X2graph $win [lindex $xvec 0]] \
                [expr 100+$XYGraph($win,sizeY)]
  }

  set i 0
  foreach x $xvec {
    set y [lindex $yvec $i]
    set X [X2graph $win $x]
    set Y [Y2graph $win $y]

    if { $pnt } {
      set o [$win create text $X $Y -text "  $symbol  " -fill $col \
             -anchor c -tag $tt -font -Adobe-Helvetica-Bold-R-Normal--*-120-*]
      $win bind $o <ButtonPress-1> "showVal $win $X $Y $x $y"
      $win bind $o <ButtonRelease-1> "$win delete valobj"
    }
    if { $line } {
      lappend co $X $Y
    }
    if { $fill } {
      lappend fco $X $Y
    }
    if { $hgth } {
      $win create line $X $Y $X [expr 50+$XYGraph($win,sizeY)] \
                  -fill $col -tag $tt
    }
    incr i
  }

  if { $line  && [llength $co] >= 4 } {
    eval $win create line $co -fill $col -tag \"$tt\"
  }
  if { $fill && [llength $fco] >= 6 } {
    lappend fco $X [expr 100+$XYGraph($win,sizeY)]
    eval $win create polygon $fco -fill $col -tag \"$tt\"
  }
  return xy$tg
}

proc showVal {win X Y x y} {
  global XYGraph

  set o [$win create text [expr $X - 3] [expr $Y + 3] -text "( $x, $y )" \
         -anchor ne -tag valobj -font -Adobe-Helvetica-Bold-R-Normal--*-120-*]
  set bbox [$win bbox $o]
  $win create rectangle \
             [expr [lindex $bbox 0]-1] [expr [lindex $bbox 1]-1] \
             [expr [lindex $bbox 2]+1] [expr [lindex $bbox 3]+1] \
             -fill $XYGraph($win,bg) -tag valobj -outline {}
  $win raise $o
}

# ---------------------------------------------------------------------- #
#  The following functions allow computing of "pretty" values
#  for labeling x-y graph axes. The interface is the function pretty;
#  the functions starting with "xy" are helper functions.
#  I translated the code long ago from Fortran to C, and now to Tcl.
#  I don't know how the algorithm works but it seems to work
#                                                   Bernd Mohr
# ---------------------------------------------------------------------- #

#
# pretty: compute "pretty" values for labeling xaxes of a plot
#         returns vector of label values
#
#    min: name of variable containing minimum value
#    max: name of variable containing maximum value
#     no: name of variable containing number of requested values (hint)
#
#    min, max, no will contain the "new" min, max, number after return
#

proc Pretty {min max no} {
  upvar $min pmin
  upvar $max pmax
  upvar $no pno

  if {$pno <= 0} {set pno 5}
  set delta [xydist $pmin $pmax $pno]
  xylim $pmin $pmax $delta pmin pmax pno

  set values ""
  for {set i 0} {$i<$pno} {incr i} {
    lappend values [format "%g" [expr $pmin + $i * $delta]]
  }
  lappend values [format "%g" $pmax]
  incr pno
  return $values
}

proc xymin {x y} { if { $x < $y } {return $x} else {return $y} }
proc xymax {x y} { if { $x > $y } {return $x} else {return $y} }

proc xydist {min max no} {
  set vint {1.0 2.0 5.0 10.0}
  set sqr  {1.414214 3.162278 7.071068}

  set umin [xymin $min $max]
  set umax [xymax $min $max]

  if { $umin == $umax } {
    set dd [xymax [expr 0.25*abs($umin)] 1.0e-2]
    set umin [expr $umin - $dd]
    set umax [expr $umax + $dd]
    set no 1
  }

  set a [expr ($umax - $umin) / double($no)]
  set na1 [expr int(log10($a))]
  if { $a < 1.0 } {incr na1 -1}
  set b [expr $a / pow(10.0,$na1)]
  for {set i 0} {$i<3} {incr i} { if {$b < [lindex $sqr $i]} break }

  return [expr [lindex $vint $i] * pow(10.0,$na1)]
}

proc xylim {umin umax delta min max no} {
  upvar $min pmin
  upvar $max pmax
  upvar $no pno

  set dmin [expr double($umin) / $delta]
  set imin [expr int($dmin)]
  if {$dmin < 0.0} {incr imin -1}
  if {abs([expr $imin + 1.0 - $dmin]) < 0.0002} {incr imin}
  set pmin [expr $delta * $imin]
  if {$pmin > $umin} {set pmin $umin}

  set dmax [expr double($umax) / $delta]
  set imax [expr int($dmax + 1.0)]
  if {$dmax < -1.0} {incr imax -1}
  if {abs([expr $dmax + 1.0 - $imax]) < 0.0002} {incr imax -1}
  set pmax [expr $delta * $imax]
  if {$pmax < $umax} {set pmax $umax}

  set pno [expr abs($imax - $imin)]
}
