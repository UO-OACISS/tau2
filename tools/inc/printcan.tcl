#*********************************************************************#
#*                TAU/pC++/Sage++  Copyright (C) 1994                *#
#*                        Jerry Manic Saftware                       *#
#*  Indiana University  University of Oregon  University of Rennes   *#
#*********************************************************************#

set pr_psfile ""

#
# printCanvas: generic window to print a canvas contents as a PostScript file
#              or a text widget contents as a text file
#
#         win: pathname of window
#        name: default name for output file (without suffix)
#        type: text or canvas?
#

proc printCanvas {win name {type "canvas"}} {
  global pr_psfile

  if [winfo exists .print] {
  } else {
    toplevel .print
    wm title .print "Print"

    if { $type == "canvas" } {
      set pr_psfile "${name}.ps"
      set co [lindex [$win configure -scrollregion] 4]
    } else {
      set pr_psfile "${name}.txt"
      set co {0 0 0 0}
    }

    frame .print.top
    pack .print.top -side top -padx 15 -pady 15

    label .print.l1 -text "Filename:"
    entry .print.e1 -textvariable pr_psfile -relief sunken
    pack .print.l1 .print.e1 -side top -in .print.top -anchor w

    frame .print.bottom -relief sunken -bd 1
    pack .print.bottom -side left -padx 15 -pady 10

    button .print.b1 -text "print" -command "
        if { [info exists pr_psfile] } {
          if { \"$type\" == \"canvas\" } {
            $win postscript -colormode color -file \$pr_psfile \
                            -x 0 -y 0 -height [lindex $co 3] \
                            -width [lindex $co 2] 
          } else {
            set out \[open $pr_psfile w\]
            puts \$out \[$win get 1.0 end\]
            close \$out
          }
        }
        destroy .print
      "
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
}
