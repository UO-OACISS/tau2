#*********************************************************************#
#*                TAU/pC++/Sage++  Copyright (C) 1995                *#
#*                        Jerry Manic Saftware                       *#
#*  Indiana University  University of Oregon  University of Rennes   *#
#*********************************************************************#

toplevel .$myself
wm title .$myself "HANDY"
text .$myself.t -width 19 -height 3 -font "*-Courier-Bold-R-Normal-*-240-*" \
     -background white -foreground black
pack .$myself.t
.$myself.t insert end "\n Have A Nice Day !"
.$myself.t tag add red 2.1
.$myself.t tag add red 2.6
.$myself.t tag add red 2.8
.$myself.t tag add red 2.13
.$myself.t tag add red 2.15
.$myself.t tag configure red -foreground red
