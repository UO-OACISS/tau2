#*********************************************************************#
#*                TAU/pC++/Sage++  Copyright (C) 1994                *#
#*                        Jerry Manic Saftware                       *#
#*  Indiana University  University of Oregon  University of Rennes   *#
#*********************************************************************#

source "$TAUDIR/inc/fileio.tcl"
source "$TAUDIR/inc/projman.tcl"
source "$TAUDIR/inc/bdbm.tcl"


# This could be more friendly
proc atExit {} {
    if {[lindex [PM__Status] 0] != "NO_PROJECT"} {
	Bdb__SaveDB
    }
    exit
}




wm withdraw .

# Initialize the project manager
PM__Initialize



