# This file is part of the Score-P software (http://www.score-p.org)
#
# Copyright (c) 2009-2011,
#    *    RWTH Aachen University, Germany
#    *    Gesellschaft fuer numerische Simulation mbH Braunschweig, Germany
#    *    Technische Universitaet Dresden, Germany
#    *    University of Oregon, Eugene, USA
#    *    Forschungszentrum Juelich GmbH, Germany
#    *    German Research School for Simulation Sciences GmbH, Juelich/Aachen, Germany
#    *    Technische Universitaet Muenchen, Germany
#
# See the COPYING file in the package base directory for details.

{
    if(match($0,"sscl=")){
        line = "   "
        for(i = 1; i <= NF; i++){
            if(match($i,"^\"")){
                #remove full path
                gsub("escl=([^/]*/)*","escl=",$i)
                gsub("sscl=([^/]*/)*","sscl=",$i)
                #remove old length
                sub("\"[0-9a-z]*","\"", $i)
                #insert new length
                sub("\"", "\""length($i)-1, $i)
            }
            line = line " " $i
        }
        print line
    }
    else if(match($0,"Init_reg")){
        #remove the timestamp based region identifier
        gsub("Init_reg_[0-9a-z_]+","Init_reg_000",$0)
        print $0
    }
    else if(match($0,"#line")){
        #remove the path from the line numbering
        gsub("/([^/]*/)*","",$0)
        print $0
    }
    else if(match($0,"#include") && match($0,"opari.inc")){
        #remove the path from the line numbering
        gsub("/([^/]*/)*","",$0)
        print $0
    }
    else{
        print $0
    }
}
