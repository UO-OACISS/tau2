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
        line = "    "    
        for(i = 1; i <= NF; i++){
            source = $i
            if(match($i,"\"//")){
                # opari2 splits the ctc-string into multiple lines, for
                # comparison to reference data we put this into one line
                while(match(source, "\"//")){
                    # this is nasty, by getting a new line, we have to reset
                    # the record counter i to 1 so we don't miss the closing
                    # bracket at the end of the last line
                    getline
                    i = 1
                    sub("     \"", "", $1)
                    source = source $1
                    sub("\"//&\"", "", source)
                }
                # remove full path
                gsub("escl=([^/]*/)*","escl=",source)
                gsub("sscl=([^/]*/)*","sscl=",source)
                # remove old length
                sub("\"[0-9a-z]*","\"", source)
                # insert new length
                sub("\"", "\""length(source)-2, source)
                line = line " " source
            }
            else{
                line = line " " $i
            }
        }

        # now for fortran77 we split the line again
        while(length(line) > 66){
            subline = substr(line, 1, 66) 
            print subline"\"//"
            line = "     &\"" substr(line, 67)
        }
        print line
#	splitindex = match(line,"Type=")+5
#	print "splitindex=" splitindex "for " line
#	lineRepl = substr(line, 0, splintindex-1) "\"//\n     &\"" substr(line, splitindex)
#	line = lineRepl
#	print lineRepl
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
#    else if(match($0,"get_max_threads")){
        #remove timestamp based function specifier
#        gsub("pomp_get_max_threads[0-9a-z_]*", "pomp_get_max_threads000", $0)
#        print $0
#    }
    else if(match($0,"/cb")){
        #remove timestamp based common block identifier
        gsub("cb[0-9a-z_]*", "cb000", $0)
        print $0
    }
    else if(match($0,"CHARACTER*")){
        gsub("CHARACTER\\*[0-9a-z]*", "CHARACTER*999", $0)
        print $0
    }
    else{
        print $0
    }
}
