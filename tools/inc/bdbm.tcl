###########################################################################
# TAU Browser Databse Manager (BDBM)
#
# This is a TAU tool that runs invisibly in the TAU environment to server
# certain requests for browser information.  This code must be sourced 
# from a TAU tool skeleton (generated from tool.skel).
#
# Kurt Windisch (kurtw@cs.uoregon.edu) - 4/25/96
###########################################################################


###########################################################################
# Data Structures
#
#
# BDB_projname -> string containing name of project
#
# BDB_host -> sting containing the host computer name or "localhost"
#
# BDB_names(<progfile>,<name>) -> list of {<tag> <filen> <line> <mangledname>}
#                                 for files defining the function <name>.
#
# BDB_classes(<progfile>,<class>) -> list of {<tag> <filen> <line> for files
#                                    defining the class <class>.
#
# BDB_progfile_headers(<progfile>) -> list of header files included by
#                                     <progfile>
#
# BDB_progfiles_comptime(<progfile>) -> currently either "compiled" or
#                                       "uncompiled" for remote usage
#
# BDB_maintag -> {<depfilename> <tag>}
#
###########################################################################

set DEBUG_SET 0
proc DEBUG {} {
  global DEBUG_SET ; return $DEBUG_SET
}

###########################################################################
# Accessor Procedures
#


proc Bdb__GetFuncs {{filen ""}} {
    global BDB_names

    set retlist [list]
    if {$filen == ""} {
	foreach entry [array names BDB_names] {
	    set tmplist [split $entry ","]
	    set name [lindex $tmplist 1]
	    set progfile [lindex $tmplist 0]

	    foreach c $BDB_names($entry) {
		lappend retlist [list $name $progfile [lindex $c 0]]
	    }
	}
    } else {
	foreach entry [array names BDB_names "$filen,*"] {
	    set tmplist [split $entry ","]
	    set name [lindex $tmplist 1]
	    set progfile [lindex $tmplist 0]

	    foreach c $BDB_names($entry) {
		lappend retlist [list $name $progfile [lindex $c 0]]
	    }
	}
    }

    return $retlist
}


proc Bdb__GetClasses {{filen ""}} {
    global BDB_classes

    set retlist [list]
    if {$filen == ""} {
	foreach entry [array names BDB_classes] {
	    set tmplist [split $entry ","]
	    set name [lindex $tmplist 1]
	    set progfile [lindex $tmplist 0]

	    foreach c $BDB_classes($entry) {
		set deffile [lindex $c 1]
		set defline [lindex $c 3]
		if {![info exists \
			__tmp__unique_classes($name,$deffile,$defline)]} {
		    set __tmp__unique_classes($name,$deffile,$defline) 1
		    lappend retlist [list $name $progfile [lindex $c 0]]
		}
	    }
	}
    } else {
	foreach entry [array names BDB_classes "$filen,*"] {
	    set tmplist [split $entry ","]
	    set name [lindex $tmplist 1]
	    set progfile [lindex $tmplist 0]

	    foreach c $BDB_classes($entry) {
		lappend retlist [list $name $progfile [lindex $c 0]]
	    }
	}
    }

    return $retlist
}


proc Bdb__GetHeaders {{filen ""}} {
    global BDB_progfile_headers

    set sets [list]
    if [info exists BDB_progfile_headers] {
	foreach n [array names BDB_progfile_headers] {
	    set sets [concat $sets $BDB_progfile_headers($n)]
	}
    }
    set headers [eval lunion $sets]
    return $headers
}


# returns a list of {<file> <tag> <line> <def-file> <name>} for <file>s that 
# contain definitions for function <func>.
proc Bdb__LookupFuncTag {progfile tag} {
    global BDB_names

    set retlist [list]
    foreach entry [array names BDB_names "$progfile,*"] {
	set func [string range $entry \
		[expr [string first "," $entry] + 1] end]
	foreach elem $BDB_names($entry) {

	    if {[lindex $elem 0] == $tag} {
		set deffile [lindex $elem 1]
		set defline [lindex $elem 2]
		return [list \
			$progfile [lindex $elem 0] $defline $deffile $func]
	    }
	}
    }
    return BDBM_FAILED
}

# returns a list of {<file> <tag> <line> <def-file> <name>} for <file>s that 
# contain definitions for class.
proc Bdb__LookupClassTag {progfile tag} {
    global BDB_classes

    set retlist [list]
    foreach entry [array names BDB_classes "$progfile,*"] {
	set class [string range $entry \
		[expr [string first "," $entry] + 1] end]
	foreach elem $BDB_classes($entry) {

	    if {[lindex $elem 0] == $tag} {
		set deffile [lindex $elem 1]
		set defline [lindex $elem 2]
		return [list \
			$progfile [lindex $elem 0] $defline $deffile $class]
	    }
	}
    }
    return BDBM_FAILED
}

# returns a list of {<file> <tag> <line> <def-file> <name>} for <file>s that 
# contain definitions for function <func>.
proc Bdb__LookupFunc {func} {
    global BDB_names

    set retlist [list]
    foreach entry [array names BDB_names "*,$func"] {
	set filen [string range $entry 0 [expr [string first "," $entry] - 1]]
	foreach elem $BDB_names($entry) {
	    set deffile [lindex $elem 1]
	    set defline [lindex $elem 2]
	    if {![info exists __tmp__instances($deffile,$defline)]} {
		lappend retlist \
			[list $filen [lindex $elem 0] $defline $deffile $func]
		set __tmp__instances($deffile,$defline) 1
	    }
	}
    }
    return $retlist
}


# returns a list of {<file> <tag> <line> <def-file> <name>} for <file>s that 
# contain definitions for class <class>.
proc Bdb__LookupClass {class} {
    global BDB_classes

    set retlist [list]
    foreach entry [array names BDB_classes "*,$class"] {
	set filen [string range $entry 0 [expr [string first "," $entry] - 1]]
	foreach elem $BDB_classes($entry) {
	    set deffile [lindex $elem 1]
	    set defline [lindex $elem 2]
	    if {![info exists __tmp__instances($deffile,$defline)]} {
		lappend retlist \
			[list $filen [lindex $elem 0] $defline $deffile $class]
		set __tmp__instances($deffile,$defline) 1
	    }
	}
    }
    return $retlist
}

# Merges LookupFunc and LookupClass, returning an additional <type>.
proc Bdb__LookupName {name} {
    global BDB_names BDB_classes

    set retlist [list]

    # Look for funcs
    foreach entry [array names BDB_names "*,$func"] {
	set filen [string range $entry 0 [expr [string first "," $entry] - 1]]
	foreach elem $BDB_names($entry) {
	    set deffile [lindex $elem 1]
	    set defline [lindex $elem 2]
	    if {![info exists __tmp__instances($deffile,$defline)]} {
		lappend retlist \
			[list $filen [lindex $elem 0] $defline $deffine \
			$name FUNC]
		set __tmp__instances($deffile,$defline) 1
	    }
	}
    }

    # Look for classes
    foreach entry [array names BDB_classes "*,$class"] {
	set filen [string range $entry 0 [expr [string first "," $entry] - 1]]
	foreach elem $BDB_classes($entry) {
	    set deffile [lindex $elem 1]
	    set defline [lindex $elem 2]
	    if {![info exists __tmp__instances($deffile,$defline)]} {
		lappend retlist \
			[list $filen [lindex $elem 0] $defline $deffile CLASS]
		set __tmp__instances($deffile,$defline) 1
	    }
	}
    }

    return $retlist
}



# NOT YET USED - NOT IN INTERFACE
# returns a list of function names defined in <file> if <file> is specified,
# or for the entire application if <file> is unspecified.
proc Bdb__GetNames {{filen "-"}} {

    global BDB_names

    set retlist [list]
    if {$filen == "-"} {
	foreach entry [array names BDB_names] {
	    lappend retlist [string range $entry\
		    [expr 1 + [string last "," $entry]] end]
	}
    } else {
	foreach entry [array names BDB_names "$filen,*"] {
	    lappend retlist [string range $entry \
		    [expr 1 + [string last "," $entry]] end]
	}
    }

    return $retlist
}

# returns {<file> <tag>} specifying the depfile and tag containing the
# applications Main function, or 0 if Main is not defined in the BDB
proc Bdb__GetMaintag {} {
    global BDB_maintag

    if {[info exists BDB_maintag]} {
	return $BDB_maintag
    } else {
	return 0
    }
}

# returns the unique {<progfile> <tag>} for the definition of a given function
# returns an empty list if not definition is found
proc Bdb__FindFuncDefByName {name sig} {
    global BDB_names
    set candidates [Bdb__LookupFunc $name]
    foreach c $candidates {
	set cprogfile [lindex $c 0]
	if {![info exists __tmp__searched($cprogfile)]} {
	    set __tmp_searched($cprogfile) 1
	    foreach m $BDB_names($cprogfile,$name) {   ;# foreach match
	        set mtag [lindex $m 0]
	        if {[llength $m] >= 4} {  ;# mangled name info available
	            if { [string match [lindex $m 3] \
		                       $sig]} {
                        return [list $cprogfile $mtag]
	            }
		}
	    }
	}
    }
    
    return [list]
} 

    
# 
# End of Accessor Procedures
###########################################################################



###########################################################################
# Database Manipulation Procedures
#    

# Adds an uncompiled program file to the database - Must be called before
# calling BDB__Compile.
proc Bdb__Add {file} {
    global BDB_names BDB_progfile_headers \
	    BDB_progfiles_comptime BDB_classes
    
    if {[info exists BDB_progfiles_comptime($file)]} {
	puts "Error: Adding a file twice is illegal."
	return
    }

    set BDB_progfile_headers($file) [list]
    set BDB_progfiles_comptime($file) "uncompiled"
}
    

# Removes all traces of a program file from the database if <what> is
# "removing" or unspecified) or removes just names, classes, progfile_headers
# if <what> is "compiling".
proc Bdb__Remove {filen {what removing}} {
    global BDB_names BDB_progfile_headers \
	    BDB_progfiles_comptime BDB_classes BDB_maintag

    foreach entry [array names BDB_names "$filen,*"] {
	unset BDB_names($entry)
    }
    foreach entry [array names BDB_classes "$filen,*"] {
	unset BDB_classes($entry)
    }
    if {[info exists BDB_progfile_headers($filen)]} {
	unset BDB_progfile_headers($filen)
    }

    if {[info exists BDB_maintag] && ([lindex $BDB_maintag 0] == $filen)} {
	unset BDB_maintag
    }

    if {$what != "compiling"} {
	if {[info exists BDB_progfiles_comptime($filen)]} {
	    unset BDB_progfiles_comptime($filen)
	}
    }
}


# Compiles database info for the given program file, first removing old info
# if it is already in the DB.
proc Bdb__Compile {filen} {
    global BDB_names BDB_progfile_headers \
	    BDB_progfiles_comptime BDB_classes BDB_maintag \
	    depfile REMSH BINDIR BDB_host BDB_projname
    
    if {[info exists BDB_progfiles_comptime($filen)]} {
	if {$BDB_progfiles_comptime($filen) != "uncompiled"} {
	    Bdb__Remove $filen compiling
	}
    } else {
	puts "Browser Database Manager Error: attemped to compile file"
	puts "  ($file) that has not been added to the database."
	return
    }

    set lang [PM__SetLangOption $filen]
    set cgm [Lang_GetCGM $lang]
    set cgmfile [Bdb__cgmfile $filen]
    if {[string index $cgmfile 0] != "/"} {
	set dirn [file dirname $BDB_projname]
	set cgmfile "$dirn/$cgmfile"
    }

    if {![FileIO_file_exists $BDB_host $cgmfile]} {
	return
    }

    if {$BDB_host == "localhost"} {
	set oldcd [pwd]
	cd $depfile(dir)
	set in [open "| $BINDIR/$cgm -dumpbdb $cgmfile" r]
    } else {
	set REMBINDIR "$depfile(root)/bin/$depfile(arch)"
	set in [open "|$REMSH $BDB_host \
		-n \"$REMBINDIR/$cgm -dumpbdb $cgmfile\"" r]
    }
    while {![eof $in]} {
	gets $in ln

	# treat it as a list so that quotes are handled right
	if {[llength $ln] > 0} {
	    set com  [lindex $ln 0]
	    set arg1 [lindex $ln 1]
	    set arg2 [lindex $ln 2]
	    if {[llength $ln] >= 4} {
		set arg3 [lindex $ln 3]
	    } else {
		set arg3 "-"
	    }
	    
	switch $com {
	    ftag:   { ;# -- tag: <tag> <name> <signature>
	    set tag $arg1
	    set name $arg2
	    set sig $arg3
	    }

	    ffile:  { ;# -- file: <line|0> <file|-> @ <position>
	    if {$arg2 != "-"} {
		set BDB_names($filen,$name) \
			[lappend BDB_names($filen,$name) \
			[list $tag $arg2 $arg1 $sig]]

		if {$filen != $arg2} {
		    if {![info exists __tmp__headers($arg2)]} {
			set BDB_progfile_headers($filen) \
				[lappend BDB_progfile_headers($filen) $arg2]
			set __tmp__headers($arg2) 1
		    }
		}
	    }
	    }
	    
	    ftype: { # -- type: <par|seq> <type> <used|not>
	    if {$arg2 == "Main"} {
		set BDB_maintag [list $filen $tag]
	    }
	    }

	    ctag:   { ;# -- tag: <tag> <name> <COLL|->
	    set tag $arg1
	    set name $arg2
	    }

	    cfile:  { # -- file: <line|0> <file|-> @ <position>
	    if {$arg2 != "-"} {
		set BDB_classes($filen,$name) \
			[lappend BDB_classes($filen,$name) \
			[list $tag $arg2 $arg1]]

		if {$filen != $arg2} {
		    if {![info exists __tmp__headers($arg2)]} {
			set BDB_progfile_headers($filen) \
				[lappend BDB_progfile_headers($filen) $arg2]
			set __tmp__headers($arg2) 1
		    }
		}
	    }
	    }
	} ;# switch
    set com ""; set ln ""
    } ;# if items
    } ;# while
    if [catch {close $in} errmsg] {
	showError "Error obtaining browing info: `$errmsg'."
	if { ! [regexp -nocase "warning" $errmsg] } {
	    return NOT_OK
	}
    }

    if {$BDB_host == "localhost"} {
	set BDB_progfiles_comptime($filen) [file atime $cgmfile]
    } else {
	set BDB_progfiles_comptime($filen) "compiled"
    }
}


proc Bdb__SaveDB {} {
    global BDB_names BDB_progfile_headers BDB_host \
	    BDB_progfiles_comptime BDB_classes BDB_maintag BDB_projname \
	    depfile FileIO_Error

    if [DEBUG] { puts "SaveDB:"}

    if {![info exists BDB_projname]} { return; }

    set outfile [FileIO_file_open $BDB_host "$BDB_projname.bdb" w]
    if {$outfile == "FILEIO_ERROR"} {
	showError "Error opening browser info file: $FileIO_Error"
	return
    }

    if [DEBUG] { puts "saving progfiles_comptime"}
    foreach entry [array names BDB_progfiles_comptime] {
	puts $outfile [format "set BDB_progfiles_comptime(%s) %s" \
		$entry $BDB_progfiles_comptime($entry)]
    }

    if [DEBUG] { puts "saving progfile_headers"}
    foreach entry [array names BDB_progfile_headers] {
	puts $outfile [format "set BDB_progfile_headers(%s) \\{%s\\}" \
		$entry $BDB_progfile_headers($entry)]
    }

    if [DEBUG] { puts "saving names"}
    foreach entry [array names BDB_names] {
	regsub -all {\{} $BDB_names($entry) "\\\{" tmpout1
	regsub -all {\}} $tmpout1 "\\\}" tmpout2
	puts $outfile [format "set BDB_names(%s) \\{%s\\}" \
		$entry $tmpout2]
    }

    if [DEBUG] { puts "saving classes"}
    foreach entry [array names BDB_classes] {
	regsub -all {\{} $BDB_classes($entry) "\\\{" tmpout1
	regsub -all {\}} $tmpout1 "\\\}" tmpout2
	puts $outfile [format "set BDB_classes(%s) \\{%s\\}" \
		$entry $tmpout2]
    }

    if [DEBUG] { puts "saving maintag"}
    if {[info exists BDB_maintag]} {
	puts $outfile [format "set BDB_maintag \\{%s\\}" \
		$BDB_maintag]
    }

    flush $outfile
    FileIO_file_close $BDB_host $outfile
}


proc Bdb__LoadDB {} {
    global BDB_names BDB_progfile_headers BDB_host BDB_maintag \
	    BDB_progfiles_comptime BDB_classes BDB_projname \
	    depfile FileIO_Error

    if [DEBUG] { puts "LoadDB:"}
    
    set in [FileIO_file_open $BDB_host "$BDB_projname.bdb" r]
    if {$in == "FILEIO_ERROR"} {
	showError "Error opening browser info file: $FileIO_Error"
	return
    }

    while {[gets $in bdbinfo] >= 0} {
	if [DEBUG] {puts "reading: $bdbinfo"}
	catch "eval $bdbinfo" foo
        if [DEBUG] {puts "CATCH : $foo"}
    }

    #set bdbinfo [read $in]
    #if [DEBUG] { puts "reading $bdbinfo"}
    #catch "eval $bdbinfo" foo
    #puts "CATCH : $foo"

    set fresult [FileIO_file_close $BDB_host $in]

    if {$fresult == "FILEIO_ERROR"} {
	showError "Error opening browser info file ($BDB_projname.bdb): \
		$FileIO_Error"
	return
    }
}

#
# End of Database Manipulation Procedures
###########################################################################


###########################################################################
# Utility Procedures
#

proc Bdb__remove_elements {l filen} {
    set result [list]
    foreach elem $l {
	if {[lindex $elem 0] != $filen} {
	    lappend result $elem
	}
    }
    return $result
}   

proc Bdb__cgmfile {progfile} {
    set lang [PM__SetLangOption $progfile]
    set ext [Lang_GetExt $lang]
    return  [format "%s.%s" [file rootname $progfile] $ext]
}

#
# End Utility Procedures
###########################################################################



###########################################################################
#
# Main Program
#
# BDBM is called with a project name and a list of program files as parameters.
#

proc Bdb__Init {projname host projdir file_list} {
    global BDB_projname BDB_host BDB_progfiles_comptime
    global depfile

    set BDB_projname "$projdir/$projname"
    set BDB_host $host
    
    # If a browser database exists, load it and check file dates to see 
    # if current
    if [FileIO_file_exists $BDB_host "$projdir/$projname.bdb"] {
	
	# Load and update the database
	if [DEBUG] { puts "bdb files exists --> Bdb__LoadDB" }
	Bdb__LoadDB
	if [DEBUG] { puts "...done with Bdb__LoadDB" }
	
	# Check if specified files are in BDB and their indices built
	foreach progfile $file_list {
	    if {![info exists BDB_progfiles_comptime($progfile)]} {
		if [DEBUG] { puts "$progfile not in bdb - building index" }
		Bdb__Add $progfile
		Bdb__Compile $progfile
	    } elseif {$BDB_host == "localhost"} {
		# take adavange of local timestamps 
		if {($BDB_progfiles_comptime($progfile) != "uncompiled") \
			&& ($BDB_progfiles_comptime($progfile) < \
			[file mtime [Bdb__cgmfile $progfile]])} {
		    if [DEBUG] { puts "$progfile out of date, building index" }
		    Bdb__Compile $progfile
		}
	    } else {
		# don't have timestamps (since remote) - just compile it
		if [DEBUG] { puts "$progfile remote - build index to be safe" }
		Bdb__Compile $progfile
	    }
	}
    
	foreach progfile [array names BDB_progfiles_comptime] {
	    # check if $progfile is in $file_list.  If not, remove from BDB
	    if {[lsearch -exact $file_list $progfile] == -1} {
		Bdb__Remove $progfile
	    }
	}

    } else {
	# Create a database from scratch
	foreach progfile $file_list {
	    if [DEBUG] { puts "create a new database for $progfile" }
	    Bdb__Add $progfile
	    Bdb__Compile $progfile
	}
    }
} 


proc Bdb_DeInit {} {
    global BDB_names BDB_progfile_headers BDB_host \
	    BDB_progfiles_comptime BDB_classes BDB_maintag BDB_projname

    if [DEBUG] { puts "Bdb_DeInit!"}
    Bdb__SaveDB 
    if {![info exists BDB_projname]} { return }
    unset BDB_projname
    if [info exists BDB_host] {
	unset BDB_host 
    }
    if [info exists BDB_names] {
	unset BDB_names
    }
    if [info exists BDB_classes] {
	unset BDB_classes
    }
    if [info exists BDB_progfile_headers] {
	unset BDB_progfile_headers
    }
    if [info exists BDB_progfiles_comptime] {
	unset BDB_progfiles_comptime
    }
    if [info exists BDB_maintag] {
	unset BDB_maintag
    }
}


proc Bdb__ChangeName {newname} {
    # This procedure changes the name of the current project, without changing
    # any of the current project files. For updating with a new .pmf file, with the
    # old file as a backup.
    
    global BDB_projname
    
    set statusTemp [PM__Status]
    if {[llength $statusTemp] <= 1} {
	showError "Unable to change project name:\nIncomplete status."
	return
    } else {
	set BDB_projname "[lindex $statusTemp 4]/[file rootname $newname]"
    }
}


#
# End of Program
###########################################################################
