#===========================================================
#TAU PROJECT MANAGER (PROJMAN) - Internal Implementation
#===========================================================
#
# The global variable used throughout the ProjMan's operations
# is the PM_Globals variable. This variable is sourced whenever
# there is a need to access the global structure from within the
# tool (everywhere within this file. . .)
# 
# The format of the PM_Globals variable is as follows:
#
# PM_Globals(host) == hostname of the machine where the project resides
# PM_Globals(arch) == the architecture of the host machine
# PM_Globals(hostarch) == hostname + architecture
# PM_Globals(root) == the TAUROOT of the machine where the project resides
# PM_Globals(dir) == the directory prefix of the project's files
# PM_Globals(project) == the name of the working project
# PM_Globals(WorkingTools) == a list of the running tools
# PM_Globals(SelLineList) == list of tools that require GlobalSelectLine
# PM_Globals(SelFuncList) == list of tools that require GlobalSelectFunc
# PM_Globals(SelClassList) == list of tools that require GlobalSelectClass
#
# PM_Globals(FileList) == a list of the files currently in the project.
# PM_Globals(file,lang) == the language of the progfile
# PM_Globals(file,compile_opts) == the compiler options for the progfile
#
# The individual tools should not access these variables directly, but 
# through the accessor/set functions defined within this document.
#
###########################################################################

set PM_Globals(1) ""
source $TAUDIR/inc/selectfile.tcl
source $TAUDIR/inc/fileio.tcl
set depfile(1) ""


# A debugging framework. . .
set DEBUG_SET 0


proc DEBUG {} {
    global DEBUG_SET

    return $DEBUG_SET
}



proc PM__Orphaned {} {
    global PM_Globals 
    global myself myname
    
    set avail [winfo interps]
    set am_orphaned 0
    foreach tool $PM_Globals(WorkingTools) {
	if {[lsearch -exact $avail $tool$myname(ext)] == -1} {
	    PM__RemTool $tool
	}
    }
    if {[llength $PM_Globals(WorkingTools)] < 1 } {
	#PM__DeInit
	atExit
    }

    after 10000 PM__Orphaned
}
	




###########################################################################
#    
#    PM__AddFile -   Takes one or no arguments. If one argument, adds
#		    the progfile to the current project. If no args,
#		    displays a dialog prompting for file(s) to add.
#		    Returns the number of files added.
#
#    Usage:
#	  PM__AddFile [ <progfile> ] ==> <number_of_added_files>
#

proc PM__AddFile { {progfile "" } } {
    
    global PM_Globals
    if [DEBUG] { puts "PM__AddFile $progfile"}    
    if [string match $progfile "" ] {
	if [DEBUG] {	puts "progfile matched empty string."}
	set progfile  [getFile "Add A Source File" * 0 $PM_Globals(host) \
			   "" $PM_Globals(dir)]
	if [DEBUG] {	puts "Returned from the getFile call in AddFile."}
	if [DEBUG] {puts "progfile == $progfile"}
    }

    set numfiles 0
    foreach file $progfile {
	if [DEBUG] { puts "Calling RelativizePath"}
	set file [RelativizePath $PM_Globals(host) $PM_Globals(dir) $file]
	if [DEBUG] {	puts "PM__AddFile: file=$file"}
	PM__InsertFile $file
	lappend broadcast_for $file
	incr numfiles
    }
    if [info exists broadcast_for] {
	PM__BroadcastChanges $broadcast_for a
    }
    PM__WritePMF
    return $numfiles
}


###########################################################################
#    
#    PM_RemoveFile -   
#		    Takes one or no arguments. If one argument, removes
#		    the progfile to the current project. If no args,
#		    displays a dialog prompting for file(s) to remove.
#		    Returns the number of files removed.
#
#    Usage:
#	  PM_AddFile [ <progfile> ] ==> numfiles
#

proc PM__RemoveFile {{progfile ""} } {
    
    global PM_Globals

    if [DEBUG] {puts "PM__RemoveFile"}

    if [string match $progfile "" ] {
	return 0
    }
    
    set numfiles 0

    foreach file $progfile {
	if {[lsearch $PM_Globals(FileList) $progfile] != -1} {
	    set cnt 0
	    foreach index $PM_Globals(FileList) {
		if [string match $PM_Globals($progfile,lang) \
			$PM_Globals($index,lang)] {
		    incr cnt
		}
	    }
	    if {$cnt <= 1} {
		set index [lsearch $PM_Globals(langs) $PM_Globals($progfile,lang)]
		set PM_Globals(langs) [lreplace $PM_Globals(langs) $index $index] 
	    }
	    PM__DeleteFile $file
	    lappend broadcast_for $file
	    incr numfiles
	}
    }
    if [info exists broadcast_for] {
	PM__BroadcastChanges $broadcast_for d
    }
    if [DEBUG] { puts "PM__RF: Returned from BroadcastChanges"}
    
    return $numfiles
}

###########################################################################
#
#    PM__GetFiles -  Returns a list containing all the progfile names
#		    included in the current project. This listing is 
#		    of progfiles only, excluding header files that may
#		    be associated with the project.
#
#    Usage:
#	  PM__GetFiles ==> { <progfile> ... }
#

proc PM__GetFiles {} {
    
    global PM_Globals
    if [DEBUG] {puts "PM__GetFiles"}
    
    return $PM_Globals(FileList)
}

###########################################################################
#
#    PM_GetHeaders - Returns a list containing the names of all the 
#		     header files included in a progfile, or in the
#                    entire project if no progfile is specified. 
#		     Each header file is listed only once, although
#		     any given header file might be included multiple
#		     times within the project.
#
#    Usage:
#	  PM_GetHeaders ==> { <header file> ... }
#

proc PM__GetHeaders {{progfile "" }} {

    if [DEBUG] {puts "PM__GetHeaders"}
    set temp [Bdb__GetHeaders $progfile]

    return $temp
}


###########################################################################
#
#    PM__GetOptions - Takes a progfile as an argument, and returns 
#		     a list of options defined on that progfile.
#		     
#    Usage:
#	  PM__GetOptions <progfile>
#		==> { <lang> <compile_opt> }
#

proc PM__GetOptions { {progfile "" } } {

    global PM_Globals
    if [DEBUG] {puts "PM__GetOptions"}
    
    if {![string compare $progfile "" ]} {
	return ""
    }
    foreach item $PM_Globals(FileList) {
	lappend templist [file tail $item]
    }
    if {[set index [lsearch $templist $progfile]] != -1} {
	set progfile [lindex $PM_Globals(FileList) $index]
    }
    foreach item $PM_Globals(FileList) {
	if [string match $item $progfile] {
	    set temp [list $PM_Globals($item,lang) \
		    $PM_Globals($item,compile_opts)]
	    return $temp
	}
    }
    return PM_FAILED
}

###########################################################################
#
#    PM__GetProjectLangs -
#                       Takes nothing, and returns a list of all the languages used in the 
#                       current project.
#
#    Usage:
#             PM__GetProjectLangs ==> <list-o-langs>

proc PM__GetProjectLangs {} {
   
    global PM_Globals
    
    return $PM_Globals(langs)
}

###########################################################################
#
#    PM__SetLangOption -
#		    Takes a progfile as an argument, and optionally a
#		    language (as defined in languages.tcl). The
#		    function returns the value of the progfile's
#		    language, after those options have been set by the
#		    additional argument's values.
#
#    Usage:
#	  PM__SetLangOption <progfile> [ <lang> ] ==> <lang> 
#
#
# This needs some work. What if the language gets set to "Fred"?
# That would be bad.
#

proc PM__SetLangOption {{progfile "" } {lang "" } } {
    
    global PM_Globals
    if [DEBUG] {puts "PM__SetLangOption"}

    if [string match $progfile "" ] {
	return PM_FAILED
    }
    foreach item $PM_Globals(FileList) {
	lappend templist [file tail $item]
    }
    if {[set index [lsearch $templist $progfile]] != -1} {
	set progfile [lindex $PM_Globals(FileList) $index]
    }
    if {[lsearch $PM_Globals(FileList) $progfile] != -1} {
	if {![string compare $lang "" ]} {
	    return $PM_Globals($progfile,lang)
	} else {
	    set PM_Globals($progfile,lang) $lang
	    return $lang
	}
    }
}


###########################################################################
#
#    PM__SetCompileOption -
#		    Takes a progfile as an argument, and optionally
#		    a set of compile options. Returns the value of the
#		    progfile's compile options, after updating the
#		    options with the additional arguments.
#
#    Usage:
#	  PM__SetCompileOptions <progfile> [ <compile_opts> ]
#			 ==> <compile_opts>
#
#Note: This procedure ought only be called by Cosy, which is the 
#only place (I know of) that has a handle on all the options.
#

proc PM__SetCompileOption {{progfile "" } {compile_opts "" }} {
    
    global PM_Globals
    if [DEBUG] {puts "PM__SetCompileOption"}

    if {![string compare $progfile ""]} {
	return PM_FAILED
    }
    foreach item $PM_Globals(FileList) {
	lappend templist [file tail $item]
    }
    if {[set index [lsearch $templist $progfile]] != -1} {
	set progfile [lindex $PM_Globals(FileList) $index]
    }
    if {[lsearch $PM_Globals(FileList) $progfile] != -1} {
	if {![string compare $compile_opts ""]} {
	    return $PM_Globals($progfile,compile_opts)
	} else {
	    set PM_Globals($progfile,compile_opts) $compile_opts
	    PM__WritePMF
	    return $compile_opts
	}
    } else {
	return PM_FAILED
    }
}

###########################################################################
#
#    PM__SetProjectOption -
#             Takes in either a string of project specific loader/compiler 
#             options or nothing.
#             Returns the value of PM_Globals(Project_Opts), after setting 
#             those options (if applicable)
#
#    Usage:
#        PM__SetProjectOption [<string-o-opts>] ==> <current project options>

proc PM__SetProjectOption {{Proj_Opts ""}} {

    global PM_Globals
    
    if [DEBUG] { puts "PM__SetProjectOption '$Proj_Opts'"}
    if [string match $Proj_Opts ""] {
	return $PM_Globals(Project_Opts)
    } else {
	set PM_Globals(Project_Opts) $Proj_Opts
	PM__WritePMF
	return $PM_Globals(Project_Opts)
    }
}


###########################################################################
#
#    PM__RecompileNotify -
#		    Takes a list of progfiles as an argument, and returns
#		    nothing. (Calls Bdb_BuildIndex to update the
#		    BDB information, and updates any running tools
#		    with the salient, new information.)
#
#    Usage:
#	  PM__RecompileNotify {<progfile> ...}
#

proc PM__RecompileNotify {{progfiles {} } } {
    global PM_Globals

    if [DEBUG] {puts "PM__RecompileNotify $progfiles"}
    if {[llength $progfiles] == 0} {
	if [DEBUG] {	puts "PM_Rec - updating all"}
	PM__BroadcastChanges $PM_Globals(FileList) u
    } else {
	foreach pf $progfiles {
	    if [DEBUG] {puts "PM_Rec - Updating $pf"}
	    if {[lsearch $PM_Globals(FileList) $pf] != -1} {
		if [DEBUG] { puts "  PM_Rec - Calling Broadcasting"}
		lappend broadcast_for $pf
	    } else {
		if [DEBUG] { puts "PM_Rec - ERROR"}
		return PM_FAILED
	    }
	}
	if [info exists broadcast_for] {
	    PM__BroadcastChanges $broadcast_for u
	}
    }
}


###########################################################################
#
#    PM__BroadcastChanges -
#		    This function takes a list of progfiles and an
#                   action flag (d,a,u,p, or e), and returns 
#		    nothing. (Calls upon the separate tools to
#		    update their own information. This function
#		    will most likely be called by Cosy, although
#		    there may be other tools that modify the 
#		    project in such a way as to require changes.)
#                   The default flag is "u" for update.
#
#    Usage:
#	  PM__BroadcastChanges {<progfile> ...} <flag>
#
#
proc PM__BroadcastChanges { progfiles {flag "u" } } {

    global PM_Globals
    if [DEBUG] {puts "PM__BroadcastChanges"}
    
    switch $flag {
	a { 
	    foreach pf $progfiles { Bdb__Add $pf; } 
	}
	d { 
	    foreach pf $progfiles { Bdb__Remove $pf; }
	}
	u {
	    foreach pf $progfiles { Bdb__Compile $pf; }
	}
	p {
	    # Modify project information.
	    # May be needed in future.
	}
	e {
	    # This flag is for updating a tool after/during execution of the 
	    # project binary. No action needed here. . .
	}
	default {
	    return PM_FAILED
	}
    }

    foreach i $PM_Globals(WorkingTools) {
	if [DEBUG] {	puts "Broadcasting to $i"}
	set cmd "Tool_AcceptChanges {$progfiles} $flag"
	if [catch {async_send $i "$cmd" } retval] {
	    puts $retval
	    showError "$i did not accept the change."
	}
    }
    if [DEBUG] { puts "PM_Broadcast done"}
}


###########################################################################
#
#   PM__AddGlobalSelect -
#                    This procedure takes a toolname and a list of 
#                    required global select functions (from among
#                    selectFuncTag
#                    showFuncTag
#                    selectClassTag
#                    selectLine
#                    ), and adds the tool to
#                    the list of tools that require those functions.
#                    Returns nothing.
#
#   Usage:
#        PM__AddGlobalSelect <toolname> {<list of select functions>}
#

proc PM__AddGlobalSelect {name {funclist "" } } {
    
    global PM_Globals

    if {[lsearch $PM_Globals(WorkingTools) $name] == -1} {
	return PM_FAILED
    }
    foreach i $funclist {
	lappend PM_Globals($i) $name
    }
    
}


###########################################################################
#
#   PM__RemGlobalSelect -
#                    This procedure takes a toolname and a list of 
#                    global select functions (from among
#                    selectFuncTag
#                    showFuncTag
#                    selectClassTag
#                    selectLine
#                    ), and removes the tool from the se of tools 
#                    that require those functions.
#                    Returns nothing.
#
#   Usage:
#        PM__RemGlobalSelect <toolname> {<list of select functions>}
#

proc PM__RemGlobalSelect {name {funclist "" } } {
    
    global PM_Globals

    if {[lsearch $PM_Globals(WorkingTools) $name] == -1} {
	return PM_FAILED
    }
    foreach i $funclist {
	set PM_Globals($i) [lremove -exact $PM_Globals($i) $name]
    }
    
}


###########################################################################
#
#    PM__GlobalSelect -
#		    This procedure sets up the global select
#		    operation, sending a wide set of tools similar
#		    commands based on the type of structure to 
#		    examine and the availability of the tools.
#		    The procedure takes as arguments the name of
#		    the file being examined, the type of selection
#		    to propogate, the tag of the selected item, and
#		    an optional list of any other related information that the
#		    selection needs. Returns nothing.
#
#   NB: the list in "other" needs to be formatted 
#       {<progfile> <line>}
#       for the procedure to work properly.
#
#    Usage:
#	  PM__GlobalSelect <file> <function> <tag> [<list of other>]

proc PM__GlobalSelect {file function tag args} {

    global PM_Globals
    if [DEBUG] {puts "PM__GlobalSelect $file $function $tag '$args'"}

    # Strip off the "global_"
    set local_func [string range $function 7 end]
    
    foreach tool $PM_Globals($function) {
	if [DEBUG] {	puts "  $tool: $local_func $file $tag"}
	if [llength $args] {
	    # currently, only one additional 
	    if [DEBUG] { puts "SENDING w/ args"}
	    async_send $tool "$local_func $file $tag [lindex $args 0] [lindex $args 1]"
	    if [DEBUG] { puts "done"}
	} else {
	    if [DEBUG] { puts "SENDING $local_func $file $tag"}
	    async_send $tool "$local_func $file $tag"
	    if [DEBUG] {puts "done"}
	}
    }
}


###########################################################################
#
#   PM__AddTool -  This procedure adds a tool to the list of running
#                 tools. The name added is the name used by xsend for
#                 making inter-tool calls (i.e. it ought to be discrete).
#                 Returns nothing.
#
#   Usage:
#        PM__AddTool <toolname>
#

proc PM__AddTool {name} {
    
    global PM_Globals
    if [DEBUG] {puts "PM__AddTool"}
    lappend PM_Globals(WorkingTools) $name
}


###########################################################################
#
#   PM__RemTool -  This procedure adds a tool to the list of running
#                 tools. The name added is the name used by xsend for
#                 making inter-tool calls (i.e. it ought to be discrete).
#                 Returns nothing.
#
#   Usage:
#        PM__RemTool <toolname>
#

proc PM__RemTool {name} {
    
    global PM_Globals

    if [DEBUG] {puts "PM__RemTool"}
    set index [lsearch $PM_Globals(WorkingTools) $name]
    if {$index != -1} {
	set PM_Globals(WorkingTools)\
		[lreplace $PM_Globals(WorkingTools) $index $index]
	if {[llength $PM_Globals(WorkingTools)] < 1 } {
	    #PM__DeInit
	    atExit
	}
    } else {
	return PM_FAILED
    }
}

###########################################################################
#
#    PM__GetHost -   This procedure queries the PM for the host machine
#		    of the project. Returns a string that is the 
#		    host name. Takes no arguments.
#
#    Usage:
#	  PM__GetHost ==> <hoststring>
#

proc PM__GetHost {} {
    
    global PM_Globals
    if [DEBUG] {puts "PM__GetHost"}
    return $PM_Globals(host)
}

###########################################################################
#
#    PM__GetRoot -   This procedure queries the PM for the root of the
#		    Tau distribution. Equates to TAUROOT. Takes no
#		    arguments, and returns a string that is the
#		    project root.
#
#    Usage:
#	  PM__GetRoot ==> <rootstring>
#

proc PM__GetRoot {} {
    
    global PM_Globals
    if [DEBUG] {puts "PM__GetRoot"}
    return $PM_Globals(root)
}

###########################################################################
#
#    PM__SetRoot -   This procedure sets the PM for the root of the
#		    Tau distribution. Equates to TAUROOT. Takes the
#		    new project root string as an argument and returns
#		    nothing.
#
#    Usage:
#	  PM__SetRoot <rootstring>
#

proc PM__SetRoot {rootstring} {
    
    global PM_Globals depfile
    if [DEBUG] {puts "PM__GetRoot"}
    set PM_Globals(root) $rootstring
    set depfile(root) $PM_Globals(root)
}

###########################################################################
#
#    PM__GetHostarch -
#		    This procedure queries the PM for the host and 
#		    architecture of the project, in the format
#		    required by the tools (== host.dom.name (arch))
#		    Returns a string in that format.
#
#    Usage:
#	  PM__GetHostarch ==> <hostarchstring>
#

proc PM__GetHostarch {} {
    
    global PM_Globals
    if [DEBUG] {puts "PM__GetHostarch"}
    return $PM_Globals(hostarch)
}

###########################################################################
#
#    PM__GetArch -   This procedure Takes no arguments. Returns
#		    a string that describes the architecture of
#		    machine that tau is currently running on.
#
#    Usage:
#	  PM__GetArch ==> <archstring>
#

proc PM__GetArch {} {

    global PM_Globals
    if [DEBUG] {puts "PM__GetArch"}
    return $PM_Globals(arch)
}

###########################################################################
#
#    PM__GetDir -    This procedure takes no arguments. Returns
#		    a string that gives the path (absolute or
#		    relative) of the current project's directory.
#
#    Usage:
#	  PM__GetDir ==> <dirstring>
#

proc PM__GetDir {} {
    
    global PM_Globals
    if [DEBUG] {puts "PM__GetDir"}
    return $PM_Globals(dir)
}


###########################################################################
#
#    PM__OpenProject -
#		    This procedure takes an optional argument, that
#		    is a valid path to a valid .pmf file for use by
#		    the PM or the path to a (supposedly) .pmf-bearing
#		    directory , or no argument at all.
#		    If the file checks out with the PM, the 
#		    PM and taud will initialize with that file, and 
#		    allow the invoking tool to continue, using that
#		    project as the basis.
#		    If the argument is a directory, pass that
#		    directory to the PM, and select/create a project
#		    there.
#		    If no argument is given, the
#		    PM will prompt for a project file to open, and 
#		    either open that project or die. 
#		    The procedure returns the name of the project
#		    opened, or an error.
#
#    Usage:
#	  PM__OpenProject [[<projfile>|<projdir>] [<host>]] 
#                                    ==> <project_name> | ERROR
#

proc PM__OpenProject {{loc "" } {host "" } } {
    
    global PM_Globals

    if [DEBUG] { puts "PM__OpenProject"  }

    set PM_Globals(cod,path) ""

    if [string match $host "" ] {
	# Assume local host if empty string for host portion
	set host "localhost"
    }

    if [string match $loc "" ] {
	# If location is empty, assume current directory
	# Also, if loc is empty, assume creation of new project.
	set temp [FileIO_pwd $host]
	set retval [PM__FindOrCreatePMF $host $temp]
	if {![string match $retval ""]} {
	    # Just a wrapper to delay evaluation. . .
	    PM__ReadPMF $retval $PM_Globals(host)
	}
    } else {
	if [DEBUG] {puts "PM_OP: we got passed a path. Into the else clause."}
	# A path has been passed. Verify that it leads to either a
	# valid .pmf file, or to a directory in which to create one.
	if [string match *:* $loc] {
	    # The location/host is in <host>:<dir> form. Divvy it up.
	    # If there are colons in the directories on the path, this will 
	    # handle them very badly. . .
	    set temp [split $loc :]
	    set host [lindex $temp 0]
	    set loc [lindex $temp 1]
	}
	if [string match *.pmf [file tail $loc]] {
	    if [FileIO_file_exists $host $loc] {
		if [DEBUG] {	puts "PM__OpenProject: opening existing project $loc"}
		# so the path references a .pmf file, which exists
		set PM_Globals(host) $host
		PM__ReadPMF $loc $host
	    }
	} elseif [FileIO_dir_exists $host $loc] {
	    # else the path is to a valid drectory, hopefully bearing readable project files. . .
	    if [DEBUG] {puts "PM__OpenProject: finding .pmf file in $loc"}
	    PM__FindOrCreatePMF $host $loc
	    if {![string match $PM_Globals(host) $host]} {
		set host $PM_Globals(host)
	    }
	} else {
	    # The path references nothing! Run away!
	    puts stderr "usage: <tool> \[\[<host>\] \[<project file>|<project directory>\]\]"
	    atExit
	}
    }
    
    if [DEBUG] {puts "PM__OpenProject - Initializing BDBM !!!!"}
    
    # Start the browser database for the project.
    set projdir $PM_Globals(dir)
    set projname [file rootname $PM_Globals(project)]
    Bdb_DeInit
    Bdb__Init $projname $host $projdir [PM__GetFiles]
    
    return $PM_Globals(project)
}


###########################################################################
#
#    PM__Status -    This procedure returns a list of elements that 
#		    are the basic working information of the 
#		    current project. If there is no project running,
#		    an error message is returned. Takes no arguments.
#
#    Usage:
#	  PM__Status ==>
#	       {<projname> <host> <arch> <root> <dir>} | "UNDEFINED"
#

proc PM__Status {} {
    
    global PM_Globals
    if [DEBUG] { puts "PM__Status"}
    if {![string match $PM_Globals(project) UNDEFINED]} {
    return [list $PM_Globals(project) $PM_Globals(host) $PM_Globals(arch)\
	    $PM_Globals(root) $PM_Globals(dir)]
    } else {
	return $PM_Globals(project)
    }
}

###########################################################################
#
#    PM__ChangeProjectName -
#                      This procedure takes in a string (which must end in ".pmf"), and changes 
#                      the currently running project to a project which is identical to the current
#                      project except for its name. This allows for changes in a project without 
#                      adjusting the existing snapshot of the project. The procedure updates the
#                      information in all pertinent databases, and returns nothing.
#
#     Usage:
#          PM__ChangeProjectName <newname>
#

proc PM__ChangeProjectName {{newname ""}} {

    global PM_Globals depfile
    
    set PM_Globals(newname) ""
    
    proc PM__ChProjNameGUI {} {
	
	global PM_Globals

	toplevel .chprojname
	wm title .chprojname "Copy Project (Backup)"
	set temp [frame .chprojname.f1]
	label $temp.l -text "New Name:"
	entry $temp.ent \
	    -relief sunken \
	    -textvariable PM_Globals(newname) \
	    -width 25
	pack $temp.l $temp.ent \
	    -side left \
	    -anchor nw \
	    -expand 1 \
	    -fill x
	set temp [frame .chprojname.f2]
	button $temp.enter \
	    -text "Enter" \
	    -command {destroy .chprojname}
	button $temp.cancel \
	    -text Cancel \
	    -command {set PM_Globals(newname) ""; destroy .chprojname}
	pack $temp.enter \
	    -side left \
	    -anchor nw 
	pack $temp.cancel \
	    -side right \
	    -anchor ne
	pack .chprojname.f1 .chprojname.f2 \
	    -anchor nw \
	    -expand 1 \
	    -fill both
    } ; #End PM__ChProjNameGUI

    if {[string match $newname ""]} {
	PM__ChProjNameGUI
	tkwait window .chprojname
	if {![string match *.pmf $PM_Globals(newname)]} {
	    if [string match $PM_Globals(newname) ""] {
		showError "Cancelling; Reverting to Original Project"
		return
	    } else {
		showError "New Project Name must be in <name>.pmf form."
		return
	    }
	}
    } else {
	set PM_Globals(newname) $newname
    }
    
    PM__WritePMF
    set PM_Globals(project) $PM_Globals(newname)
    set depfile(project) $PM_Globals(project)
    PM__WritePMF
    if {[string match $newname ""]} {
	Bdb__Init $PM_Globals(project) \
	    $PM_Globals(host) $PM_Globals(dir) [PM__GetFiles]
    }
    return
}

##########################################################################
#
# Internal Functions: Used only by the PM itself.
#
##########################################################################


proc PM__InsertFile {progfile} {
    # This procedure inserts a file alphabetically into the 
    # global file list. Returns nothing.

    global PM_Globals
    if [DEBUG] {puts "PM__InsertFile '$progfile'"}
    if {[llength $PM_Globals(FileList)] == 0} {
	set PM_Globals(FileList) [list $progfile]
    } else {
	lappend PM_Globals(FileList) $progfile
	set PM_Globals(FileList) [lsort $PM_Globals(FileList)]
    }
    set temp [Lang_GuessLang $progfile]
    set PM_Globals($progfile,lang) $temp
    if {[lsearch [PM__GetProjectLangs] $temp] == -1} {
	lappend PM_Globals(langs) $temp
    }
    set PM_Globals($progfile,compile_opts) ""
    
    #This call should pass the newly added file to all
    #currently running tools.
    #PM__BroadcastChanges $progfile a
}

##########################################################################

##########################################################################
proc PM__DeleteFile {progfile} {
    # This procedure takes a file name, and checks it against
    # the global file list. If the file is in the global file list,
    # it gets the axe. If not, returns immediatley.

    global PM_Globals
    if [DEBUG] {puts "PM__DeleteFile: progfile == $progfile"}

    if {[set index [lsearch $PM_Globals(FileList) $progfile]] != -1 } {
	set PM_Globals(FileList) [lreplace $PM_Globals(FileList) $index $index]
	unset PM_Globals($progfile,lang)
	unset PM_Globals($progfile,compile_opts)

	if [DEBUG] {puts "Returned from PM__BroadcastChanges"}
	PM__WritePMF
	if [DEBUG] {	puts "Returned from PM__WritePMF"}
	if [DEBUG] {	puts "in PM__DF: $progfile was deleted."}
	return
    }
    puts "In PM__DeleteFile: file wasn't found. No action taken."
    return
}
##########################################################################

##########################################################################
#
#    PM__ReadPMF
#
#    This procedure takes the path to a .pmf file (a project file), and 
#    parses the format of that file into a set of information that
#    pertains to the selected project. If the format of the file differs
#    from what the procedure expects, an error will be generated and 
#    the user will be offered the chance to recover or bow out of the
#    operation.
#    The format of the .pmf file is detailed in the file interface.pmfile
#    included in the doc directory of this distribution.
#
#    The host parameter is needed, because the host of the machine that
#    we're going to be getting this information form may be anywhere
#    on the net. Reading the host line from the file will eliminate
#    any ambiguity. SAS 6/12

proc PM__ReadPMF {path {host "" } } {
    
    global PM_Globals depfile
    global TAUROOT

    if [DEBUG] {puts "PM__ReadPMF"}
    # If host parameter empty, assume local operation.
    if {![string compare $host ""]} {
	set host "localhost"
    }

    #    if ![file readable $path] {
    #	showError "$path is not readable."
    #	return FILE_NOT_READ
    #    }

    set pmfile [FileIO_file_open $host $path]
    if [string match $pmfile "FILEIO_ERROR"] {
	showError "Error in PM__ReadPMF:  can't open *.pmf file for reading."
	return FILE_NOT_READ
    }
    if {[llength PM_Globals(FileList)] != 0} {
	set PM_Globals(FileList) [list]
    }
    gets $pmfile projname

    # The first line of a .pmf file is the project name
    set PM_Globals(project) $projname
    set depfile(project) $projname

    # The second line is always the value of TAUROOT for the project.
    gets $pmfile PM_Globals(root)

    # The third line is the list of languages used by the project.
    gets $pmfile temp
    set temp [lindex [split $temp :] 1]
    set PM_Globals(langs) $temp

    # The fourth line is the project options used by the build script for this project.
    gets $pmfile temp
    set temp [lindex [split $temp :] 1]
    set PM_Globals(Project_Opts) $temp

    while {[gets $pmfile temp] != -1} {
	# Allowing for comments after the initial project name entry
	# and for blank lines between entries
	if {![string compare [string index $temp 0] #]} {
	    lappend PM_Globals(project,comm) $temp
	} else {
	    if [string compare $temp ""] {
		
		#From here to the end of the if statement, the file 
		#must be formatted in blocks of:
		#
		#    <any number of blank lines>
		#    File:<filename>
		#    Lang:<language of the file>
		#    Comp_Opts:<compile options>

		
		set temp [lindex [split $temp :] 1]
		lappend PM_Globals(FileList) $temp
		gets $pmfile holder
		set PM_Globals($temp,lang) \
			[lindex [split $holder :] 1]
		gets $pmfile holder
		set PM_Globals($temp,compile_opts) \
			[lindex [split $holder :] 1]
	    }
	}   
    }
    set errorval [FileIO_file_close $host $pmfile]
    if [string match $errorval "FILEIO_ERROR"] {
	showError "Error in PM__ReadPMF:  can't close *.pmf file."
	return FILE_NOT_READ
    }
    if {[file dirname $path] == "."} {
	set PM_Globals(dir) [pwd]
    } else {
	set PM_Globals(dir) [file dirname $path]
    }
    set depfile(root) $PM_Globals(root)
    FileIO_INIT $PM_Globals(root)
    set depfile(dir) $PM_Globals(dir)
}
##########################################################################

##########################################################################
#PM__WritePMF
#
# Since any and all the information in a project can change dynamically
# during the course of a tau session, all the stored information in tau
# should be written out to the project file upon termination of the 
# program or upon switching projects. This procedure creates/overwrites
# the .pmf file associated with the project with the new information that
# may or may not have changed. The format conforms to that expected by
# PM__ReadPMF.

proc PM__WritePMF {} {
    
    global PM_Globals
    global TAUROOT
    if [DEBUG] { puts "PM__WritePMF"}

    if {$PM_Globals(project) == "NO_PROJECT"} { 
	showError "Error in PM__WritePMF:  tau daemon thinks there is no project."
	return FILE_NOT_WRITTEN
    }

    set path "$PM_Globals(dir)/$PM_Globals(project)"
    
    # a host arg isn't needed, since that information is stored in the 
    # PM_Globals array.
    set pmfile [FileIO_file_open $PM_Globals(host) \
		    $PM_Globals(dir)/$PM_Globals(project) w]
    if [string match $pmfile "FILEIO_ERROR"] {
	showError "Error in PM__WritePMF:  can't open *.pmf file for writing."
	return FILE_NOT_WRITTEN
    }

    # Write out the project name
    puts $pmfile "$PM_Globals(project)"

    # Write out the project's conception of TAUROOT
    puts $pmfile "$PM_Globals(root)"

    # Write out the list of languages used by the project
    puts $pmfile "Proj_Langs: $PM_Globals(langs)"

    # Write out the project options for the project
    puts $pmfile "Project_Options: $PM_Globals(Project_Opts)"

    # Put back any comments that the project file may have contained.
    if {![string compare [array names PM_Globals(project,comm)] ""]} {
	foreach line $PM_Globals(project,comm) {
	    puts $pmfile $line
	}
    }

    # Now, put out all the file information in the proper format.
    foreach file $PM_Globals(FileList) {
	puts $pmfile ""
	puts $pmfile "File:$file"
	puts $pmfile "Lang:$PM_Globals($file,lang)"
	puts $pmfile "Comp_Opts:$PM_Globals($file,compile_opts)"
    }
    set errorval [FileIO_file_close $PM_Globals(host) $pmfile]
    if [string match $errorval "FILEIO_ERROR"] {
	showError "Error in PM__WritePMF:  can't close *.pmf file."
	return FILE_NOT_WRITTEN
    }

}
##########################################################################


##########################################################################
#    PM__Initialize
#
#    This procedure initializes the PM, setting up the global access
#    variables and associated values. Called whenever tau is invoked.
#    *NB:* The tau daemon is the root of all activities within tau, and
#    this procedure is the root of all PM activities within the taud.

proc PM__Initialize {} {
    
    global PM_Globals TAUROOT depfile

    if [DEBUG] {puts "PM__Initialize"}
    set PM_Globals(WorkingTools) [list]
    set PM_Globals(project) "NO_PROJECT"
    set PM_Globals(host) "localhost"
    # The fileselect box has been modified to use depfile(host) as its host argument.
    # This will help generic-ize the tool. . .
    set depfile(host) "localhost"
    set PM_Globals(root) $TAUROOT
    set depfile(root) $PM_Globals(root)
    set PM_Globals(dir) ""
    set PM_Globals(langs) [list]
    # This method of finding arch may be apochryphal
    if [DEBUG] { puts "PM_G(root) == $PM_Globals(root)"}
    if [DEBUG] {puts "archfind at $PM_Globals(root)/utils/archfind"}
    if [catch { exec $PM_Globals(root)/utils/archfind } retval] {
	#if [DEBUG] {puts "archfind not found"}
	showError "Error in PM__Initialize: $retval."
	#atExit
	exit
    } else {
	if [DEBUG] {puts "In init: arch == $retval"}
	set PM_Globals(arch) $retval
    }
    set depfile(arch) PM_Globals(arch)
    set PM_Globals(hostarch) "$PM_Globals(host) ($PM_Globals(arch))"
    set depfile(hostarch) $PM_Globals(hostarch)
    # Set comments to nothing, to avoid problem with initial project.
    set PM_Globals(project,comm) ""
    set PM_Globals(FileList) [list]

    FileIO_INIT $TAUROOT

    after 10000 PM__Orphaned
}
##########################################################################


##########################################################################
#
#    PM__DeInit
#
#     This procedure takes no args. It removes all traces of all values stored in PM_Globals,
#     and then re-inits the values with the built-in defaults. Returns nothing.

proc PM__DeInit {} {
    
    global PM_Globals

    foreach item [array names PM_Globals] {
	unset PM_Globals($item)
    }
    set PM_Globals(1) ""
    PM__Initialize
}


##########################################################################
#PM__FindOrCreatePMF
#    This procedure spawns a toplevel window, with a file selection
#    box filtered to only find .pmf files for reading. If no such
#    project file exists in the local directory, the user may create
#    a new project in the current directory. That project comes with
#    the bare minimum of information (a bare bones main.c or main.cc
#    file, for example) to build a useful application from scratch.
#    If one or more project files exists, they will be displayed in
#    the selector box, and one file may be chosen. This file will be
#    the basis for the selection.
#    Returns the path to a project file.
#
#    NB: The path ought to be the path to a directory, not a file.

proc PM__FindOrCreatePMF {{host "" } {path "" }} {

    if [DEBUG] { puts "PM__FindOrCreatePMF"}

    global PM_Globals depfile
    
    set pmflist [list]
    
    if [string match $host "" ] {
	set host "localhost"
    }
    if [string match $path "" ] {
	# If the path is empty, then find or create starting with
	# the current dir. If the host is remote, start based on
	# the root directory of the remote machine. I guess. . .
	set path [FileIO_pwd $host]
	if [string match $path "FILEIO_ERROR"] {
	    if [DEBUG] { puts "PM__FOCPMF: pwd failed for $host"}
	    return PM_FAILED
	}
    }
    #if {[string match $host [PM__GetHost]] && \
    #	    [string match $path [PM__GetDir]]} {
    #	return $path ; #we're referencing the same path. . .
    #    }
    if { [llength [PM__GetFiles] ] > 0 } {
	set $PM_Globals(project) ""
	foreach file $PM_Globals(FileList) {
	    if [DEBUG] {puts "referencing $file"}
	    foreach item [array names PM_Globals ${file},*] {
		unset PM_Globals($item)
		if [DEBUG] {puts "Unset PM_G($item)"}
	    }
	}
	set PM_Globals(FileList) [list]
    }
    set lsholder [FileIO_ls $host $path]
    if [string match $lsholder "FILEIO_ERROR"] {
	if [DEBUG] {puts "PM__FOCPMF: ls failed on $host, in $path"}
	return PM_FAILED
    }
    foreach item $lsholder {
	if [string match *.pmf $item] {
	    lappend pmflist $item
	}
    }
    if [llength $pmflist] {
	set path [getFile "Pick A Project File" *.pmf 1 $host]
	if [DEBUG] {puts "PM__FOCPMF: path == $path "}
    } else {
	set path [getFile "Find or Create A Project" * 1 $host]
	if [DEBUG] {puts "PM__FOCPMF: path == $path"}
    }
    if [DEBUG] { puts "PM__FOCPMF: files are [PM__GetFiles]"}
    if {![llength [PM__GetFiles]]} {
	if [DEBUG] {puts "PM__FOCPMF: about to read with no files."}
	PM__ReadPMF $path $PM_Globals(host)
    }
    set PM_Globals(hostarch) "$PM_Globals(host) ($PM_Globals(arch))"
    set PM_Globals(dir) [file dirname $path]
    set PM_Globals(done) $path
    if [catch { FileIO_exec $PM_Globals(host) $PM_Globals(root)/utils/archfind } retval] {
	if [DEBUG] {puts "archfind not found"}
	atExit
    } else {
	if [DEBUG] {puts "In init: arch == $retval"}
	set PM_Globals(arch) $retval
    }
    set depfile(host) $PM_Globals(host)
    set depfile(hostarch) $PM_Globals(hostarch)
    set depfile(dir) $PM_Globals(dir)
    set depfile(arch) $PM_Globals(arch)
    set depfile(root) $PM_Globals(root)
    return $path
}

##########################################################################


##########################################################################
#PM__CreateNewProject
#
#   This procedure takes a path to the proj dir, a name for the project, 
#   and a valid language as it's parameter, and 
#   formats a new project (with one file, a skeleton main file) in
#   the given language. If the directory that was specified is not
#   valid or writeable, then the creation process generates an error. 
#   Returns the name of the new project.
#
#   NB: When a new language is added to languages.tcl, the new language
#       needs to be added to the switch in this procedure.

proc PM__CreateNewProject {path lang {host "" } } {
    
    global PM_Globals depfile
    if [DEBUG] {puts "PM__CreateNewProject"}
    if [DEBUG] {puts "host == $host"}
    
    if [FileIO_file_exists $host $path] {
	displayMessage "PM__CNP: $path already exists on $host."
	return PM_FAILED
    }

    set newmain [file dirname $path]/[file rootname [file tail $path]]
    set templatef [Lang_GetTemplateName $lang]
    set in [FileIO_file_open $host \
	    $PM_Globals(root)/$templatef r]

    set newfilen [format "%s.%s" $newmain [Lang_GetProgExt $lang]]
    if {![FileIO_file_exists $host $newfilen]} {
    if [DEBUG] {
	puts "New program file name: $newfilen"
	global FileIO_Error ; puts "FileIO_Error: $FileIO_Error"
	puts "Input file handle: $in"
    }
    set out [FileIO_file_open $host $newfilen w]
    if [string match $out "FILEIO_ERROR"] {
	if [DEBUG] {puts "PM__CNP: couldn't open $newfilen for writing."}
	return PM_FAILED
    }
    set temparch [FileIO_exec $host "$PM_Globals(root)/utils/archfind"]
    if [string match $temparch "FILEIO_ERROR"] {
	if [DEBUG] {puts "PM__CNP: archfind not accessed correctly."}
	return PM_FAILED
    } else {
	set PM_Globals(arch) $temparch
	set depfile(arch) $temparch
    }
    while {[gets $in line] >= 0} {
	puts $out $line
    }
    close $in
    close $out
    }   ;# if newfile exists

    set PM_Globals(host) $host
    set depfile(host) $host
    set PM_Globals(project) [file tail $path]
    set depfile(project) [file tail $path]
    set PM_Globals(project,comm) ""
    set PM_Globals(langs) [list $lang]
    set PM_Globals(dir) [file dirname $path]
    set depfile(dir) [file dirname $path]
    set PM_Globals(hostarch) "$PM_Globals(host) /($PM_Globals(arch)/)"
    set depfile(hostarch) $PM_Globals(hostarch)
    set PM_Globals(FileList) [list [file tail $newfilen]]
    set PM_Globals([file tail $newfilen],lang) $lang
    set PM_Globals([file tail $newfilen],compile_opts) ""
    set PM_Globals(Project_Opts) ""
    PM__WritePMF 
    return PM_Globals(project)
}
