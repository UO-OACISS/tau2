#===========================================================
#TAU PROJECT MANAGER (PROJMAN) - External Interface Implementation
#===========================================================
#
# The functions described below are sourced at the startup of 
# all tools that need to access the Project Manager's list of current 
# files, and the language and compile options associated with each of 
# those files.
#
# The Project Manager (referred to hereafter as the PM) has access
# to a limited subset of the BDBM information. Each of these interface
# functions communicates with the PM through the TAU daemon to request
# information, which is then returned via RPC from the interface
# function. The TAU daemon MUST be running for these functions to
# operate.
#
# This document refers tp three types of source code files:
#   1. A "program file" (abbreviated "progfile") contains
#	primary source code, such as function definitions. Examples
#	of progfiles are ".c, .C, or .pc" files.
#
#   2. A "header file" is a file that may be multiply defined into 
#	progfiles, usually containing function and type declarations.
#	Examples of header files are .h files in C, C++, or pC++.
#
#   3. A "depfile" is a representation of the abstract syntax tree for a
#	single progfile and multiple included header files. The depfile
#	is created by the compiler parser and read by TAU via a "CGM"
#	program, to be described elsewhere.
#
# The functions in the PM interface usually take progfile names, rather
# than depfile names, as parameters.
#
# In this document names surrounded by angle brackets, "<>", are representations
# of parameters or return values.  The brackets do not be used in actual code.
# Items surrounded by square brackets, "[]", are optional, and the brackets
# should not appear in actual code.  The "==>" symbol is used to indicate
# the format of the return value which follows.
#
# Any PM interface functions will return the string "PM_FAILED" if the
# operation failed.
###########################################################################


  
###########################################################################
#    
#    PM_AddFile -   Takes one or no arguments. If one argument, adds
#		    the progfile to the current project. If no args,
#		    displays a dialog prompting for file(s) to add.
#		    Returns the number of added files.
#
#    Usage:
#	  PM_AddFile [ <progfile> ]
#

proc PM_AddFile {{progfile ""}} {
    if [catch {async_send taud "PM__AddFile $progfile"}] {
	return PM_FAILED
    } else {
	return 1
    }
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
#	  PM_AddFile [ <progfile> ] 
#

proc PM_RemoveFile {{progfile ""}} {
    if [catch {async_send taud "PM__RemoveFile $progfile"}] {
	return PM_FAILED
    } else {
	return 1
    }
}


###########################################################################
#
#    PM_GetFiles -  Returns a list containing all the progfile names
#		    included in the current project. This listing is 
#		    of progfiles only, excluding header files that may
#		    be associated with the project.
#
#    Usage:
#	  PM_GetFiles ==> { <progfile> ... }
#

proc PM_GetFiles {} {
    if [catch {xsend taud "PM__GetFiles"} retval] {
	return PM_FAILED
    } else {
	return $retval
    }
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
#	  PM_GetHeaders [<progfile>] ==> { <header file> ... }
#
proc PM_GetHeaders {{progfile ""}} {
    if [catch {xsend taud "PM__GetHeaders $progfile"} retval] {
	return PM_FAILED
    } else {
	return $retval
    }
}


###########################################################################
#
#    PM_GetOptions - Takes a progfile as an argument, and returns 
#		     a list of options defined on that progfile.
#		     
#    Usage:
#	  PM_GetOptions <progfile>
#		==> { <lang> <compile_opt> <pre_compile_opts> }
#

proc PM_GetOptions {{progfile ""}} {
    if [catch {xsend taud "PM__GetOptions $progfile"} retval] {
	return PM_FAILED
    } else {
	return $retval
    }
}

###########################################################################
#
#    PM__GetProjectLangs -
#                       Takes nothing, and returns a list of all the languages used in the 
#                       current project.
#
#    Usage:
#             PM__GetProjectLangs ==> <list-o-langs>

proc PM_GetProjectLangs {} {
    if [catch {xsend taud "PM__GetProjectLangs"} retval] {
	return PM_FAILED
    } else {
	return $retval
    }
}

###########################################################################
#
#    PM_SetLangOption -
#		    Takes a progfile as an argument, and optionally a
#		    language (as defined in languages.tcl). The
#		    function returns the value of the progfile's
#		    language, after those options have been set by the
#		    additional argument's values.
#
#    Usage:
#	  PM_SetLangOption <progfile> [ <lang> ] ==> <lang> 
#

proc PM_SetLangOption {{progfile ""} {lang ""}} {
    if [catch {xsend taud "PM__SetLangOption $progfile $lang"} retval] {
	return PM_FAILED
    } else {
	return $retval
    }
}


###########################################################################
#
#    PM_SetCompileOption -
#		    Takes a progfile as an argument, and optionally
#		    a set of compile options. Returns the value of the
#		    progfile's compile options, after updating the
#		    options with the additional arguments.
#
#    Usage:
#	  PM_SetCompileOption <progfile> [ <compile_opts> ]
#			 ==> <compile_opts>
#

proc PM_SetCompileOption {{progfile ""} {compile_opts ""}} {
    if [catch {xsend taud \
	    "PM__SetCompileOption $progfile \"$compile_opts\""} retval] {
	return PM_FAILED
    } else {
	return $retval
    }
}


###########################################################################
#
#    PM_SetProjectOption -
#             Takes in either a string of project specific loader/compiler options or nothing.
#             Returns the value of PM_Globals(Project_Opts), after setting those options
#             (if applicable)
#
#    Usage:
#        PM_SetProjectOption [<string-o-opts>] ==> <current project options>

proc PM_SetProjectOption {{Proj_Opts ""}} {
    if [catch {xsend taud "PM__SetProjectOption \"$Proj_Opts\""} retval] {
	return PM_FAILED
    } else {
	return $retval
    } 
}


###########################################################################
#
#    PM_RecompileNotify -
#		    Takes a list of progfiles as an argument, and returns
#		    nothing. (Calls Bdb_BuildIndex to update the
#		    BDB information, and updates any running tools
#		    with the salient, new information.)
#
#    Usage:
#	  PM_RecompileNotify {<progfile> ...}
#

proc PM_RecompileNotify {{progfiles {}}} {
    if [catch {xsend taud "PM__RecompileNotify {$progfiles}"} retval] {
	return PM_FAILED
    } else {
	return $retval
    }
}


###########################################################################
#
#    PM_BroadcastChanges -
#		    This function takes a list of progfiles and an
#                   action flag (d,a,u,p or e), and returns 
#		    nothing. (Calls upon the separate tools to
#		    update their own information. This function
#		    will most likely be called by Cosy, although
#		    there may be opther tools that modify the 
#		    project in such a way as to require changes.)
#
#    Usage:
#	  PM_BroadcastChanges {<progfile> ...} <flag>
#

proc PM_BroadcastChanges {progfiles {flag "u"}} {
    if [catch {async_send taud "PM__BroadcastChanges {$progfiles} $flag"}] {
	return PM_FAILED
    } else {
	return 1
    }
}

###########################################################################
#
#   PM__AddGlobalSelect -
#                    This procedure takes a toolname and a list of 
#                    required global select functions (from among
#                    global_selectFuncTag
#                    global_showFuncTag
#                    global_selectClassTag
#                    global_selectLine
#                    ), and adds the tool to
#                    the list of tools that require those functions.
#                    Returns nothing.
#
#   Usage:
#        PM__AddGlobalSelect <toolname> {<list of select functions>}
#

proc PM_AddGlobalSelect {name {funclist ""}} {
    if [catch {xsend taud "PM__AddGlobalSelect $name {$funclist}"} retval] {
	return PM_FAILED
    } else {
	return $retval
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

proc PM_RemGlobalSelect {name {funclist ""}} {
    if [catch {xsend taud "PM__RemGlobalSelect $name {$funclist}"} retval] {
	return PM_FAILED
    } else {
	return $retval
    }
}


###########################################################################
#
#    PM_GlobalSelect -
#		    This procedure sets up the global select
#		    operation, sending a wide set of tools similar
#		    commands based on the type of structure to 
#		    examine and the availability of the tools.
#		    The procedure takes as arguments the name of
#		    the file being examined, the type of selection
#		    to propogate, the tag of the selected item, and
#		    an optional list of any other related information that the
#		    selection needs. Returns nothing.
#     NB: As of 6/26/96, the only tool that takes optional args is global_selectLine, which
#           requires the progfile and line reference.
#
#    Usage:
#	  PM_GlobalSelect <file> <function> <tag> [<list of other>]

proc PM_GlobalSelect {file function tag {other ""}} {
    if [catch {xsend taud "PM__GlobalSelect\
	    $file $function $tag $other"} retval] {
	return PM_FAILED
    } else {
	return $retval
    }
}

###########################################################################
#
#   PM_AddTool -  This procedure adds a tool to the list of running
#                 tools. The name added is the name used by xsend for
#                 making inter-tool calls (i.e. it ought to be discrete).
#                 Returns nothing.
#
#   Usage:
#        PM_AddTool <toolname>
#

proc PM_AddTool {name} {
    if [catch {xsend taud "PM__AddTool $name"} retval] {
	return PM_FAILED
    } else {
	return $retval
    }
}


###########################################################################
#
#   PM_RemTool -  This procedure removes a tool from the list of running
#                 tools. Returns nothing.
#
#   Usage:
#        PM_RemTool <toolname>
#

proc PM_RemTool {name} {
    if [catch {xsend taud "PM__RemTool $name"} retval] {
	return PM_FAILED
    } else {
	return $retval
    }
}


###########################################################################
#
#    PM_GetHost -   This procedure queries the PM for the host machine
#		    of the project. Returns a string that is the 
#		    host name. Takes no arguments.
#
#    Usage:
#	  PM_GetHost ==> <hoststring>
#

proc PM_GetHost {} {
    if [catch {xsend taud "PM__GetHost"} retval] {
	return PM_FAILED
    } else {
	return $retval
    }
}

###########################################################################
#
#    PM_GetRoot -   This procedure queries the PM for the root of the
#		    Tau distribution. Equates to TAUROOT. Takes no
#		    arguments, and returns a string that is the
#		    project root.
#
#    Usage:
#	  PM_GetRoot ==> <rootstring>
#

proc PM_GetRoot {} {
    if [catch {xsend taud "PM__GetRoot"} retval] {
	return PM_FAILED
    } else {
	return $retval
    }
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

proc PM_SetRoot {rootstring} {
    if [catch {xsend taud "PM__SetRoot $rootstring"} retval] {
	return PM_FAILED
    } else {
	return $retval
    }
}

###########################################################################
#
#    PM_GetHostarch -
#		    This procedure queries the PM for the host and 
#		    architecture of the project, in the format
#		    required by the tools (== host.dom.name (arch))
#		    Returns a string in that format.
#
#    Usage:
#	  PM_GetHostarch ==> <hostarchstring>
#

proc PM_GetHostarch {} {
    if [catch {xsend taud "PM__GetHostarch"} retval] {
	return PM_FAILED
    } else {
	return $retval
    }
}

###########################################################################
#
#    PM_GetArch -   This procedure Takes no arguments. Returns
#		    a string that describes the architecture of
#		    machine that tau is currently running on.
#
#    Usage:
#	  PM_GetArch ==> <archstring>
#

proc PM_GetArch {} {
    if [catch {xsend taud "PM__GetArch"} retval] {
	return PM_FAILED
    } else {
	return $retval
    }
}

###########################################################################
#
#    PM_GetDir -    This procedure takes no arguments. Returns
#		    a string that gives the path (absolute or
#		    relative) of the current project's directory.
#
#    Usage:
#	  PM_GetDir ==> <dirstring>
#

proc PM_GetDir {} {
    if [catch {xsend taud "PM__GetDir"} retval] {
	return PM_FAILED
    } else {
	return $retval
    }
}

###########################################################################
#
#    PM_OpenProject -
#		    This procedure takes an optional argument, that
#		    is a valid path to a valid .pmf file for use by
#		    the PM or the path to a (supposedly) .pmf-bearing
#		    directory , or no argument at all.
#		    If the file checks out with the PM, the 
#		    PM and taud will initialize with that file, and 
#		    allow the invoking tool to continue, using that
#		    project as the basis.
#		    If the arguement is a directory, pass that
#		    directory to the PM, and select/create a project
#		    there.
#		    If no argument is given, the
#		    PM will prompt for a project file to open, and 
#		    either open that project or die. 
#		    The procedure returns the name of the project
#		    opened, or an error.
#
#    Usage:
#	  PM_OpenProject [[<projfile>|<projdir>] [<host>]] 
#                                       ==> <project_name> | ERROR
#

proc PM_OpenProject {{loc ""} {host ""}} {
    if [catch {xsend taud "PM__OpenProject $loc"} retval] {
	return PM_FAILED
    } else {
	return $retval
    }
    
}

###########################################################################
#
#    PM_Status -    This procedure returns a list of elements that 
#		    are the basic working information of the 
#		    current project. If there is no project running,
#		    an error message is returned. Takes no arguments.
#
#    Usage:
#	  PM_Status ==>
#	       {<projname> <host> <arch> <root> <dir>} | NO_PROJECT
#

proc PM_Status {} {
    if [catch {xsend taud "PM__Status"} retval] {
	return PM_FAILED
    } else {
	return $retval
    }
}

###########################################################################
#
#    PM_ChangeProjectName -
#                      This procedure takes in a string (which must end in ".pmf"), and changes 
#                      the currently running project to a project which is identical to the current
#                      project except for its name. This allows for changes in a project without 
#                      adjusting the existing snapshot of the project. The procedure updates the
#                      information in all pertinent databases, and returns nothing.
#
#     Usage:
#          PM_ChangeProjectName <newname>
#

proc PM_ChangeProjectName {{newname ""}} {
    if [catch {xsend taud "PM__ChangeProjectName $newname"} retval] {
	return PM_FAILED
    } else {
	return $retval
    }
}