###########################################################################
# TAU Browser Databse Manager Utilities
#
# This file will be included by other TAU tools needing to interface with
# the browser database.  It contains functions that (via the BDBM) 
# querry and modify the database.
#
# The functions described below are to be sourced into all TAU tools needing
# the depfile indexing services of the browser database.
# Each function in the BDBM interfacecommunicates with the Tau daemon, 
# which contains the BDBM, requesting information, which is returned via 
# RPC from the interface function.  The TAU daemon MUST be running for these
# functions to operate.
#
# This document refers to three types of source code files: 
#   1. A "program file" (abbreviated "progfile") contains primary source code, 
#      such as function definitions.  Examples of progfiles are 
#      ".c, .C, or .pc" files.  
#   2. A "header file" is a file that may be multiply included into progfiles,
#      usually containing function and type declarations.  Examples of header
#      files are .h files in C, C++, or pC++.  
#   3. A "depfile" is a representation of the abstract syntax tree for a
#      single progile and multiple included header files.  The depfile is
#      created by the compiler parser and read by TAU via a "CGM" program,
#      described in the document, "Interfacing to TAU".
# The functions in the BDBM interface usually take progfile names, rather
# than depfile names, as parameters.
# 
# In this document names surrounded by angle brackets, "<>", are 
# representations of parameters or return values.  The brackets are not 
# used in actual code. Items surrounded by square brackets, "[]", are 
# optional, and the brackets should not appear in actual code.  The "==>" 
# symbol is used to indicate the format of the return value which follows.
#  
# Any BDBM interface functions will return the string "BDBM_FAILED" if the
# operation failed.
#
# Kurt Windisch (kurtw@cs.uoregon.edu) - 5/24/96
###########################################################################


###########################################################################
#
# Bdb_GetFuncs - Return a list containing all functions 
#                that are defined separately in a given progfile, or if
#                the progfile is unspecified, return a list of all functions
#                in the application.
#    Usage:
#      Bdb_GetFuncs [ <progfile> ]
#        ==> { {<func_name> <progfile> <tag>} ... }
#

proc Bdb_GetFuncs {{progfile ""}} {
    if [catch {xsend taud "Bdb__GetFuncs $progfile"} retval] {
	return BDBM_FAILED
    } else {
	return $retval
    }
}


###########################################################################
#
# Bdb_GetClasses - Return a list containing all classes (if C++ based language)
#                  that are defined separately in a given progfile, or if
#                  the progfile is unspecified, return a list of all classes
#                  in the application.
#    Usage:
#      Bdb_GetClasses [ <progfile> ]
#        ==> { {<classname> <progfile> <tag>} ... }
#


proc Bdb_GetClasses {{progfile ""}} {
    if [catch {xsend taud "Bdb__GetClasses $progfile"} retval] {
	return BDBM_FAILED
    } else {
	return $retval
    }
}


###########################################################################
#
# Bdb_GetHeaders - Return a list of the header files included by a given
#                  progfile, or if a progfile is not specified, return a 
#                  list of all header files in the application.
#    Usage:
#      Bdb_GetHeaders [ <progfile> ]
#        ==> { <header_name> ... }
#

proc Bdb_GetHeaders {{progfile ""}} {
    if [catch {xsend taud "Bdb__GetHeaders $progfile"} retval] {
	return BDBM_FAILED
    } else {
	return $retval
    }
}


###########################################################################
#
# Bdb_AddFile - Adds a specified progfile to the browser database (but does 
#               not build the index for it).  Header files are implicitly
#               added at compile-time.  The language is returned.
#    Usage:
#      Bdb_AddFile <progfile>
#        ==> <language>

proc Bdb_AddFile {progfile} {
    if [catch {xsend taud "Bdb__Add $progfile"} retval] {
	return BDBM_FAILED
    } else {
	return $retval
    }
}


###########################################################################
#
# Bdb_RemoveFile - Removes a given progfile and any uniquely included 
#                  header files from the browser database.  Returns nothing.
#    Usage:
#      Bdb_RemoveFile <progfile>

proc Bdb_RemoveFile {progfile} {
    if [catch {xsend taud "Bdb__Remove $progfile"} retval] {
	return BDBM_FAILED
    } else {
	return $retval
    }
}


###########################################################################
#
# Bdb_BuildIndex - Builds or update an index for a given progfile from the 
#                  compiled depfile.  This must be called anytime the depfile 
#                  for a progfile is updated.  Returns nothing.
#    Usage:
#      Bdb_BuildIndex <progfile>

proc Bdb_BuildIndex {progfile} {
    if [catch {xsend taud "Bdb__Compile $progfile"} retval] {
	return BDBM_FAILED
    } else {
	return $retval
    }
}


###########################################################################
#
# Bdb_QueryFuncTag - Lookup a tag in a given progfile and returns
#                    a list of associated progfile, tag and line, and
#                    definition file.
#    Usage:
#      Bdb_QueryFuncTag <progfile> <tag>
#        ==> { <progfile> <tag> <linenum> <def-file> <name>}

proc Bdb_QueryFuncTag {progfile tag} {
    if [catch {xsend taud "Bdb__LookupFuncTag $progfile $tag"} retval] {
	return BDBM_FAILED
    } else {
	return $retval
    }
}


###########################################################################
#
# Bdb_QueryClassTag - Lookup a tag in a given progfile and returns
#                    a list of associated progfile, tag and line, and
#                    definition file.
#    Usage:
#      Bdb_QueryClassTag <progfile> <tag>
#        ==> { <progfile> <tag> <linenum> <def-file> <name>}

proc Bdb_QueryClassTag {progfile tag} {
    if [catch {xsend taud "Bdb__LookupClassTag $progfile $tag"} retval] {
	return BDBM_FAILED
    } else {
	return $retval
    }
}


###########################################################################
#
# Bdb_QueryFunc - Lookup a function name in the browser DB index and return
#                 a list of associated progfile, tag, line, definition file, 
#                 and function name lists, one per distinct instance in the 
#                 index.
#    Usage:
#      Bdb_QueryFunc <funcname> 
#        ==> { { <progfile> <tag> <linenum> <deffile> <funcname> } ... }

proc Bdb_QueryFunc {func} {
    if [catch {xsend taud "Bdb__LookupFunc $func"} retval] {
	return BDBM_FAILED
    } else {
	return $retval
    }
}


###########################################################################
#
# Bdb_QueryClass - Lookup a class name in the browser DB index and return
#                  a list of associated progfile, tag, line, definiton file,
#                  and class name lists, one per distinct instance in the 
#                  index.
#    Usage:
#      Bdb_QueryClass <classname>
#        ==> { { <progfile> <tag> <linenum> <deffile> <classname> } ... }
proc Bdb_QueryClass {class} {
    if [catch {xsend taud "Bdb__LookupClass $class"} retval] {
	return BDBM_FAILED
    } else {
	return $retval
    }
}


###########################################################################
#
# Bdb_QueryName -  Lookup a name in the browser DB index and return
#                  a list of associated progfile, tag, line, and type lists,
#                  one per distinct instance in the index.  The name may
#                  refer to either a FUNC or CLASS and all matches on either
#                  will be returned.  Valid types are "FUNC" and "CLASS".
#    Usage:
#      Bdb_QueryName <name>
#        ==> { { <progfile> <tag> <linenum> <def-file> <name> <type> } ... }
proc Bdb_QueryName {name} {
    if [catch {xsend taud "Bdb__LookupName $class"} retval] {
	return BDBM_FAILED
    } else {
	return $retval
    }
}


###########################################################################
#
# Bdb_GetMaintag - Returns the filename and tag associated with the 
#                  Main function of the applications, or 0 if a main
#                  function is not defined.
#    Usage:
#      Bdb_GetMaintag
#        ==> { <progfile> <tag> }
proc Bdb_GetMaintag {} {
    if [catch {xsend taud "Bdb__GetMaintag"} retval] {
	return BDBM_FAILED
    } else {
	return $retval
    }
}

###########################################################################
#
# Bdb_FindFuncDefByName - Returns the UNIQUE progfile and tag associated 
#                         with the definition of a given function, or 
#                         returns an empty list if no defintion is found.
#
#    Usage:
#      Bdb_FindFuncDefByName <name> <mangled_name>
#        ==> { <progfile> <tag> }
proc Bdb_FindFuncDefByName {name sig} {
    if [catch {xsend taud "Bdb__FindFuncDefByName $name $sig"} retval] {
	return BDBM_FAILED
    } else {
	return $retval
    }
}


###########################################################################
#
# Bdb_ChangeName - Changes the project name without disturbing the rest of the database.
#                               Useful when making a backup of the current project.
#
#    Usage:
#      Bdb_ChangeName <newname>

proc Bdb_ChangeName {newname} {
    if [catch {xsend taud "Bdb__ChangeName $newname"} retval] {
	puts "BDBM_FAILED ine Bdb_ChangeName"
	return BDBM_FAILED
    } else {
	return $retval
    }
}



###########################################################################
#
# Bdb_Init - Initializes the browswer database for a specific project.
#            Bdb_DeInit must be used between sucessive calls to Init.
#    Usage:
#      Bdb_Init <project_name> <host> <project_directory> <progfile_list>
#
# No interface function is provided for Bdb_Init, since it is only called
# from within the daemon.


###########################################################################
#
# Bdb_DeInit - Uninitializes the browswer database for a specific project.
#    Usage:
#      Bdb_Init 
#
# No interface function is provided for Bdb_Init, since it is only called
# from within the daemon.
