#*********************************************************************#
#*                TAU/pC++/Sage++  Copyright (C) 1994                *#
#*                        Jerry Manic Saftware                       *#
#*  Indiana University  University of Oregon  University of Rennes   *#
#*********************************************************************#


###########################################################################
# CGM MANAGER  - External Interface
# =============================================
# 
# The functions described below are to be sourced into all TAU tools needing
# the multiple depfile management services of the CGM manager.  
# 
# This document refers to three types of source code files: 
#   1. A "program file" (abbreviated "progfile") contains primary source code, 
#      such as function definitions. Examples of progfiles are ".c, .C, or .pc"
#      files.  
#   2. A "header file" is a file that may be multiply included into progfiles,
#      usually containing function and type declarations.  Examples of header
#      files are .h files in C, C++, or pC++.  
#   3. A "depfile" is a representation of the abstract syntax tree for a
#      single progile and multiple included header files.  The depfile is
#      created by the compiler parser and read by TAU via a "CGM" program,
#      described in the document, "Interfacing to TAU".
# The functions in the CGM interface usually take progfile names, rather
# than depfile names, as parameters.
# 
# In this document names surrounded by angle brackets, "<>", are reps
# of parameters or return values.  The brackets are not used in actual code.
# Items surrounded by square brackets, "[]", are optional, and the brackets
# should not appear in actual code.  The "==>" symbol is used to indicate
# the format of the return value which follows.
#   
# Any CGM interface functions will return the string "CGM_FAILED" if the
# operation failed.
###########################################################################


###########################################################################
# Data Structures
#
#   depfiles_loaded -> list of program files for which depfiles are loaded.
#
#   funcs
#       funcs(<file>,<tag>,state)  -> Info for call-graph layout
#       funcs(<file>,<tag>,fold)   -> Info for call-graph layout
#       funcs(<file>,<tag>,name)   -> <name-str>
#       funcs(<file>,<tag>,mname)  -> <mangled-name-str>
#       funcs(<file>,<tag>,class)  -> {<class-name> <class-type> <class-tag>}
#       funcs(<file>,<tag>,file)   -> {<line-num> <filename>}
#       funcs(<file>,<tag>,pos)    -> <pos-data>                     
#       funcs(<file>,<tag>,type)   -> {<par-or-seq> <type> <used|not>
#       funcs(<file>,<tag>,num)    -> <num-calls>
#       funcs(<file>,<tag>,calls)  -> <list-of-calls>
#       funcs(<file>,<tag>,childpos) -> <list-of-callsites>
#       funcs(<file>,<tag>,childline) -> <list-of-callsite-line-numbers>
#
#
#   classes
#       classes(<file>,<tag>,file) -> {<line-num> <filename>
#       classes(<file>,<tag>,name) -> <name-str>
#       classes(<file>,<tag>,coll) -> <COLL>
#       classes(<file>,<tag>,pos)  -> <pos-data>
#       classes(<file>,<tag>,typepos) -> <type-positions>
#       classes(<file>,<tag>,subs) -> <subclasses>       
#       classes(<file>,<tag>,fmempos) -> <member-functions>
#       classes(<file>,<tag>,basepos) -> <base-classes-and-postions> 
#       classes(<file>,<tag>,mem)  -> <member-func-name-str>
#
#   members
#       members(<file>,<tag>) -> {<member-name> <tag> <type>}
#
#
#   depfile(<file>,path)      - pathname of depfile
#   depfile(<file>,file)      - filename part of pathname
#   depfile(<file>,allftags)  - the ids of all functions
#   depfile(<file>,allctags)  - the ids of all classes
#   depfile(<file>,numfunc)   - number of functions defined in depfile
#   depfile(<file>,numused)   - number of functions used (called) in program
#   depfile(<file>,numprof)   - number of functions profiled
#   depfile(<file>,ino)       - inode of depfile
#   depfile(<file>,type)      - type/format of depfile
#
###########################################################################


###########################################################################
# Inialization
#
# Initialize tool specific depfile options
#
proc initDep {} {
    global depfile depfiles_loaded
    set depfiles_loaded [list]
}
###########################################################################


###########################################################################
#
# Cgm_FuncList - Gets a list of functions defined in the given file 
#                (progfile or header allowed).
#
#    Usage:
#      Cgm_FuncList <file> 
#         ==> { <funcname> ... }
proc Cgm_FuncList {fname} {
    global funcs

    set funclist [list]
    foreach entry [array names funcs "*,file"] {
	if {[lindex $funcs($entry) 1] == $fname} {
	    regexp {(.+),([0-9]+),file$} $entry \
		    fullmatch funcprogfile functag
	    if {[lindex $funcs($funcprogfile,$functag,class) 2] == -2} {
		lappend funclist [list \
			$funcs($funcprogfile,$functag,name) \
			$funcprogfile \
			$functag]
	    }
	}
    }
    return $funclist
}


###########################################################################
#
# Cgm_MethodList - Gets a list of methods  defined in the given class
#                  Can only be used with -dumpall or -dumptxt
#
#    Usage:
#      Cgm_MethodList <progfile> <classname> <tag>
#         ==> { {<methlodname> <methodprogfile> <methodtag>} ... }
proc Cgm_MethodList {progfile classname classtag} {
    global funcs

    set methodlist [list]
    foreach entry [array names funcs "*,class"] {
	if {([lindex $funcs($entry) 0] == $classname) && \
		([lindex $funcs($entry) 2] == $classtag)} {
	    regexp {(.+),([0-9]+),class$} $entry \
		    fullmatch methodprogfile methodtag
	    set tmpinfo $funcs($methodprogfile,$methodtag,file)
	    set deffile [lindex $tmpinfo 1]
	    set defline [lindex $tmpinfo 0]
	    set methodname $funcs($methodprogfile,$methodtag,name)
	    if {$defline != 0 && ![info exists \
		    __tmp_unique_methods($methodname,$deffile,$defline)]} {
		set __tmp_unique_methods($methodname,$deffile,$defline) 1
		lappend methodlist [list \
			$methodname \
			$methodprogfile \
			$methodtag]
	    }
	}
    }
    return $methodlist
}


###########################################################################
#
# Cgm_FuncInfo - Gets function information from a depfile that has been 
#                loaded into memory with the Cgm_LoadDep or Cgm_LoadAllDeps 
#                functions.  The specified progfile MUST be loaded.  
#                Valid fields are: state, fold, name, class, file, pos,
#                type, num, calls, childpos, and childline.  
#                If the optional <newval> parameter is given, the 
#                function info is reset to a new value.
#    Usage:
#      Cgm_FuncInfo <progfile> <tag> <field> [ <newval> ]
#        ==> (See data descriptions)
proc Cgm_FuncInfo {progfile tag field {newval ""}} {
    global depfiles_loaded funcs

    if {[lsearch -exact $depfiles_loaded $progfile] < 0} {
	return CGM_FAILED
    }
    
    if {$newval == ""} {
	if [info exists funcs($progfile,$tag,$field)] {
	    return $funcs($progfile,$tag,$field)
	} else {
	    return CGM_DOESNT_EXIST
	}
    } else {
	set funcs($progfile,$tag,$field) $newval
    }
}


###########################################################################
#
# Cgm_ClassInfo - Gets class information from a depfile that has been 
#                 loaded into memory with the Cgm_LoadDep or Cgm_LoadAllDeps 
#                 functions.  The specified progfile MUST be loaded.  
#                 Valid fields are: file, name, coll, pos,
#                 typepos, subs, fmempos, basepos, and mem.
#                 If the optional <newval> parameter is given, the 
#                 class info is reset to a new value.
#    Usage:
#      Cgm_ClassInfo <progfile> <tag> <field> [ <newval> ]
#        ==> (See data descriptions)
proc Cgm_ClassInfo {progfile tag field {newval ""}} {
    global depfiles_loaded classes

    if {[lsearch -exact $depfiles_loaded $progfile] < 0} {
	return CGM_FAILED
    }
    
    if {$newval == ""} {
	if [info exists classes($progfile,$tag,$field)] {
	    return $classes($progfile,$tag,$field)
	} else {
	    return CGM_DOESNT_EXIST
	}
    } else {
	set classes($progfile,$tag,$field) $newval
    }
}


###########################################################################
#
# Cgm_MemInfo - Gets member information from a depfile that has been 
#               loaded into memory with the Cgm_LoadDep or Cgm_LoadAllDeps 
#               functions.  The specified progfile MUST be loaded.  
#               If the optional <newval> parameter is given, the 
#               member info is reset to a new value.
#    Usage:
#      Cgm_MemInfo <progfile> <tag> [ <newval> ] 
#        ==> { <member-name> <tag> <type> }
proc Cgm_MemInfo {progfile tag {newval ""}} {
    global depfiles_loaded members

    if {[lsearch -exact $depfiles_loaded $progfile] < 0} {
	return CGM_FAILED
    }
    
    if {$newval == ""} {
	if [info exists members($progfile,$tag)] {
	    return $members($progfile,$tag)
	} else {
	    return CGM_DOESNT_EXIST
	}
    } else {
	set members($progfile,$tag) $newval
    }
}


###########################################################################
#
# Cgm_DepInfo - Gets general depfile information from a depfile that has been 
#               loaded into memory with the Cgm_LoadDep or Cgm_LoadAllDeps 
#               functions.  The specified progfile MUST be loaded.  
#               Valid fields are: path, file, allftags, allctags, numfunc,
#               numused, numprof, ino, and type.
#               If the optional <newval> parameter is given, the 
#               depfile info is reset to a new value.
#    Usage:
#      Cgm_DepInfo <progfile> <field> [ <newval> ] 
#        ==> (See data descriptions)
proc Cgm_DepInfo {progfile field {newval ""}} {
    global depfiles_loaded depfile

    if {[lsearch -exact $depfiles_loaded $progfile] < 0} {
	return CGM_FAILED
    }
    
    if {$newval == ""} {
	if [info exists depfile($progfile,$field)] {
	    return $depfile($progfile,$field)
	} else {
	    return CGM_DOESNT_EXIST
	}
    } else {
	set depfile($progfile,$field) $newval
    }
}


###########################################################################
#
# Cgm_IsDepLoaded - Boolean function returns 1 if the depfile associated 
#                   with the specified progfile is loaded into memory, 
#                   or 0 (zero) otherwise.
#    Usage:
#      Cgm_IsDepLoaded <progfile>
#        ==> <Boolean>
proc Cgm_IsDepLoaded {progfile} {
    global depfiles_loaded
    if {[lsearch -exact $depfiles_loaded "$progfile"] >= 0} {
	return 1
    } else {
	return 0
    }
}


###########################################################################
#
# Cgm_LoadDep - Loads the depfile information for a given progfile with the
#               given cgm options.  Returns nothing.  See the CGM document,
#               "Interfacing with TAU" for option descriptions.
#    Usage:
#      Cgm_LoadDep <progfile> <cgm-option>
proc Cgm_LoadDep {progfile option} {

  # -- get access to global vars
  global depfile depfiles_loaded
  global funcs classes members
  # OBSOLETE: global functags classtags
  global BINDIR REMSH

  if {[Cgm_IsDepLoaded $progfile]} {
      # Remove the old dep data
      Cgm_RemoveDep $progfile
  }
      
  # -- determine name of depfile
  set lang [PM_SetLangOption $progfile]
  set depfile($progfile,file) [format "%s.%s" [file rootname $progfile] \
	  [Lang_GetExt $lang]]
  set depfile($progfile,path) "$depfile(dir)/$depfile($progfile,file)"

  # -- check file
  if { $depfile($progfile,path) == [status RR] } {
    showError "Cannot reload file: `$depfile($progfile,path)'." fatal
    return CGM_FAILED
  } elseif { $depfile(host) == "localhost" } {
    # -- depfile on local host
    if {![file exists $depfile($progfile,path)] || \
	    ![file readable $depfile($progfile,path)]} {
      showError "Cannot open file: `$depfile($progfile,path)'.\n\
	      Browser information not compiled!"
      return CGM_FAILED
    }
    file stat $depfile($progfile,path) statbuf
    set depfile($progfile,ino) $statbuf(ino)
  } else {
    # -- depfile on remote host
    set depfile($progfile,ino) -1
  }

  set temp_statusInfo [PM_Status]
  if {[string match $temp_statusInfo "UNDEFINED"]} {
      puts "Cgm_LoadDep: There is not a project loaded at this time."
      return
  }

  # All these sets are here to replace a similar functionality that used
  # to be in SetDepfileBasics. 
  set depfile($progfile,allftags) ""
  set depfile($progfile,allctags) ""
  set depfile($progfile,numfunc) 0
  set depfile($progfile,numused) 0
  set depfile($progfile,numprof) ""
  # End additional sets

  set cgm_cmd [Lang_GetCGM $lang]

  if { $depfile(host) == "localhost" } {
    # -- change directory to the one where depfile is located
    # -- so that relative pathnames coming from Sage++ are
    # -- interpreted correctly
    cd $depfile(dir)

    # -- read directly from cgm
    set readcom "|$BINDIR/$cgm_cmd $option $depfile($progfile,file)"
  } else {
    set REMBINDIR "$depfile(root)/bin/$depfile(arch)"

    # -- read through remote shell from cgm
    set readcom "|$REMSH $depfile(host) \
                -n \"cd $depfile(dir); $REMBINDIR/$cgm_cmd $option $depfile($progfile,file)\""
  }

  # -- initialize global variables
  set numfunc 0

  # -- open access to cgm reader
  set in [open $readcom r]

  # -- output for functions have header containing number of functions
  if { $option != "-dumpch" } {
    # -- Read number of functions
    gets $in line
    scan $line "%d" depfile($progfile,numfunc)
    gets $in line
  }

  # -- Read data from cgm
  while {[gets $in line] >= 0} {
      # treat it as a list so that quotes are handled right
      if {[llength $line] > 0} {
      set com  [lindex $line 0]
      set arg1 [lindex $line 1]
      set arg2 [lindex $line 2]
      if {[llength $line] >= 4} {
	  set arg3 [lindex $line 3]
      } else {
	  set arg3 "-"
      }

      switch $com {
      ftag:   { # -- tag: <tag> <name> <signature>
	      set tag $arg1
	      lappend depfile($progfile,allftags) $tag
	      set funcs($progfile,$tag,state) 0
	      set funcs($progfile,$tag,fold) 1
	      incr numfunc
	      
	      set funcs($progfile,$tag,name) $arg2
	      set funcs($progfile,$tag,mname) $arg3
      }

      fclass: { # -- class: <name|-> <type|-> <tag|-2>
	      set funcs($progfile,$tag,class) [list $arg1 $arg2 $arg3]
      }

      ffile:  { # -- file: <line|0> <file|-> @ <position>
                set funcs($progfile,$tag,file) [list $arg1 $arg2]

                if { $arg1 != 0 } {
                  set x [string first "@" $line]
                  incr x 2
                  set funcs($progfile,$tag,pos) [string range $line $x end]
                }
              }

      ftype:  { # -- type: <par|seq> <type> <used|not>
                set funcs($progfile,$tag,type) [list $arg1 $arg2 $arg3]
                if { $arg3 == "used" } { incr depfile($progfile,numused) }
              }

      fcalls: { # -- calls: <num-called> - <list of calls>
                set funcs($progfile,$tag,num) $arg1

                set x [string first "-" $line]
                incr x 2
                set funcs($progfile,$tag,calls) [string range $line $x end]
              }

      fpos:   { # -- pos: <list of call-site-positions>
                set x [string first ":" $line]
                incr x 2
                set funcs($progfile,$tag,childpos) [string range $line $x end]
              }

      fline:  { # -- line: <list of call-site-linenumbers>
                set x [string first ":" $line]
                incr x 2
                set funcs($progfile,$tag,childline) [string range $line $x end]
              }

      ctag:   { # -- tag: <tag> <name> <COLL|->
                set tag $arg1
                if [ info exists classes($progfile,$tag,name) ] {
                  set tag -2
                } else {
                  lappend depfile($progfile,allctags) $tag
                  set classes($progfile,$tag,name) $arg2
                  set classes($progfile,$tag,coll) $arg3
		  #OBSOLETE: lappend classtags($progfile,$arg2) $arg1
                }
              }

      cfile:  { # -- file: <line|0> <file|-> @ <position>
                set classes($progfile,$tag,file) [list $arg1 $arg2]

                if { $arg1 != 0 } {
                  set x [string first "@" $line]
                  incr x 2
                  set classes($progfile,$tag,pos) [string range $line $x end]
                }
              }

      cpos:   { # -- pos: <list of type-positions>
                set x [string first ":" $line]
                incr x 2
                set classes($progfile,$tag,typepos) [string range $line $x end]
              }

      csub:   { # -- sub: <list-of-subclasses>
                set x [string first ":" $line]
                incr x 2
                set classes($progfile,$tag,subs) [string range $line $x end]
              }
      cmpos:  { # -- mpos: <list of member-func-decl-positions>
                set x [string first ":" $line]
                incr x 2
                set classes($progfile,$tag,fmempos) [string range $line $x end]
              }

      cbase:  { # -- base: <list-of-baseclasses-and-positions>
                set x [string first ":" $line]
                incr x 2
                set classes($progfile,$tag,basepos) [string range $line $x end]
              }

      cmem:   { # -- mem: <member> <tag> <type>
                set members($progfile,$arg1) [lrange $line 1 3]
                lappend classes($progfile,$tag,mem) \
			[list $progfile $arg2 $arg1]
              }
    }
    set line ""
    set com ""
  } ;# if items
  } ;# while
  if [catch {close $in} errmsg] {
    showError "$readcom: `$errmsg'."
    if { ! [regexp -nocase "warning" $errmsg] } {
      return CGM_FAILED
    }
  }
  lappend depfiles_loaded $progfile
  return $numfunc
}


###########################################################################
#
# Cgm_LoadAllDeps - Loads the depfiles for ALL progfiles in the application
#                   with the given cgm options.  This may take a substantial
#                   ammount of time!  Returns nothing.
#    Usage:
#      Cgm_LoadAllDeps <cgm-option>
proc Cgm_LoadAllDeps {option} {
    foreach progfile [PM_GetFiles] {
	Cgm_LoadDep $progfile $option
    }
}


###########################################################################
#
# Cgm_RemoveDep - Removes the depfile information for a given progfile from
#                 memory.  Returns nothing.
#    Usage:
#      Cgm_RemoveDep <progfile>
proc Cgm_RemoveDep {filen} {
    global funcs classes members depfile
    global depfiles_loaded

    set depfiles_loaded [lremove -exact $depfiles_loaded "$filen"]
    
    foreach entry [array names depfile "$filen,*"] {
	unset depfile($entry)
    }
    foreach entry [array names funcs "$filen,*"] {
	unset funcs($entry)
    }
    foreach entry [array names classes "$filen,*"] {
	unset classes($entry)
    }
    foreach entry [array names members "$filen,*"] {
	unset members($entry)
    }    
}


###########################################################################
#
# Cgm_RemoveAllDeps - Removes all the depfile information from
#                     memory.  Returns nothing.
#    Usage:
#      Cgm_RemoveAllDeps 
proc Cgm_RemoveAllDeps {} {
    global funcs classes members depfile
    global depfiles_loaded

    if [info exists funcs] {
	unset funcs
    }
    if [info exists classes] {
	unset classes
    }
    if [info exists members] {
	unset members
    }
    if [info exists depfile(path)] {
	unset depfile(path)
    }
    if [info exists depfile(file)] {
	unset depfile(file)
    }
    if [info exists depfile(allftags)] {
	unset depfile(allftags)
    }
    if [info exists depfile(allctags)] {
	unset depfile(allctags)
    }
    if [info exists depfile(numfunc)] {
	unset depfile(numfunc)
    }
    if [info exists depfile(numused)] {
	unset depfile(numused)
    }
    if [info exists depfile(numprof)] {
	unset depfile(numprof)
    }
    if [info exists depfile(ino)] {
	unset depfile(ino)
    }
    if [info exists depfile(type)] {
	unset depfile(type)
    }

    set depfiles_loaded [list]
}





# -- initialize global variables
initDep
