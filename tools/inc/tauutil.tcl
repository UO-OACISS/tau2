#*********************************************************************#
#*                TAU/pC++/Sage++  Copyright (C) 1994                *#
#*                        Jerry Manic Saftware                       *#
#*  Indiana University  University of Oregon  University of Rennes   *#
#*********************************************************************#

#
# tauutil.c: accomodates standalone version of cosy - 6/97 kal
#


proc tau_ProjectOptions {} {
    #This procedure takes no arguments. It access the current set of project options, and
    #allows the user to make changes in the variables that tau is using to manipulate
    #the project. Currently the TAUROOT variable is accessible. . .

    global depfile srVar
    proc tau_ProjOptsInternal {} {
	
	global depfile 
	
    }
    proc maintainPMF {throw these away} {
	global depfile

	if {![string match ".pmf" [file ext $depfile(projnameTemp)]]} {
	    set depfile(projnameTemp) "$depfile(projnameTemp).pmf"
	    .tauProjOpts.fr3.newNameEnt icursor \
		[expr [string length $depfile(projnameTemp)] - 4]
	}
    }

    # sageRoot is an obsolete concept - now sageroot refers to tau root
    if [winfo exists .tauProjOpts] {
	raise .tauProjOpts
    } else {
	toplevel .tauProjOpts
	wm title .tauProjOpts "Project Options"
	set rootTemp [frame .tauProjOpts.fr1]
	pack $rootTemp \
	    -expand 1 \
	    -fill both \
	    -padx 10 \
	    -pady 10
	label $rootTemp.sageRootlabel -text "TAU Root:"
	entry $rootTemp.sageRootentry -width 50 -textvariable srVar
	pack $rootTemp.sageRootlabel $rootTemp.sageRootentry \
	    -side left \
	    -anchor nw \
	    -fill x
	$rootTemp.sageRootentry delete 0 end
	$rootTemp.sageRootentry insert end $depfile(root)
	bind $rootTemp.sageRootentry <Return> {.tauProjOpts.fr2.enter flash;\
						   .tauProjOpts.fr2.enter invoke}
	button $rootTemp.sageRootSearch \
	    -text Search \
	    -command {set srVar \
			  [getFile "Set TAUROOT Directory" \
			       "" 0 $depfile(host) ""]}
	pack $rootTemp.sageRootSearch -side left -anchor nw
	set nameTemp [frame .tauProjOpts.fr3]
	pack $nameTemp \
	    -expand 1 \
	    -fill both \
	    -padx 10 \
	    -pady 10
	label $nameTemp.newNameLabel \
	    -text "Project Name:"
	entry $nameTemp.newNameEnt -width 30 -textvariable depfile(projnameTemp)
	pack $nameTemp.newNameLabel $nameTemp.newNameEnt \
	    -side left \
	    -anchor nw \
	    -fill x
	$nameTemp.newNameEnt delete 0 end
	set depfile(projnameTemp) $depfile(project)
	trace variable depfile(projnameTemp) w maintainPMF
	set depfile(projnameTemp) $depfile(project)
	set this [frame .tauProjOpts.fr2]
	pack $this \
	    -expand 1 \
	    -fill both \
	    -padx 10 \
	    -pady 10
	button $this.enter -text Enter \
	    -command {set depfile(root) $srVar;\
			  set depfile(project) $depfile(projnameTemp); \
			  Bdb_ChangeName $depfile(projnameTemp); \
			  PM_SetRoot $srVar; \
			  PM_ChangeProjectName $depfile(projnameTemp); \
                          PM_BroadcastChanges {-} p;
			  destroy .tauProjOpts}
	button $this.cancel -text Cancel -command "destroy .tauProjOpts"
	pack $this.enter \
	    -side left 

	pack $this.cancel \
	    -side right 
    }
}


proc tau_OpenProject {} {
    # For use by tau after the initial project is opened, to
    # switch projects and associated information.
    global depfile pm_status
    
    PM_OpenProject
    set pm_status [PM_Status]
    set depfile(project) [lindex $pm_status 0]
    set depfile(host)    [lindex $pm_status 1]
    set depfile(arch)    [lindex $pm_status 2]
    set depfile(root)    [lindex $pm_status 3]
    set depfile(dir)     [lindex $pm_status 4]
    if {[string match $depfile(dir) "."]} {
	set depfile(dir) [pwd]
    }
    PM_BroadcastChanges [list] p
    #SetFileDisplay
}


proc tau_CopyProject {} {
    # For use by tau and cosy after the initial project is
    # opened, to change the name of the project
    global depfile pm_status
    
    PM_ChangeProjectName
    set pm_status [PM_Status]
    set depfile(project) [lindex $pm_status 0]
    PM_BroadcastChanges [list] p
}


proc tau_AddFile {} {
    # Interface for adding a file to the project, and updating the display.

    PM_AddFile
    #SetFileDisplay
}


proc tau_DeleteFile {{filename ""}} {

    global myself depfile

    set lb .$myself.info2.ext.int.bot.list
       
    if [string match $filename ""] {
	if [string match [$lb curselection] ""] {
	    showError "No File Selected"
	    return
	} else {
	    set temp [$lb get [$lb curselection]]
	    scan $temp "%s" filename
	}
    }
#    puts "tau_DF: Expanded filename == $depfile(dir)/$filename"
#    PM_RemoveFile $depfile(dir)/$filename
    PM_RemoveFile $filename
    #SetFileDisplay
}


proc SetFileDisplay {} {
    # This procedure gets the current list of files and associated language
    # information, and displays it in the GUI.
    # Returns nothing.
    
    global myself

    set filelist [PM_GetFiles]
    if [winfo exists .tau.info2.ext.int.bot.list] {
	.tau.info2.ext.int.bot.list delete 0 end
    }
    if [winfo exists .cosy.info2.ext.int.bot.list] {
	.cosy.info2.ext.int.bot.list delete 0 end
    }

    foreach elem $filelist {
	set langtemp [PM_SetLangOption $elem]
	set holder $elem
	for {set i 1} {$i < [expr 68 - [string length $elem]]} {incr i} {
	    set holder "$holder "
	}
	set holder "$holder$langtemp"
	if [winfo exists .tau.info2.ext.int.bot.list] {
	    .tau.info2.ext.int.bot.list insert end $holder
	}
	if [winfo exists .cosy.info2.ext.int.bot.list] {
	    .cosy.info2.ext.int.bot.list insert end $holder
	}
    }
}
