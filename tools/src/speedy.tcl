#*********************************************************************#
#*                TAU/pC++/Sage++  Copyright (C) 1994                *#
#*                        Jerry Manic Saftware                       *#
#*  Indiana University  University of Oregon  University of Rennes   *#
#*********************************************************************#
displayMessage hourglass "Loading $myself..."

source "$TAUDIR/inc/depfile.tcl"
source "$TAUDIR/inc/selectfile.tcl"
source "$TAUDIR/inc/xygraph.tcl"
source "$TAUDIR/inc/printcan.tcl"
source "$TAUDIR/inc/limit.tcl"

#
# -- global variables
#

set speedy_progfile "";   # the single pc++ program file

set sp_title "General"
set sp_curr_param ""
set sp_curr_group .invalid
set sp_frac 0

set sp_var(1,param) "--NONE--"
set sp_var(2,param) "--NONE--"

set sp_max_namlen 22
set sp_max_vallen 28

set sp_color [list \
    red blue forestgreen orange purple brown \
    grey coral skyblue seagreen indianred royalblue \
    violet steelblue magenta black ]

# -- symbolic labels for discrete value paramters
set sp_symb(BarrierByMsgs) {False True}
set sp_vals(BarrierByMsgs) {0 1}
set sp_symb(ProcessMsgType) {AtBarriersAndRepliesOnly
                             PredefinedPolling
                             PeriodicPolling
                             PredefinedAndPeriodicPolling
                             Interruptible }
set sp_vals(ProcessMsgType) {0 1 2 3 4}
set sp_symb(Topology) {FullyConnected
                       Ring
                       Shared
                       RingWithContention
                       PipelinedRing }
set sp_vals(Topology) {0 1 2 3 4}
set sp_symb(_ExperimentMode) {WholeProgram SelectedParts}
set sp_vals(_ExperimentMode) {K E}
set sp_symb(_TrcRuntimeSystem) {Awesime LWP "POSIX Threads" "ATT Task Library"}
set sp_vals(_TrcRuntimeSystem) {awe lwp pthread task}
set sp_symb(_UseElemSize) {False True}
set sp_vals(_UseElemSize) {0 1}
set sp_symb(_KeepTemp) {False True}
set sp_vals(_KeepTemp) {0 1}

set sp_name(_vParam1)          "varying parameter 1"
set sp_name(_vParam2)          "varying parameter 2"
set sp_name(_NumProc)          "Number of Processors"
set sp_name(_ExperimentMode)   "ExperimentMode"
set sp_name(_TrcRuntimeSystem) "Generate Traces using"
set sp_name(_UseElemSize)      "Use Element Size"
set sp_name(_TraceOverhead)    "Tracing Overhead"
set sp_name(_KeepTemp)         "Keep Temporary Files"
set sp_name(Delta)             "Polling Period"
set sp_name(CommStartupTime)   "Latency"
set sp_name(--NONE--)          ""

#
# exit - TAU exit function that communicates the event to other tau tools.
#
rename exit exit.old
proc exit {{status 0}} {
    global myself

    PM_RemGlobalSelect $myself \
	    { global_selectFuncTag }
    PM_RemTool $myself

    exit.old
}


#
# wraplindex: like lindex, but index "wraps around" if bigger
#             than length of list
#

proc wraplindex {list index} {
  lindex $list [expr $index%[llength $list]]
}

#
# resetParam: Set default parameter values
#

proc resetParam {} {
  global sp_param

  # -- General
  set sp_param(_vParam1)              --NONE--
  set sp_param(_vParam2)              --NONE--
  set sp_param(_NumProc)              1
  set sp_param(_ExperimentMode)       K
  set sp_param(_RunList1)             1
  set sp_param(_RunList2)             1
  set sp_param(_TrcRuntimeSystem)     task
  set sp_param(_UseElemSize)          0
  set sp_param(_TraceOverhead)        0
  set sp_param(_KeepTemp)             0

  # -- Barrier Model
  set sp_param(BarrierByMsgs)         1
  set sp_param(BarrierMsgSize)        0
  set sp_param(BarrierEntryTime)      0.0
  set sp_param(BarrierExitTime)       0.0
  set sp_param(BarrierCheckTime)      0.0
  set sp_param(BarrierExitCheckTime)  0.0
  set sp_param(BarrierModelTime)      0.0

  # -- Processor Model:
  set sp_param(MipsRatio)             1.0
  set sp_param(ProcessMsgType)        0
  set sp_param(Delta)                 0.0

  # -- Runtime System Model:
  set sp_param(BytePackTime)          0.0
  set sp_param(PackStartupTime)       0.0
  set sp_param(GetMsgTryTime)         0.0
  set sp_param(GetMsgStartupTime)     0.0
  set sp_param(GetMsgTime)            0.0
  set sp_param(MsgProcessingTime)     0.0
  set sp_param(RequestMsgSize)        0

  # -- Interconnect Network Model:
  set sp_param(CommStartupTime)       0.0
  set sp_param(ByteTransferTime)      0.0
  set sp_param(Topology)              2

  # -- Network Interface Model:
  set sp_param(ToNWStartupTime)       0.0
  set sp_param(ByteToNWTransferTime)  0.0
  set sp_param(AddMsgStartupTime)     0.0
  set sp_param(AddMsgTryTime)         0.0
  set sp_param(AddMsgTime)            0.0
}

#
# proc discreteParam: create parameter entry in display area for
#                     a discrete value parameter
#
#                win: window id
#            varName: parameter name used in parameter file
#

proc discreteParam {win varName {handler {}}} {
  global sp_param sp_symb sp_vals sp_name
  global sp_max_namlen sp_max_vallen

  if [info exists sp_name($varName)] {
    set name $sp_name($varName)
  } else {
    set name $varName
  }

  frame $win
  label $win.name -text "${name}:" -width $sp_max_namlen -anchor e
  pack $win.name -side left -anchor e -padx 5

  # -- find symbolic name of current value
  set i [lsearch -exact $sp_vals($varName) $sp_param($varName)]
  set v [lindex $sp_symb($varName) $i]

  # -- setup menu for changing discrete values
  menubutton $win.val -text $v -relief sunken -borderwidth 2 \
                      -menu $win.val.m -width $sp_max_vallen -anchor w \
                      -highlightthickness 0
  menu $win.val.m -tearoff 0
  set i 0
  foreach s $sp_symb($varName) {
    $win.val.m add command -label $s -command "
      $win.val configure -text \"$s\"
      set sp_param($varName) \"[lindex $sp_vals($varName) $i]\"
      if { \"$handler\" != \"\" } {
        $handler $varName \"[lindex $sp_vals($varName) $i]\"
      }
    "
    incr i
  }
  pack $win.val -side left -anchor w -padx 5

  label $win.lab -text { } -width 7 -anchor w
  pack $win.lab -side left -anchor w -padx 5

  # -- set trace on variable, so we can keep parameter updated
  trace variable sp_param($varName) w "setDisLabel $win.val"
  
  return $win
}

#
# setDisLabel: update value label for discrete parameter values
#
#         win: window id of menubutton representing the value
#       array: array name (should be always "sp_param")
#       field: parameter name
#          op: mode of access (should be always "w")
#

proc setDisLabel {win array field op} {
  global sp_param sp_symb sp_vals

  set i [lsearch -exact $sp_vals($field) $sp_param($field)]
  set v [lindex $sp_symb($field) $i]
  $win configure -text $v
}

#
# proc numberParam: create parameter entry in display area for
#                   a number value parameter
#
#              win: window id
#          varName: parameter name used in parameter file
#             unit: unit label to display right of entry (can be empty)
#            isint: is integer?
#

proc numberParam {win varName unit isint} {
  global myself
  global sp_param sp_curr_param sp_name
  global sp_max_namlen sp_max_vallen

  if [info exists sp_name($varName)] {
    set name $sp_name($varName)
  } else {
    set name $varName
  }

  if {![info exists sp_param($varName)]} {
    set sp_param($varName) {}
  }

  frame $win
  label $win.name -text "${name}:" -width $sp_max_namlen -anchor e
  pack $win.name -side left -anchor e -padx 5

  entry $win.val -relief sunken -borderwidth 2 -width $sp_max_vallen \
                 -textvariable sp_param($varName) -highlightthickness 0
  pack $win.val -side left -anchor w -padx 5 -ipadx 4 -ipady 3

  if { $unit != "" } {
    label $win.lab -text "\[$unit\]" -width 7 -anchor w
    pack $win.lab -side left -anchor w -padx 5
  }

  bind $win.val <1> "set sp_curr_param $varName; setVal .$myself.p 0"

  if {$isint} {
      trace variable sp_param($varName) w forceInt
  } else {
      trace variable sp_param($varName) w forceReal
  }

  set sp_param($varName) $sp_param($varName)
  
  return $win
}

#
# group*: display group of parameters
#

proc group0 {w} {
  global sp_title sp_curr_group

  set sp_title "General"
  set win $w.grp0

  # -- "undisplay" old group"
  if [winfo exists $sp_curr_group] {
    pack forget $sp_curr_group
  }

  if {![winfo exists $win]} {
    frame $win
    frame $win.left
    frame $win.right

    numberParam $win.p1 _NumProc "" true
    discreteParam $win.p2 _ExperimentMode
    discreteParam $win.p3 _TrcRuntimeSystem
    discreteParam $win.p4 _UseElemSize
    numberParam $win.p5 _TraceOverhead us false
    discreteParam $win.p6 _KeepTemp

    pack $win.p1 $win.p2 $win.p3 $win.p4 $win.p5 $win.p6\
         -side top -padx 5 -pady 5 -anchor w -in $win.left
    pack $win.left $win.right -side left -anchor nw
  }

  pack $win -side top -padx 5 -pady 5 -anchor w -in $w.grp
  set sp_curr_group $win
}

proc group1 {w} {
  global sp_title sp_curr_group

  set sp_title "Barrier"
  set win $w.grp1

  # -- "undisplay" old group"
  if [winfo exists $sp_curr_group] {
    pack forget $sp_curr_group
  }

  if {![winfo exists $win]} {
    frame $win
    frame $win.left
    frame $win.right

    discreteParam $win.p1 BarrierByMsgs
    numberParam $win.p2 BarrierMsgSize bytes true
    numberParam $win.p3 BarrierEntryTime us false
    numberParam $win.p4 BarrierExitTime us false
    numberParam $win.p5 BarrierCheckTime us false
    numberParam $win.p6 BarrierExitCheckTime us false
    numberParam $win.p7 BarrierModelTime us false

    pack $win.p1 $win.p2 $win.p3 $win.p4 $win.p5 $win.p6 \
         -side top -padx 5 -pady 5 -anchor w -in $win.left
    #pack <win> -side top -padx 5 -pady 5 -anchor w -in $win.right
    pack $win.left $win.right -side left -anchor nw
  }

  pack $win -side top -padx 5 -pady 5 -anchor w -in $w.grp
  set sp_curr_group $win
}

proc group2 {w} {
  global sp_title sp_curr_group

  set sp_title "Processor"
  set win $w.grp2

  # -- "undisplay" old group"
  if [winfo exists $sp_curr_group] {
    pack forget $sp_curr_group
  }

  if {![winfo exists $win]} {
    frame $win
    frame $win.left
    frame $win.right

    numberParam $win.p1 MipsRatio "" false
    discreteParam $win.p2 ProcessMsgType
    numberParam $win.p3 Delta us false

    pack $win.p1 $win.p2 $win.p3 \
         -side top -padx 5 -pady 5 -anchor w -in $win.left
    #pack <win> -side top -padx 5 -pady 5 -anchor w -in $win.right
    pack $win.left $win.right -side left -anchor nw
  }

  pack $win -side top -padx 5 -pady 5 -anchor w -in $w.grp
  set sp_curr_group $win
}

proc group3 {w} {
  global sp_title sp_curr_group

  set sp_title "Runtime System"
  set win $w.grp3

  # -- "undisplay" old group"
  if [winfo exists $sp_curr_group] {
    pack forget $sp_curr_group
  }

  if {![winfo exists $win]} {
    frame $win
    frame $win.left
    frame $win.right

    numberParam $win.p1 BytePackTime us false
    numberParam $win.p2 PackStartupTime us false
    numberParam $win.p3 GetMsgTryTime us false
    numberParam $win.p4 GetMsgStartupTime us false
    numberParam $win.p5 GetMsgTime us false
    numberParam $win.p6 MsgProcessingTime us false
    numberParam $win.p7 RequestMsgSize bytes true

    pack $win.p1 $win.p2 $win.p3 $win.p4 $win.p5 $win.p6 $win.p7 \
         -side top -padx 5 -pady 5 -anchor w -in $win.left
    #pack <win> -side top -padx 5 -pady 5 -anchor w -in $win.right
    pack $win.left $win.right -side left -anchor nw
  }

  pack $win -side top -padx 5 -pady 5 -anchor w -in $w.grp
  set sp_curr_group $win
}

proc group4 {w} {
  global sp_title sp_curr_group

  set sp_title "Interconnect Network"
  set win $w.grp4

  # -- "undisplay" old group"
  if [winfo exists $sp_curr_group] {
    pack forget $sp_curr_group
  }

  if {![winfo exists $win]} {
    frame $win
    frame $win.left
    frame $win.right

    numberParam $win.p1 CommStartupTime us false
    numberParam $win.p2 ByteTransferTime us false
    discreteParam $win.p3 Topology

    pack $win.p1 $win.p2 $win.p3 \
         -side top -padx 5 -pady 5 -anchor w -in $win.left
    #pack <win> -side top -padx 5 -pady 5 -anchor w -in $win.right
    pack $win.left $win.right -side left -anchor nw
  }

  pack $win -side top -padx 5 -pady 5 -anchor w -in $w.grp
  set sp_curr_group $win
}

proc group5 {w} {
  global sp_title sp_curr_group

  set sp_title "Network Interface"
  set win $w.grp5

  # -- "undisplay" old group"
  if [winfo exists $sp_curr_group] {
    pack forget $sp_curr_group
  }

  if {![winfo exists $win]} {
    frame $win
    frame $win.left
    frame $win.right

    numberParam $win.p1 ToNWStartupTime us false
    numberParam $win.p2 ByteToNWTransferTime us false
    numberParam $win.p3 AddMsgStartupTime us false
    numberParam $win.p4 AddMsgTryTime us false
    numberParam $win.p5 AddMsgTime us false

    pack $win.p1 $win.p2 $win.p3 $win.p4 $win.p5 \
         -side top -padx 5 -pady 5 -anchor w -in $win.left
    #pack <win> -side top -padx 5 -pady 5 -anchor w -in $win.right
    pack $win.left $win.right -side left -anchor nw
  }

  pack $win -side top -padx 5 -pady 5 -anchor w -in $w.grp
  set sp_curr_group $win
}

#
# setVaryParam:
#
#        param:
#          val:
#

proc setVaryParam {param {val {}}} {
  global myself
  global sp_var sp_param sp_name sp_symb sp_vals

  regexp {[12]} $param n

  # -- set default parameter values
  if { $val == "" } {
    set val $sp_param($param)
    set def $sp_param(_RunList$n)
  } else {
    set def ""
  }

  # -- create window if necessary
  # -- do nothing / hide window if there is no varying parameter
  set win .$myself.v$n
  if [winfo exists $win ] {
    if { $val == "--NONE--" } {
      set sp_var($n,param) "--NONE--"
      wm withdraw $win
      return
    } else {
      wm deiconify $win
    }
  } else {
    if { $val == "--NONE--" } {
      set sp_var($n,param) "--NONE--"
      return
    }
    toplevel $win
  }

  if { $sp_var($n,param) != $val } {
    # -- different parameter

    if [ info exists sp_symb($val) ] {
      # -- this is a discrete parameter
      # -- get rid of old widgets
      if [ winfo exists $win.1 ] {
        pack forget $win.1 $win.2 $win.3
      }
      if [ winfo exists $win.d ] {
        destroy $win.d
      }

      # -- build new discrete windows
      frame $win.d
      frame $win.d.l
      frame $win.d.r
      set i 0
      foreach v $sp_vals($val) {
        set sp_var($n,dis,$i) {}
        checkbutton $win.d.c$i -offvalue {} -onvalue $v -relief flat \
                    -text "[lindex $sp_symb($val) $i] ($v)" \
                    -variable sp_var($n,dis,$i)
        if {$i % 2 == 0} {
          pack $win.d.c$i -side top -anchor w -in $win.d.l
        } else {
          pack $win.d.c$i -side top -anchor w -in $win.d.r
        }
        incr i
      }
      pack $win.d.l -side left -padx 5 -pady 5 -anchor nw
      pack $win.d.r -side right -padx 5 -pady 5 -anchor nw
      pack $win.d -side top -padx 5 -pady 5

      set sp_var($n,mode) dis

      # -- set default values
      foreach d $def {
        set sp_var($n,dis,[lsearch -exact $sp_vals($val) $d]) $d
      }
    } else {
      # -- this is number parameters
      # -- get rid of possible discrete parameter widgets
      if [ winfo exists $win.d ] {
        destroy $win.d
      }

      if { ! [ winfo exists $win.1 ] } {
        # -- we don't have old number parameter widgets around
        # -- create them
        frame $win.1
        radiobutton $win.1.r -text "multiples of" -width 12 -anchor w\
                    -relief flat -variable sp_var($n,mode) -value "mult"
        entry $win.1.e1 -relief sunken -borderwidth 2 -width 5 \
              -textvariable sp_var($n,mval)
        label $win.1.l2 -text "from"
        entry $win.1.e2 -relief sunken -borderwidth 2 -width 5 \
              -textvariable sp_var($n,mfrom)
        label $win.1.l3 -text "to"
        entry $win.1.e3 -relief sunken -borderwidth 2 -width 5 \
              -textvariable sp_var($n,mto)
        checkbutton $win.1.c -text "including 1" -variable sp_var($n,mone) \
              -relief flat
        pack $win.1.r $win.1.e1 \
             $win.1.l2 $win.1.e2 $win.1.l3 $win.1.e3 \
             -side left -padx 5
        pack $win.1.c -side left -padx 15

        bind $win.1.e1 <1> "set sp_var($n,mode) mult"
        bind $win.1.e2 <1> "set sp_var($n,mode) mult"
        bind $win.1.e3 <1> "set sp_var($n,mode) mult"

        frame $win.2
        radiobutton $win.2.r -text "powers of" -width 12 -anchor w\
                    -relief flat -variable sp_var($n,mode) -value "pow"
        entry $win.2.e1 -relief sunken -borderwidth 2 -width 5 \
              -textvariable sp_var($n,pval)
        label $win.2.l2 -text "from"
        entry $win.2.e2 -relief sunken -borderwidth 2 -width 5 \
              -textvariable sp_var($n,pfrom)
        label $win.2.l3 -text "to"
        entry $win.2.e3 -relief sunken -borderwidth 2 -width 5 \
              -textvariable sp_var($n,pto)
        pack $win.2.r $win.2.e1 \
             $win.2.l2 $win.2.e2 $win.2.l3 $win.2.e3 \
             -side left -padx 5

        bind $win.2.e1 <1> "set sp_var($n,mode) pow"
        bind $win.2.e2 <1> "set sp_var($n,mode) pow"
        bind $win.2.e3 <1> "set sp_var($n,mode) pow"

        frame $win.3
        radiobutton $win.3.r -text "random sequence" \
                    -relief flat -variable sp_var($n,mode) -value "rand"
        entry $win.3.e1 -relief sunken -borderwidth 2 -width 30 \
              -textvariable sp_var($n,seq)
        pack $win.3.r $win.3.e1 -side left -padx 5
        bind $win.3.e1 <1> "set sp_var($n,mode) rand"
      }
      pack $win.1 $win.2 $win.3 -side top -padx 5 -pady 5 -anchor w

      # -- setting default values for this number parameter
      # reset values
      set sp_var($n,mode)  rand
      set sp_var($n,mone)  0
      set sp_var($n,mval)  {}
      set sp_var($n,mfrom) {}
      set sp_var($n,mto)   {}
      set sp_var($n,pval)  {}
      set sp_var($n,pfrom) {}
      set sp_var($n,pto)   {}
      set sp_var($n,seq)   $sp_param($val)

      if { $def != "" } {
        set sp_var($n,mode)  rand
        set sp_var($n,seq)   $def
      }
    }

    # -- setting window title
    if [info exists sp_name($val)] {
      wm title $win "Set Parameter $n: $sp_name($val)"
    } else {
      wm title $win "Set Parameter $n: $val"
    }
    set sp_var($n,param) $val
  }
}

#
# createWindow: create and display main window for TAU master control
#

proc createWindow {} {
  global myself
  global TAUDIR
  global sp_var sp_max_namlen sp_max_vallen

  toplevel .$myself
  wm title .$myself "SPEEDY"
  wm iconbitmap .$myself @$TAUDIR/xbm/$myself.xbm

  # -- menubar
  frame .$myself.bar -relief raised -borderwidth 2
  menubutton .$myself.bar.b1 -text File -menu .$myself.bar.b1.m1 -underline 0
  menu .$myself.bar.b1.m1
  .$myself.bar.b1.m1 add command -label "Load experiment" -underline 0 \
                     -command "loadParam Experiment *.desc"
  .$myself.bar.b1.m1 add command -label "Print Graph" -underline 0 \
                     -command "printCanvas .$myself.sp6 graph"
  .$myself.bar.b1.m1 add command -label "Show Values" -underline 5 \
                     -command "showValues"
  .$myself.bar.b1.m1 add command -label "Show Speedup" -underline 5 \
                     -command "showSpeedup"
  .$myself.bar.b1.m1 add separator
  .$myself.bar.b1.m1 add command -label "Exit" -underline 0 -command "exit"

  menubutton .$myself.bar.b3 -text Help -menu .$myself.bar.b3.m1 -underline 0
  menu .$myself.bar.b3.m1
  .$myself.bar.b3.m1 add command -label "on $myself" -underline 3 \
               -command "xsend tau \[list showHelp $myself 1-$myself 1\]"
  .$myself.bar.b3.m1 add separator
  .$myself.bar.b3.m1 add command -label "on menubar" -underline 3 \
               -command "xsend tau \[list showHelp $myself 1.1-menu 1\]"
  .$myself.bar.b3.m1 add command -label "on display area" -underline 3 \
               -command "xsend tau \[list showHelp $myself 1.2-display 1\]"
  .$myself.bar.b3.m1 add command -label "on command buttons" -underline 3 \
               -command "xsend tau \[list showHelp $myself 1.3-buttons 1\]"
  .$myself.bar.b3.m1 add separator
  .$myself.bar.b3.m1 add command -label "on using help" -underline 3 \
               -command "xsend tau \[list showHelp general 1-help 1\]"

  createToolMenu .$myself.bar.b4

  pack .$myself.bar.b1 -side left -padx 5
  pack .$myself.bar.b3 .$myself.bar.b4 -side right -padx 5
  pack .$myself.bar -side top -fill x

  # -- speedy command buttons
  frame .$myself.sp1
  button .$myself.sp1.b1 -text "Compile" -command compileProg
  button .$myself.sp1.b2 -text "View/Set Parameters" \
                         -command "viewParam .$myself.p"
  button .$myself.sp1.b3 -text "Run Experiment" -command runExperiment
  pack .$myself.sp1.b1 .$myself.sp1.b2 .$myself.sp1.b3 -side left \
       -padx 5 -ipadx 5
  pack .$myself.sp1 -side top -fill x -padx 5 -pady 5

  # -- separator line
  frame .$myself.line1 -borderwidth 1 -relief sunken
  pack .$myself.line1 -side top -fill x -expand 1 -ipady 1

  # -- speedy experiment control
  initvParam 1 0
  frame .$myself.sp2
  discreteParam .$myself.sp2.p _vParam1 setVaryParam
  frame .$myself.sp3
  label .$myself.sp3.name -text values: -width $sp_max_namlen -anchor e
  pack .$myself.sp3.name -side left -anchor e -padx 5
  label .$myself.sp3.val -relief sunken -borderwidth 2 -width $sp_max_vallen \
                 -textvariable sp_param(_RunList1) -anchor w
  bind .$myself.sp3.val <1> "setVaryParam _vParam1"
  pack .$myself.sp3.val -side left -anchor w -padx 5 -ipadx 4 -ipady 3
  pack .$myself.sp2.p .$myself.sp3 -side top -padx 5 -pady 2 \
       -in .$myself.sp2 -anchor w

  initvParam 2 1
  frame .$myself.sp4
  discreteParam .$myself.sp4.p _vParam2 setVaryParam
  frame .$myself.sp5
  label .$myself.sp5.name -text values: -width $sp_max_namlen -anchor e
  pack .$myself.sp5.name -side left -anchor e -padx 5
  label .$myself.sp5.val -relief sunken -borderwidth 2 -width $sp_max_vallen \
                 -textvariable sp_param(_RunList2) -anchor w
  bind .$myself.sp5.val <1> "setVaryParam _vParam2"
  pack .$myself.sp5.val -side left -anchor w -padx 5 -ipadx 4 -ipady 3
  pack .$myself.sp4.p .$myself.sp5 -side top -padx 5 -pady 2 \
       -in .$myself.sp4 -anchor w

  pack .$myself.sp2 .$myself.sp4 -side top -padx 5 -pady 5

  # -- separator line
  frame .$myself.line2 -borderwidth 1 -relief sunken
  pack .$myself.line2 -side top -fill x -expand 1 -ipady 1

  # -- speedy result display
  NewXYGraph .$myself.sp6 300 300
  pack .$myself.sp6 -side top -padx 5 -pady 5
}

#
# showValues:
#

proc showValues {} {
  global sp_fname
  global myself

  if [winfo exists .showval] {
    raise .showval
  } else {
    toplevel .showval
    wm title .showval "Result Values"
    wm minsize .showval 50 50

    set sp_fname "graph.values"

    # -- menubar
    frame .showval.bar -relief raised -borderwidth 2
    menubutton .showval.bar.b1 -text File -menu .showval.bar.b1.m1 -underline 0
    menu .showval.bar.b1.m1
    .showval.bar.b1.m1 add command -label "Print Text" -underline 0 \
                       -command "printCanvas .showval.t1 values text"
    .showval.bar.b1.m1 add separator
    .showval.bar.b1.m1 add command -label "Exit" -underline 0 \
                       -command "destroy .showval"

    menubutton .showval.bar.b3 -text Help -menu .showval.bar.b3.m1 -underline 0
    menu .showval.bar.b3.m1
    .showval.bar.b3.m1 add command -label "on $myself" -underline 3 \
                 -command "xsend tau \[list showHelp $myself 1-$myself 1\]"
    .showval.bar.b3.m1 add separator
    .showval.bar.b3.m1 add command -label "on menubar" -underline 3 \
                 -command "xsend tau \[list showHelp $myself 1.1-menu 1\]"
    .showval.bar.b3.m1 add command -label "on display area" -underline 3 \
                 -command "xsend tau \[list showHelp $myself 1.2-display 1\]"
    .showval.bar.b3.m1 add command -label "on command buttons" -underline 3 \
                 -command "xsend tau \[list showHelp $myself 1.3-buttons 1\]"
    .showval.bar.b3.m1 add separator
    .showval.bar.b3.m1 add command -label "on using help" -underline 3 \
                 -command "xsend tau \[list showHelp general 1-help 1\]"

    pack .showval.bar.b1 -side left -padx 5
    pack .showval.bar.b3 -side right -padx 5
    pack .showval.bar -side top -fill x

    # -- text output field
    frame .showval.txt
    pack .showval.txt -side top -fill both -expand 1
    text .showval.t1 -height 10 -background white \
                      -foreground black -padx 5 -pady 5
    scrollbar .showval.s1 -orient vert -relief sunken \
              -command ".showval.t1 yview"
    .showval.t1 configure -yscrollcommand ".showval.s1 set"
    pack .showval.s1 -side right -fill y -in .showval.txt
    pack .showval.t1 -side top -fill both -expand 1 -in .showval.txt

    # -- put text in there
    formatValues .showval.t1
  }
}

#
# formatValues:
#
#          win:
#

proc formatValues {win} {
  global sp_time sp_param sp_max_vallen sp_color sp_name
  global sp_speedup sp_vals sp_symb

  # -- delete old contents
  $win configure -state normal
  $win delete 1.0 end
  $win tag delete [lpick [$win tag names] {val*}]

  # -- if no results we are done
  if { ! [info exists sp_time] } {
    $win configure -state disabled
    return
  }

  # -- write time header if varying parameter(s) exist
  $win insert end "Execution Time \[s\]\n"
  $win insert end "==================\n\n"

  if { $sp_param(_vParam1) != "--NONE--" } {
    if [info exists sp_name($sp_param(_vParam1))] {
      $win insert end [format "%*s | " $sp_max_vallen \
                       $sp_name($sp_param(_vParam1))]
    } else {
      $win insert end [format "%*s | " $sp_max_vallen $sp_param(_vParam1)]
    }
    foreach v $sp_param(_RunList1) { $win insert end [format "%10.3g" $v] }
    $win insert end [format "\n%*.*s-+-" $sp_max_vallen $sp_max_vallen \
         "------------------------------------------------------------"]
    foreach v $sp_param(_RunList1) { $win insert end "----------" }
    if { $sp_param(_vParam2) != "--NONE--" } {
      if [info exists sp_name($sp_param(_vParam2))] {
        $win insert end [format "\n%*s |\n" $sp_max_vallen \
                         $sp_name($sp_param(_vParam2))]
      } else {
        $win insert end [format "\n%*s |\n" $sp_max_vallen $sp_param(_vParam2)]
      }
    } else {
      $win insert end "\n"
    }
  }

  # -- print out time results in table form
  set i 0
  foreach vec [lsort [array names sp_time]] {
    if { $sp_param(_vParam2) != "--NONE--" } {
      set n2 [lindex $sp_param(_RunList2) $i]
      if [info exists sp_symb($sp_param(_vParam2)) ] {
        set lval [lindex $sp_symb($sp_param(_vParam2)) \
                  [lsearch -exact $sp_vals($sp_param(_vParam2)) $n2]]
        $win insert end [format "%*s | " $sp_max_vallen $lval] 
      } else {
        $win insert end [format "%*.3g | " $sp_max_vallen $n2] 
      }
    } elseif { $sp_param(_vParam1) != "--NONE--" } {
      $win insert end [format "%*s | " $sp_max_vallen " "]
    }
    foreach t $sp_time($vec) { $win insert end [format "%10.3f" $t] }
    $win insert end "\n"
    $win tag add val$i {end - 1 lines linestart} {end -1 lines lineend}
    $win tag configure val$i -foreground [wraplindex $sp_color $i]
    incr i
  }

  # -- add speedup numbers if available
  if { [winfo exists .showsu.g] && [info exists sp_speedup] } {
    $win insert end "\nSpeedup\n"
    $win insert end "=======\n\n"

    $win insert end [format "%*s | " $sp_max_vallen "Number of Processors"]
    foreach v $sp_param(_RunList1) { $win insert end [format "%10.3g" $v] }
    $win insert end [format "\n%*.*s-+-" $sp_max_vallen $sp_max_vallen \
         "------------------------------------------------------------"]
    foreach v $sp_param(_RunList1) { $win insert end "----------" }
    if { $sp_param(_vParam2) != "--NONE--" } {
      if [info exists sp_name($sp_param(_vParam2))] {
        $win insert end [format "\n%*s |\n" $sp_max_vallen \
                         $sp_name($sp_param(_vParam2))]
      } else {
        $win insert end [format "\n%*s |\n" $sp_max_vallen $sp_param(_vParam2)]
      }
    } else {
      $win insert end "\n"
    }

    foreach i [lsort -integer [array names sp_speedup]] {
      if { $sp_param(_vParam2) != "--NONE--" } {
        set n2 [lindex $sp_param(_RunList2) $i]
        if [info exists sp_symb($sp_param(_vParam2)) ] {
          set lval [lindex $sp_symb($sp_param(_vParam2)) \
                    [lsearch -exact $sp_vals($sp_param(_vParam2)) $n2]]
          $win insert end [format "%*s | " $sp_max_vallen $lval] 
        } else {
          $win insert end [format "%*.3g | " $sp_max_vallen $n2] 
        }
      } else {
        $win insert end [format "%*s | " $sp_max_vallen " "]
      }
      foreach t $sp_speedup($i) { $win insert end [format "%10.3f" $t] }
      $win insert end "\n"
      $win tag add val$i {end - 1 lines linestart} {end -1 lines lineend}
    }
  }

  $win configure -state disabled
}

#
# showSpeedup:
#

proc showSpeedup {} {
  global sp_fsname
  global myself

  if [winfo exists .showsu] {
    raise .showsu
  } else {
    toplevel .showsu
    wm title .showsu "Speedup"

    # -- menubar
    frame .showsu.bar -relief raised -borderwidth 2
    menubutton .showsu.bar.b1 -text File -menu .showsu.bar.b1.m1 -underline 0
    menu .showsu.bar.b1.m1
    .showsu.bar.b1.m1 add command -label "Print Graph" -underline 0 \
                      -command "printCanvas .showsu.g speedup"
    .showsu.bar.b1.m1 add separator
    .showsu.bar.b1.m1 add command -label "Exit" -underline 0 \
                      -command "destroy .showsu"

    menubutton .showsu.bar.b3 -text Help -menu .showsu.bar.b3.m1 -underline 0
    menu .showsu.bar.b3.m1
    .showsu.bar.b3.m1 add command -label "on $myself" -underline 3 \
                  -command "xsend tau \[list showHelp $myself 1-$myself 1\]"
    .showsu.bar.b3.m1 add separator
    .showsu.bar.b3.m1 add command -label "on menubar" -underline 3 \
                  -command "xsend tau \[list showHelp $myself 1.1-menu 1\]"
    .showsu.bar.b3.m1 add command -label "on display area" -underline 3 \
                  -command "xsend tau \[list showHelp $myself 1.2-display 1\]"
    .showsu.bar.b3.m1 add command -label "on command buttons" -underline 3 \
                  -command "xsend tau \[list showHelp $myself 1.3-buttons 1\]"
    .showsu.bar.b3.m1 add separator
    .showsu.bar.b3.m1 add command -label "on using help" -underline 3 \
                  -command "xsend tau \[list showHelp general 1-help 1\]"

    pack .showsu.bar.b1 -side left -padx 5
    pack .showsu.bar.b3 -side right -padx 5
    pack .showsu.bar -side top -fill x

    # -- graph canvas
    NewXYGraph .showsu.g 300 300
    pack .showsu.g

    # -- draw speedup curves
    resetSpeedup .showsu.g
    speedup .showsu.g
    legendSpeedup .showsu.g
  }
}

#
# resetSpeedup:
#
#          win:
#

proc resetSpeedup {win} {
  global sp_param sp_sut sp_name

  # -- compute speedup values from time values
  if [info exists sp_speedup] { unset sp_speedup }

  # -- determine legend title
  set lt ""
  if { $sp_param(_vParam2) != "--NONE--" } {
    if [info exists sp_name($sp_param(_vParam2))] {
      set lt "$sp_name($sp_param(_vParam2))"
    } else {
      set lt "$sp_param(_vParam2)"
    }
  }

  # -- setup titels and x axis labels
  ResetGraph .showsu.g
  SetTitles .showsu.g "Speedup" "Number of Processors"
  if { $lt != "" } { SetLegendTitle .showsu.g $lt }
  set min [lindex $sp_param(_RunList1) 0]
  set max [lindex $sp_param(_RunList1) [expr [llength $sp_param(_RunList1)]-1]]
  if { $min == $max } {
    set labs $min
    set min [expr $min - 1]
    set max [expr $max + 1]
  } else {
    set no 5
    set labs [Pretty min max no]
  }
  SetXAxis .showsu.g $min $max
  if { [llength $sp_param(_RunList1)] < 6 } {
    eval AddXLabels .showsu.g $sp_param(_RunList1)
  } else {
    eval AddXLabels .showsu.g $labs
  }
  set sp_sut ""
}

#
# legendSpeedup:
#
#           win:
#

proc legendSpeedup {win} {
  global sp_param sp_color sp_symb sp_vals

  if { $sp_param(_vParam2) != "--NONE--" } {
    set i 0
    foreach n2 $sp_param(_RunList2) {
      if [info exists sp_symb($sp_param(_vParam2)) ] {
        set lval [lindex $sp_symb($sp_param(_vParam2)) \
                  [lsearch -exact $sp_vals($sp_param(_vParam2)) $n2]]
      } else {
        set lval $n2
      }
      AddLegendItem $win $i $lval "line$i" [wraplindex $sp_color $i] "x"
      incr i
    }
  }
}

#
# speedup:
#
#     win:
#

proc speedup {win} {
  global sp_param sp_time sp_color sp_speedup sp_sut

  # -- no results; cannot do anything
  if { ! [info exists sp_time] } return

  # -- speedup only good if we have number of processor results
  if { ($sp_param(_vParam1) != "_NumProc") ||
       ([lindex $sp_param(_RunList1) 0] != 1) } return

  # -- compute speedup values from time values
  if [info exists sp_speedup] { unset sp_speedup }
  set i 0
  set min 1
  set max 1
  foreach vec [lsort [array names sp_time]] {
    set time1 [lindex $sp_time($vec) 0]
    foreach t $sp_time($vec) {
      set s [expr $time1 / $t]
      lappend sp_speedup($i) $s
      if { $s < $min } { set min $s }
      if { $s > $max } { set max $s }
    }
    incr i
  }

  # -- reset y axis labels
  if { $sp_sut != "" } { eval $win delete $sp_sut }
  if { $min == $max } {
    set labs [format "%.3f" $min]
    set min [expr $min - 1]
    set max [expr $max + 1]
  } else {
    set no 5
    set labs [Pretty min max no]
  }
  SetYAxis $win $min $max
  set sp_sut [eval AddYLabels $win $labs]

  # -- draw speedup curves
  foreach i [lsort -integer [array names sp_speedup]] {
    set l [expr [llength $sp_speedup($i)] - 1]
    lappend sp_sut [AddLine $win [wraplindex $sp_color $i] "PL" \
      [lrange $sp_param(_RunList1) 0 $l] $sp_speedup($i) "x" "line$i"]
  }
  update
}

#
# viewParam: ExtraP parameter file viewer
#
#       win: window id for viewer
#

proc viewParam {win} {
  global sp_title
  global sp_param sp_curr_param sp_frac

  if [winfo exists $win] {
    raise $win
  } else {
    toplevel $win
    wm title $win "ExtraP Parameter Viewer"

    # -- group area
    frame $win.mod1
    button $win.mod1.b0 -text "General" -command "group0 $win" -width 20
    button $win.mod1.b1 -text "Barrier" -command "group1 $win" -width 20
    button $win.mod1.b2 -text "Processor" -command "group2 $win" -width 20
    pack $win.mod1.b0 $win.mod1.b1 $win.mod1.b2 \
         -side left -ipadx 5 -padx 5 
    frame $win.mod2
    button $win.mod2.b3 -text "Runtime System" -command "group3 $win" \
           -width 20
    button $win.mod2.b4 -text "Interconnect Network" -command "group4 $win" \
           -width 20
    button $win.mod2.b5 -text "Network Interface" -command "group5 $win" \
           -width 20
    pack $win.mod2.b3 $win.mod2.b4 $win.mod2.b5 \
         -side left -ipadx 5 -padx 5
    pack $win.mod1 $win.mod2 -side top -anchor w -pady 5

    # -- separator line
    frame $win.line1 -borderwidth 1 -relief sunken
    pack $win.line1 -side top -fill x -expand 1 -ipady 1

    # -- display area
    label $win.title -textvariable sp_title -relief groove -borderwidth 3
    pack $win.title -side top -pady 10
    frame $win.grp
    group5 $win
    group4 $win
    group3 $win
    group2 $win
    group1 $win
    group0 $win
    pack $win.grp -side top

    # -- separator line
    frame $win.line2 -borderwidth 1 -relief sunken
    pack $win.line2 -side top -fill x -expand 1 -ipady 1

    # -- button area
    frame $win.but
    scale $win.but.s -from 0 -to 10000 -length 300 -showvalue no \
          -orient h -command setNewVal
    pack $win.but.s -side top -pady 5 -fill x

    button $win.but.b0 -text "-100" -width 5 -command "setVal $win -100"
    button $win.but.b1 -text "-10"  -width 5 -command "setVal $win -10"
    button $win.but.b2 -text "-1"   -width 5 -command "setVal $win -1"
    button $win.but.b3 -text "+1"   -width 5 -command "setVal $win 1"
    button $win.but.b4 -text "+10"  -width 5 -command "setVal $win 10"
    button $win.but.b5 -text "+100" -width 5 -command "setVal $win 100"
    pack $win.but.b0 $win.but.b1 $win.but.b2 \
         $win.but.b3 $win.but.b4 $win.but.b5 \
         -side left -padx 2
    pack $win.but -side top -pady 5

    # -- separator line
    frame $win.line3 -borderwidth 1 -relief sunken
    pack $win.line3 -side top -fill x -expand 1 -ipady 1

    # -- command buttons
    frame $win.com
    button $win.com.b1 -text "Reset" -command resetParam
    button $win.com.b2 -text "Load" -command "loadParam Parameter *.par"
    button $win.com.b3 -text "Save" -command saveParam
    button $win.com.b4 -text "Close" -command "destroy $win"
    pack $win.com.b1 $win.com.b2 $win.com.b3 \
         -side left -ipadx 5 -padx 5 -pady 10
    pack $win.com.b4 -side right -ipadx 5 -padx 5 -pady 10
    pack $win.com -side top -fill x
  }
}

#
# compileProg: compile trace-generating executable with cosy
#

proc compileProg {} {
  global sp_param
  global myself

  # -- disable buttons; so only one command can be active at a time
  .$myself.sp1.b1 configure -state disabled
  .$myself.sp1.b3 configure -state disabled

  # -- launch cosy if not yet running
  launch cosy .cosy -waitfor
  xsend cosy "set cosy_var(reset) 0"

  # -- do a "make realclean" to get sure
  # -- then initiate the compiler
  xsend cosy "doExec {make realclean}"
  xsend cosy "doBuild trc-$sp_param(_TrcRuntimeSystem)-ms 1"

  # -- cleanup
  CA
}

#
# computeRunList:
#

proc computeRunList {n} {
  global sp_var sp_param sp_vals

  if { $sp_param(_vParam$n) == "--NONE--"} {
    set sp_param(_RunList$n) 1
  } elseif { ! [info exists sp_var($n,mode)] } {
    return OK
  } elseif { $sp_var($n,mode) == "mult" } {
    set sp_param(_RunList$n) ""

    if { $sp_var($n,mval) == "" }  {
      showError "Parameter $n: No multiple value."; return NOT_OK
    }
    if { $sp_var($n,mfrom) == "" } {
      showError "Parameter $n: No multiple start value."; return NOT_OK
    }
    if { $sp_var($n,mto) == "" }   {
      showError "Parameter $n: No multiple end value."; return NOT_OK
    }

    set hasOne 0
    for { set i $sp_var($n,mfrom) } \
        { $i <= $sp_var($n,mto) }   \
        { set i [expr $i + $sp_var($n,mval)] } {
      if { $i == 1 } { set hasOne 1 }
      lappend sp_param(_RunList$n) "$i"
    }
    if { $sp_var($n,mone) && ! $hasOne } {
       lappend sp_param(_RunList$n) "1"
       set sp_param(_RunList$n) [lsort -real $sp_param(_RunList$n)]
    }
  } elseif { $sp_var($n,mode) == "pow" } {
    set sp_param(_RunList$n) ""

    if { $sp_var($n,pval) == "" }  {
      showError "Parameter $n: No power value."; return NOT_OK
    }
    if { $sp_var($n,pfrom) == "" } {
      showError "Parameter $n: No power start value."; return NOT_OK
    }
    if { $sp_var($n,pto) == "" }   {
      showError "Parameter $n: No power end value."; return NOT_OK
    }

    for { set i $sp_var($n,pfrom) } \
        { $i <= $sp_var($n,pto) }   \
        { set i [expr $i * $sp_var($n,pval)] } {
      lappend sp_param(_RunList$n) "$i"
    }
  } elseif { $sp_var($n,mode) == "rand" } {
    set sp_param(_RunList$n) $sp_var($n,seq)
  } elseif { $sp_var($n,mode) == "dis" } {
    set sp_param(_RunList$n) ""

    set i 0
    foreach v $sp_vals($sp_var($n,param)) {
      if { $sp_var($n,dis,$i) != {} } {
        if { $sp_var($n,dis,$i) != $v } { puts "NNNNNN!!" }
        lappend sp_param(_RunList$n) $v
      }
      incr i
    }
  } else {
    puts stderr "speedy: unknown mode $sp_var($n,mode) for parameter $n"
    exit
  }
  return OK
}

#
# generateTraces: generate event traces and crt files
#
#          tname: event trace basename (without .trc suffix)
#        numproc: number of processors
#

proc generateTraces {tname numproc} {
  global sp_param
  global depfile BINDIR

  # -- generate trace only if not yet existing (why repeat the work?)
  if { ! [file exists ${tname}.trc] } {
    set com "trc-$sp_param(_TrcRuntimeSystem)-ms"
    append com " -pcxx_NUMPROC $numproc"
    append com " -pcxx_EVENTCLASS B+R+Ti+Tr+$sp_param(_ExperimentMode)"
    append com " -pcxx_TRACEFILE ${tname}.trc"
    if { [xsend cosy "doExec \"$com\""] } { return NOT_OK }

    if { $depfile(host) == "localhost" } {
      set com "$BINDIR/trc2crt"
    } else {
      set com "$depfile(root)/bin/$depfile(arch)/trc2crt"
    }
    if { $sp_param(_UseElemSize) } { append com " -e" }
    if { $sp_param(_TraceOverhead) } {
      append com " -t $sp_param(_TraceOverhead)"
    }
    append com " -i $tname -o ${tname}. -s $tname"
    if { [xsend cosy "doExec \"$com\""] } { return NOT_OK }
  }
  return OK
}

#
# runExperiment: execute experiment according to settings and parameters
#

proc runExperiment {} {
  global myself
  global sp_var sp_param sp_name sp_time sp_color sp_vals sp_symb
  global depfile BINDIR REMSH

  # -- disable buttons; so only one command can be active at a time
  .$myself.sp1.b1 configure -state disabled
  .$myself.sp1.b3 configure -state disabled

  set yax ""
  if [info exists sp_time] { unset sp_time }

  # -- compute values for which to run the experiment from settings
  if { [computeRunList 1] == "NOT_OK" } { CA; return }
  if { [computeRunList 2] == "NOT_OK" } { CA; return }

  # -- determine titles
  set t ""
  set lt ""
  if { $sp_param(_vParam1) != "--NONE--" } {
    if [info exists sp_name($sp_param(_vParam1))] {
      set t "$sp_name($sp_param(_vParam1))"
    } else {
      set t $sp_param(_vParam1)
    }
  }
  if { $sp_param(_vParam2) != "--NONE--" } {
    if { $sp_param(_vParam1) != "--NONE--" } {
      if { $sp_param(_vParam1) == $sp_param(_vParam2) } {
        showError "varying parameters equal."
        CA; return
      }

      # setup legend
      if [info exists sp_name($sp_param(_vParam2))] {
        set lt "$sp_name($sp_param(_vParam2))"
      } else {
        set lt "$sp_param(_vParam2)"
      }
    } else {
      showError "varying parameter 1 not set."
      CA; return
    }
  }

  # -- generate generic basename
  set i 0
  while { [file exists "sp$i.desc"] } { incr i }
  set bname "sp$i"
  saveParam ${bname}

  # -- setup result display
  ResetGraph .$myself.sp6
  SetTitles .$myself.sp6 "Execution Time \[s\]" $t
  if { $lt != "" } { SetLegendTitle .$myself.sp6 $lt }
  set min [lindex $sp_param(_RunList1) 0]
  set max [lindex $sp_param(_RunList1) [expr [llength $sp_param(_RunList1)]-1]]
  if { $min == $max } {
    set labs $min
    set min [expr $min - 1]
    set max [expr $max + 1]
  } else {
    set no 5
    set labs [Pretty min max no]
  }
  SetXAxis .$myself.sp6 $min $max
  if { $t != ""} {
    if { [llength $sp_param(_RunList1)] < 6 } {
      eval AddXLabels .$myself.sp6 $sp_param(_RunList1)
    } else {
      eval AddXLabels .$myself.sp6 $labs
    }
  }
  if [winfo exists .showsu] { resetSpeedup .showsu.g }

  # -- launch cosy if not yet running
  launch cosy .cosy -waitfor
  xsend cosy "set cosy_var(reset) 0"

  # -- generate single trace if varying parameter is NOT _NumProc
  # -- and run trc2crt on the generated trace
  if { $sp_param(_vParam1) != "_NumProc" } {
    set tname ${bname}-$sp_param(_NumProc)
    if { [generateTraces $tname $sp_param(_NumProc)] == "NOT_OK" } {
      CA $bname; return
    }
  }

  # -- run experiments
  set ymin NA
  set ymax NA
  set i 0
  foreach n2 $sp_param(_RunList2) {
    # -- setup legend if necessary
    if { $lt != "" } {
      if [info exists sp_symb($sp_param(_vParam2)) ] {
        set lval [lindex $sp_symb($sp_param(_vParam2)) \
                  [lsearch -exact $sp_vals($sp_param(_vParam2)) $n2]]
      } else {
        set lval $n2
      }
      AddLegendItem .$myself.sp6 $i $lval "line$i" [wraplindex $sp_color $i] "x"
      if [winfo exists .showsu] {
        AddLegendItem .showsu.g $i $lval "line$i" [wraplindex $sp_color $i] "x"
      }
    }

    # -- generate traces if varying parameter1 IS _NumProc
    # -- and run trc2crt on the generated traces
    if { $sp_param(_vParam2) == "_NumProc" } {
      set tname ${bname}-$n2
      if { [generateTraces $tname $n2] == "NOT_OK" } { CA $bname; return }
    }

    set j 0
    foreach n1 $sp_param(_RunList1) {
      # -- generate traces if varying parameter1 IS _NumProc
      # -- and run trc2crt on the generated traces
      if { $sp_param(_vParam1) == "_NumProc" } {
        set tname ${bname}-$n1
        if { [generateTraces $tname $n1] == "NOT_OK" } { CA $bname; return }
      }

      # -- generate parameter file and run ExtraP
      set ename ${bname}.$i.$j
      if { [generateParam $ename $sp_param(_vParam1) $n1\
                                 $sp_param(_vParam2) $n2 ] == "NOT_OK" } {
        CA $bname; return
      }

      if { $depfile(host) == "localhost" } {
        set com "$BINDIR/XtraSim"
      } else {
        set com "$depfile(root)/bin/$depfile(arch)/XtraSim"
      }
      append com " -i ${tname}. -n -p $ename -s $ename"
      if { [xsend cosy "doExec \"$com\""] } { CA $bname; return }

      # -- read result of extrapolation (Timer1Time)
      if { $sp_param(_ExperimentMode) == "K" } {
        set timestr "Timer1Time:"
      } else {
        set timestr "SelectedTime:"
      }

      if { $depfile(host) == "localhost" } {
        set readcom ${ename}.stat
      } else {
        set readcom "|$REMSH $depfile(host) \
                    -n \"cd $depfile(dir); cat ${ename}.stat\""
      }
      set in [open $readcom "r"]
      set line ""
      while {[gets $in line] >= 0} {
        set field ""
        set value ""
        scan $line "%s %s" field value
        if { $field == $timestr } {
          break;
        }
        set line ""
      }
      if [catch {close $in} errmsg] {
        showError "$path: `$errmsg'."
        { CA $bname; return }
      }

      # -- store and display result
      lappend sp_time($bname,$i) $value
      if { $ymin == "NA" } {
        set ymin $value
        set ymax $value
      } else {
        if { $value < $ymin } { set ymin $value }
        if { $value > $ymax } { set ymax $value }
      }
      set min $ymin
      set max $ymax
      if { $yax != "" } { eval .$myself.sp6 delete $yax $ydt }
      if { $ymin == $ymax } {
        set min [expr $min - 1]
        set max [expr $max + 1]
        set labs [format "%.3f" $value]
      } else {
        set no 5
        set labs [Pretty min max no]
      }

      SetYAxis .$myself.sp6 $min $max
      set yax [eval AddYLabels .$myself.sp6 $labs]
      set ydt ""
      for {set k 0} {$k <= $i} {incr k} {
        if { $k == $i } {
          lappend ydt [AddLine .$myself.sp6 [wraplindex $sp_color $k] "PL" \
            [lrange $sp_param(_RunList1) 0 $j] $sp_time($bname,$k) "x" "line$k"]
        } else {
          lappend ydt [AddLine .$myself.sp6 [wraplindex $sp_color $k] "PL" \
            $sp_param(_RunList1) $sp_time($bname,$k) "x" "line$k"]
        }
      }
      update
      if [winfo exists .showsu.g] { speedup .showsu.g }
      if [winfo exists .showval.t1] { formatValues .showval.t1 }
      incr j
    }
    incr i
  }
  CA $bname
}

#
#   CA: cleanup procedure for commands
#
# file:
#

proc CA {{file {}}} {
  global sp_param
  global myself

  # -- get rid of temporary files if specified
  if { $file != "" && ! $sp_param(_KeepTemp) } {
    xsend cosy "doExec \"rm ${file}*.trc ${file}*.crt\
                ${file}*.par ${file}*.stat ${file}*.info\""
  }

  # -- activate command buttons again
  .$myself.sp1.b1 configure -state normal
  .$myself.sp1.b3 configure -state normal
}

#
# generateParam: generate Parameter file
#
#          file: filename of parameter file
#         param: name of varying parameter
#         value: value for varying parameter
#

proc generateParam {file param1 value1 param2 value2} {
  global sp_param
  global depfile REMSH

  if { $depfile(host) == "localhost" } {
    set writecom $file.par
  } else {
    set writecom "|$REMSH $depfile(host) \"cd $depfile(dir); cat > $file.par\""
  }
  set out [open $writecom "w"]
  puts $out "# created by speedy"

  foreach n [array names sp_param] {
    if { $n == $param1 } {
      set val $value1
    } elseif { $n == $param2 } {
      set val $value2
    } else {
      set val $sp_param($n)
    }

    if { [regexp "^_" $n] } {
      puts $out "#$n $val"
    } else {
      puts $out "$n $val"
    }
  }

  if [catch {close $out} errmsg] {
    showError "$path: `$errmsg'."
    return NOT_OK
  }
  return OK
}

#
# setVal: update numerical parameter after increment button press
#
#   incr: increment value
#

proc setVal {win incr} {
  global sp_param sp_curr_param sp_frac

  # -- check if a parameter field is selected
  if { "$sp_curr_param" != "" } {

    # -- if parameter field is empty (perhaps user deleted everything)
    # -- set it to 0
    if { "$sp_param($sp_curr_param)" == "" } {
      set sp_param($sp_curr_param) 0
    }

    # -- update parameter value and determine fractional value
    set sp_param($sp_curr_param) [expr $sp_param($sp_curr_param) + $incr]
    set sp_frac [expr $sp_param($sp_curr_param)-int($sp_param($sp_curr_param))]

    # -- update scale
    $win.but.s set [expr int($sp_param($sp_curr_param))]
  }
}

#
# setNewVal: update numerical parameter after scale movement
#
#    newval: integer part of new value (scales can only handle integer values)
#

proc setNewVal {newval} {
  global sp_param sp_curr_param sp_frac

  # -- update parameter value using the fractional value stored in global var
  set sp_param($sp_curr_param) [expr $newval + $sp_frac]
}

#
# loadParam: read parameter file
#
#      what: type of file (used for title display)
#   pattern: file pattern describing this file
#

proc loadParam {what pattern} {
  global sp_param

  set path [getFile "Select $what File" $pattern]
  if { $path == "" } return

  if { ! [file readable $path] } {
      showError "Cannot open file: `$path'."
      return
  }

  set lnum 1
  set in [open $path "r"]
  while {[gets $in line] >= 0} {
    if { ("$line" != "") && ([regexp {^#[^_]} $line] == 0) } {
      if { [scan $line "%s %s" field value] != 2 } {
        showError "$path:$lnum: invalid line `$line'"
      } else {
        regsub "$field " $line {} line
        regsub {^#} $field {} field
        if { ! [info exists sp_param($field)] } {
          if [regexp {^[^_]} $field] {
            showError "$path:$lnum: unknown parameter `$field'"
          }
        } else {
          set sp_param($field) "$line"
        }
      }
    }
    incr lnum
    set line ""
  }

  if [catch {close $in} errmsg] {
    showError "$path: `$errmsg'."
    return
  }
}

#
# saveParam: write parameter file
#

proc saveParam {{file {}}} {
  global sp_param sp_time

  if { $file == {} } {
    set path [getFile "Select Parameter File" *.par]
    if { $path == "" } return
    if { [file exists $path] } {
      if [tk_dialog .e "WARNING" "File exists. Override?" \
          warning 0 override cancel] {
        return
      }
    }
  } else {
    set path "$file.desc"
  }

  set in [open $path "w"]
  puts $in "# created by speedy"

  foreach n [array names sp_param] {
    if { [regexp "^_" $n] } {
      puts $in "#$n $sp_param($n)"
    } else {
      puts $in "$n $sp_param($n)"
    }
  }

  if [info exists sp_time($file)] {
    puts $in "#_$file $sp_time($file)"
  }

  if [catch {close $in} errmsg] {
    showError "$path: `$errmsg'."
    return
  }
}

#
#   initvParam: initialize varying parameter menu
#
#          num:
# dis_param_ok:
#

proc initvParam {num dis_param_ok} {
  global sp_symb sp_vals sp_param sp_name

  foreach n [array names sp_param] {
    if { ![regexp "^_" $n] && (![info exists sp_symb($n)] || $dis_param_ok) } {
      if [info exists sp_name($n)] {
        lappend l [list $sp_name($n) $n]
      } else {
        lappend l [list $n $n]
      }
    }
  }
  lappend l [list "Number of Processors" _NumProc ]
  set l [linsert [lsort $l] 0 [list --NONE-- --NONE--]]

  set len [llength $l]
  for {set i 0} {$i < $len} {incr i} {
    lappend sp_symb(_vParam$num) [lindex [lindex $l $i] 0]
    lappend sp_vals(_vParam$num) [lindex [lindex $l $i] 1]
  }
}

#
# launchCOSY {}
#

proc launchCOSY {} {
  global depfile
  global BINDIR
  global myname

  if { $depfile(host) == "localhost" } {
    set path "$depfile(file)"
  } else {
    set path "$depfile(host):$depfile(path)"
  }

  set avail [winfo interps]
  if { [ lsearch -exact $avail cosy$myname(ext)] == -1 } {
    exec $BINDIR/cosy $path $depfile(root) $depfile(arch) &

    # -- wait for cosy to startup
    while { [ lsearch -exact $avail cosy$myname(ext)] == -1 } {
      set avail [winfo interps]
    }
  }

  # -- after giving cosy time to startup
  # -- set cosy into mode which keeps all output
  xsend cosy "set cosy_var(reset) 0"
}


proc Tool_AcceptChanges {progfiles flag} {
    global myself depfile \
	    showFile selectBox

    switch $flag {

        d {
        }


        a {
        }


        u {
        }
	
	e {
	    loadProfile
	}

	p {
	    set status [PM_Status]
	    if {$status != "NO_PROJECT"} {
		set depfile(project) [lindex $pm_status 0]
		set depfile(host)    [lindex $pm_status 1]
		set depfile(arch)    [lindex $pm_status 2]
		set depfile(root)    [lindex $pm_status 3]
		set depfile(dir)     [lindex $pm_status 4]
	    }
	}
    }
}



#  initSpeedy - Speedy only works with simple pc++ projects, ie, pc++ projects
#               with only one program file.
proc initSpeedy {} {
    global speedy_progfile

    set files [PM_GetFiles]
    if {[llength $files] > 1 || [llength $files] == 0} {
	showError "Speedy is only compatible with single-file pC++ projects."
	exit
    }
    if {[file extension [lindex $files 0]] != ".pc"} {
	showError "Speedy is only compatible with single-file pC++ projects."
	exit
    }

    set speedy_progfile [lindex $files 0]
}


# ------------
# -- main code
# ------------

if {$argc == 0} {
  set parg [pwd]
} elseif {$argc == 1} {
    set parg [lindex $argv 0]
    if {[file extension $parg] != ".pmf"} {
	set parg "$parg.pmf"
    }
} else {
  puts stderr "usage: $myself \[\[host:\]projFile \| \[host:\]directory\]"
  exit
}

# Init the project manager (taud)
launchTauDaemon -waitfor
PM_AddTool $myself
PM_AddGlobalSelect $myself {global_selectFuncTag}

# Initialize the project
# Coordinate w/ PM
set pm_status [PM_Status]
if {[lindex $pm_status 0] == "NO_PROJECT"} {
    # Open or create a project
    set colon [string first ":" $parg]
    if {$colon > 0} {
        set hostarg [string range $parg 0 [expr $colon - 1]]
        set patharg [string range $parg [expr $colon + 1] end]
    } else {
        set hostarg localhost
        set patharg $parg
    }

    set projfile [PM_OpenProject $patharg $hostarg]
    if {$projfile == "NO_PROJECT"} {
        showError "No project opened!"
        exit
    }
    set pm_status [PM_Status]
}

set depfile(project) [lindex $pm_status 0]
set depfile(host)    [lindex $pm_status 1]
set depfile(arch)    [lindex $pm_status 2]
set depfile(root)    [lindex $pm_status 3]
set depfile(dir)     [lindex $pm_status 4]

# Check for language-tool compatibility
if {![Lang_CheckCompatibility [PM_GetProjectLangs] $myself]} {
    showError "$myself is not compatible with the project language(s)."
    exit
}

# Tool Init
initSpeedy

# -- create new toplevel window
resetParam
createWindow
launchTAU

removeMessage
