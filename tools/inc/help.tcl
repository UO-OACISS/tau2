#*********************************************************************#
#*                TAU/pC++/Sage++  Copyright (C) 1994                *#
#*                        Jerry Manic Saftware                       *#
#*  Indiana University  University of Oregon  University of Rennes   *#
#*********************************************************************#

#
# helpSetTagsFormat: set and configure tag in help text window
#
#               tag: name of tag
#              pos1: start position of tag
#              pos2: end position of tag
#            config: options to set for this tag
#

proc helpSetTagsFormat {tag pos1 pos2 config} {
  .help.t1 tag add $tag "$pos1" "$pos2"
  eval .help.t1 tag configure $tag $config
}

#
# helpSetTagsLink: set and configure hypertext links in help text window
#
#             tag: name of tag to use for this link
#            pos1: start position of link
#            pos2: end position of link
#             com: command to execute when user clicks on link
#

proc helpSetTagsLink {tag pos1 pos2 com} {
  .help.t1 tag add $tag "$pos1" "$pos2"
  .help.t1 tag configure $tag -foreground red -underline true
  .help.t1 tag bind $tag <Button-1> "$com"
}

#
# showHelp: help main function, displays on page of help text
#
#     prog: program name
#  section: name of help page
#    isTop: set is page is root of new help tree
#

proc showHelp {prog section {isTop 0}} {
  global TAUDIR
  global helpstack

  # -- read requested help page
  set file "$TAUDIR/help/$prog/$section"
  if { ! [file readable $file] } {
    showError "No help on $prog:$section"
    if [ info exists helpstack(length) ] { popStack helpstack }
    return
  }
  
  if [ winfo exists .help ] {
    # -- help window already displayed; reset help window (delete text)
    .help.t1 configure -state normal
    .help.t1 delete 1.0 end
    if { [set tags [lpick [.help.t1 tag names] {tag*}]] != "" } {
      eval .help.t1 tag delete $tags
    }

    # -- if new help tree, reset stack of help pages
    if { $isTop } { initStack helpstack }
  } else {
    # -- create and setup new help window 
    toplevel .help

    text .help.t1 -width 60 -height 24 -background white -foreground black \
                  -exportselection false -selectbackground white \
                  -selectborderwidth 0
    scrollbar .help.s1 -orient vert -relief sunken -command ".help.t1 yview"
    .help.t1 configure -yscrollcommand ".help.s1 set"

    frame .help.move -borderwidth 2 -relief raised
    frame .help.dummy
    button .help.pv -text "prev"
    button .help.up -text "up"
    button .help.nt -text "next"
    button .help.cl -text "close" -command "destroy .help"
    pack .help.cl -side right -in .help.move -padx 15 -ipadx 3 -ipady 3
    pack .help.pv .help.nt -side left -in .help.dummy -padx 15 -ipadx 3 -ipady 3
    pack .help.up -side left -in .help.move -padx 15 -pady 15 -ipadx 15 -ipady 3
    pack .help.dummy -side left -in .help.move -padx 30 -pady 15

    pack .help.move -side bottom -fill x
    pack .help.s1 -side right  -fill y
    pack .help.t1 -side top -expand yes -fill both

    initStack helpstack
  }

  # -- if there is somthing on the help pages stack, enable "up" button
  # -- and point to that page
  set t [topStack helpstack]
  if { $t == "" } {
    .help.up configure -state disabled
  } else {
    .help.up configure -state normal \
             -command "popStack helpstack; showHelp [lindex $t 0] [lindex $t 1]"
  }

  # -- process text of help file
  set no 0
  set lineno 1
  set in [open $file r]
  while { [gets $in line] >= 0 } {
    if { [regexp "^#" $line] } {
      # -- comment; ignore
    } elseif { [regexp "^@\(\[^\{\]\) ?\(.*$\)" $line dummy com var] } {
      # -- command
      switch -exact $com {
        T  { # -- title
             wm title .help "HELP:     $prog - $var"
           }
        P  { # -- prev button
             if { $var == "" } {
               .help.pv configure -state disabled
             } else {
               .help.pv configure -state normal -command $var
             }
           }
        N  { # -- next button
             if { $var == "" } {
               .help.nt configure -state disabled
             } else {
               .help.nt configure -state normal -command $var
             }
           }
        L  { # -- hypertext link
             set text " [lindex $var 0]"
             regexp -indices {[^ ]} $text pos
             .help.t1 insert end "$text\n"
             .help.t1 tag add tag$no \
                                $lineno.[lindex $pos 0] "$lineno.1 lineend"
             .help.t1 tag configure tag$no -foreground red -underline true
             .help.t1 tag bind tag$no <Button-1> \
                "pushStack helpstack \{[list $prog $section]\}; [lindex $var 1]"
             incr no
             incr lineno
           }
        D  { # -- display format description
             set dformat([lindex $var 0]) [lindex $var 1]
           }
      }
    } elseif { [string first "@\{" $line] != -1 } {
      # -- text with inlined commands
      set todo ""
      while { [regexp "\(^\[^@\]*\)@\{\(\[A-Za-z\]*\) \{\(\[^\}\]*\)\}\(.*$\)" \
                      $line dummy bef c text end] } {
        set line "$bef$text$end"
        set pos1 [string first $text " $line"]
        set pos2 [expr $pos1+[string length $text]]

        if { $c == "L" } {
          # -- text with embedded link
          regexp "^ *\{\(\[^\}\]*\)\}\}\(.*$\)" $end dummy com newend
          set line "$bef$text$newend"
          append todo "helpSetTagsLink \
                      tag$no $lineno.$pos1 \"$lineno.$pos2\" \
                      \"pushStack helpstack \{[list $prog $section]\}; $com\";"
        } else {
          # -- text with embedded format command
          set end [string range $end 1 end]
          set line "$bef$text$end"
          if [ info exists dformat($c) ] {
            append todo "helpSetTagsFormat \
                     tag$no $lineno.$pos1 \"$lineno.$pos2\" \"$dformat($c)\";"
          }
        }
        incr no
      }
      .help.t1 insert end " $line\n"
      incr lineno
      eval $todo
    } else {
      # -- plain text
      .help.t1 insert end " $line\n"
      incr lineno
    }
  }
  close $in
  .help.t1 configure -state disabled
  bind .help.t1 <Control-Shift-Button-1> "showHelp $prog $section"
  update
  if { $isTop } { raise .help }
}
