

#
# checklist - creates a scrolled checkbutton list.
#
#         w - the widget's pathname
#   buttons - a list of buttons.  Each button is described as a list
#             containing the button text, and optionaly, the button's 
#             command and varaible.  For example:
#                { {"Button 1 Text" {puts "Button 1 pressed."} buts(1) } \
#                  {"Button 2 Text" {puts "Button 2 pressed."} buts(2) } }
#             The command is called with the checkbutton name appended
#             to the end of the argument list.
#     width - width of the visible canvas
#    height - height of the visible canvas
#

proc checklist { w butlist {width 200} {height 100} } {

    set num_buttons [llength $butlist]
    frame $w -borderwidth 2 -relief raised
    frame $w.top
    canvas $w.top.canvas \
	    -width $width -height $height \
	    -yscrollcommand "$w.top.yscroll set" \
	    -xscrollcommand "$w.xscroll set" 	    
    scrollbar $w.top.yscroll -command "$w.top.canvas yview"
    pack $w.top.canvas -side left -fill both -expand yes
    pack $w.top.yscroll -side right -fill y -expand no
    scrollbar $w.xscroll -orient horiz -command "$w.top.canvas xview" 

    pack $w.top -side top -fill both -expand yes
    pack $w.xscroll -side bottom -fill x -expand no
    

    set num 0
    set canvas_height 0
    set max_width 0
    foreach button $butlist {
	if {[llength $button] == 3} {
	    # Use all parameters
	    set cmd [lindex $button 1]
	    checkbutton $w.top.canvas.but$num -text [lindex $button 0] \
		    -relief flat -command "$cmd $w.top.canvas.but$num" \
		    -variable [lindex $button 2]
	} elseif {[llength $button] == 2} {
	    # Use command parameter only
	    set cmd [lindex $button 1]
	    checkbutton $w.top.canvas.but$num -text [lindex $button 0] \
		    -relief flat -command "$cmd $w.top.canvas.but$num" \
	} elseif {[llength $button] == 1} {
	    # Use only the name
	    checkbutton $w.top.canvas.but$num -text [lindex $button 0] \
		    -relief flat
	} else {
	    puts "bad checklist button parameters."
	    exit
	}

	set tag [$w.top.canvas create window 0 $canvas_height \
		-anchor nw -window $w.top.canvas.but$num]
	set button_coords [$w.top.canvas bbox $tag]
	if {[expr [lindex $button_coords 2] - [lindex $button_coords 0]] \
		> $max_width} {
	    set max_width [expr [lindex $button_coords 2] - \
		    [lindex $button_coords 0]]
	}
	incr canvas_height \
		[expr [lindex $button_coords 3] - [lindex $button_coords 1]]
	incr num 1
    }

    $w.top.canvas configure -scrollregion [list 0 0 $max_width $canvas_height]
}
