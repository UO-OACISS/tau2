#*********************************************************************#
#*                TAU/pC++/Sage++  Copyright (C) 1994                *#
#*                        Jerry Manic Saftware                       *#
#*  Indiana University  University of Oregon  University of Rennes   *#
#*********************************************************************#

# -- generic stack for tcl
#
# -- stacks are implemented by a "struct"
# -- stack(length) contains the number of elements stored in stack
# -- stack(data)   is a list of the values

#
# initStack: initialize stack
#
#     stack: variable name for new stack
#

proc initStack {stack} {
  upvar $stack s

  set s(length) 0
  set s(data) [list]
}

#
# pushStack: push value onto stack
#
#     stack: variable name for stack
#       val: value to push
#

proc pushStack {stack val} {
  upvar $stack s

  incr s(length)
  lappend s(data) $val
}

#
#  popStack: pop value from stack
#
#     stack: variable name for stack
#

proc popStack {stack} {
  upvar $stack s

  if { $s(length) } {
    set i [incr s(length) -1]
    set r [lindex $s(data) $i]
    set s(data) [lreplace $s(data) $i $i]
    return $r
  } else {
    return ""
  }
}

#
#  topStack: return top value from stack without popping it
#
#     stack: variable name for stack
#

proc topStack {stack} {
  upvar $stack s

  lindex $s(data) [expr $s(length) - 1]
}
