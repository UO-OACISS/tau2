## These traces also work on array routines, but you have to be careful,
## because ALL elements of the array will be traced with the same procedure.
## You can place multiple traces on each variable (ie - int and len 8).
## Mail jhobbs@cs.uoregon.edu with problems/questions.
##
## For those who aren't using Tk4 with these, here is the bell equivalent:
if [string compare [info commands bell] "bell"] {
  proc bell {} { puts -nonewline "\007" }
}

## This first routine would be used for ensuring that an array value
## would always exist.  It would be used with a read trace like so:
## % set default 1
## % set array(0) $default
## % trace variable array r "forceValue $default"
## Now any other accesses to the array will ensure at least the default value.
proc forceValue {default name el op} {
  global $name
  if {$el != ""} { set name "$name\($el)" }
  if ![info exists $name] { set $name $default }
}


proc forceInt {name el op} {
  global $name ${name}_int
  if {$el != ""} {
    set old  "${name}_int\($el)"
    set name "$name\($el)"
  } else { set old "${name}_int" }
  if ![regexp {^[-+]?[0-9]*$} [set $name]] {
    set $name [set $old]
    bell; return
  }
  set $old [set $name]
}

proc forceReal {name el op} {
  global $name ${name}_real
  if {$el != ""} {
    set old  "${name}_real\($el)"
    set name "$name\($el)"
  } else { set old "${name}_real" }
  if ![regexp {^[-+]?[0-9]*\.?[0-9]*([0-9]\.?e[-+]?[0-9]*)?$} [set $name]] {
    set $name [set $old]
    bell; return
  }
  set $old [set $name]
}

proc forceRegexp {regexp name el op} {
  global $name ${name}_regexp
  if {$el != ""} {
    set old  "${name}_regexp\($el)"
    set name "$name\($el)"
  } else { set old "${name}_regexp" }
  if ![regexp "$regexp" [set $name]] {
    set $name [set $old]
    bell; return
  }
  set $old [set $name]
}

proc forceAlpha {name el op} {
  global $name ${name}_alpha
  if {$el != ""} {
    set old  "${name}_alpha\($el)"
    set name "$name\($el)"
  } else { set old "${name}_alpha" }
  if ![regexp {^[a-zA-Z]*$} [set $name]] {
    set $name [set $old]
    bell; return
  }
  set $old [set $name]
}

proc forceLen {len name el op} {
  global $name ${name}_len
  if [string comp $el {}] {
    set old  ${name}_len\($el)
    set name $name\($el)
  } else { set old ${name}_len }
  if {[string length [set $name]] > $len} {
    set $name [set $old]
    bell; return
  }
  set $old [set $name]
}


## Don't execute the example code:
return


## Here is a wish example to use the routines.  Remember that with
## write traces, a valid value must be set for each variable both
## before AND after the trace is established.

## The order must be:
## 1) variable init
## 2) textvariable specification
## 3) set trace
## 4) variable reinit

set a(1) {}; set a(2) {}; set b {}; set c {}; set d {}; set e {}
set maxLen 8
pack [label .la1 -text "Integer 1:"] -anchor w
pack [entry .a1 -textvariable a(1)]
pack [label .la2 -text "Integer 2:"] -anchor w
pack [entry .a2 -textvariable a(2)]
pack [label .lb -text "Real:"] -anchor w
pack [entry .b -textvariable b]
pack [label .lc -text "Alpha:"] -anchor w
pack [entry .c -textvariable c]
pack [label .ld -text "Hex (using forceRegexp):"] -anchor w
pack [entry .d -textvariable d]
pack [label .le -text "Limit to 8:"] -anchor w
pack [entry .e -textvariable e]
trace variable a w forceInt
trace variable b w forceReal
trace variable c w forceAlpha
# This regexp is for hex numbers
trace variable d w {forceRegexp {^(0x)?[0-9a-fA-F]*$}}
trace variable e w "forceLen $maxLen"
set a(1) {}; set a(2) {}; set b {}; set c {}; set d {}; set e {}
