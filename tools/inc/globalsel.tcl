#*********************************************************************#
#*                TAU/pC++/Sage++  Copyright (C) 1994                *#
#*                        Jerry Manic Saftware                       *#
#*  Indiana University  University of Oregon  University of Rennes   *#
#*********************************************************************#

#
# global_selectFuncTag: select function in all TAU tools which support this
#
#             progfile: depfile containing the tag
#                  tag: function id to select
#

proc global_selectFuncTag {progfile tag} {
  selectFuncTag $progfile $tag
  xsend cagey "selectFuncTag $progfile $tag"
  xsend fancy "selectFuncTag $progfile$tag"
  xsend spiffy "selectFuncTag $progfile $tag"
  xsend racy "selectFuncTag $progfile $tag"
}

#
# global_showFuncTag: show function information
#
#           progfile: depfile containing the tag
#                tag: function id to show
#

proc global_showFuncTag {progfile tag} {
  xsend tau "showFuncTag $progfile $tag"
}

#
# global_selectClassTag: select class in all TAU tools which support this
#
#              progfile: depfile containing the tag
#                   tag: class id to select
#

proc global_selectClassTag {progfile tag} {
  selectClassTag $progfile $tag
  xsend fancy "selectClassTag $progfile $tag"
  xsend spiffy "selectClassTag $progfile $tag"
  xsend classy "selectClassTag $progfile $tag"
}

#
# global_selectLine: select line in source file
#
#              line: line number to select
#              file: file name to select
#               tag: function id to select if line is call site
#          progfile: depfile containing the tag
#

proc global_selectLine {line file {tag -1} {progfile ""}} {
  xsend fancy "selectLine $line $file $tag $progfile"
  xsend cagey "selectLine $line $file $tag $progfile"
  xsend spiffy "selectLine $line $file $tag $progfile"
}



