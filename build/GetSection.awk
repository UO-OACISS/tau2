#!/bin/awk -f
#
#USAGE: GetSection.awk filen=<FILENAME> <PROJECTFILE>
#

BEGIN {
  FS=":[\t]*|[\t]"
  in_section=0;
}

/^File:/ {
  if($2 == filen) {
    in_section=1;
  }
}

/^$/ {
  if(in_section == 1) {
    exit 0;
  }
}
  
{
  if(in_section == 1) {
    printf("%s\\n", $0);
  }
}

