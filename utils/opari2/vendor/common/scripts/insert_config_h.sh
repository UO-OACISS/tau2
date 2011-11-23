#!/bin/sh

###
# Checks all .c and .cpp files of a directory whether they include
# "<config.h> first. If not the include is inserted.
#
# Author: Johannes Spazier <johannes.spazier@tu-dresden.de>
###

# functions
find_config() {

    # search for first include
    ret=`cat $1 | grep -e "^[[:space:]]*#[[:space:]]*include" | head -n 1`

    # remove all tabs and spaces
    ret=`echo $ret | sed 's/[\t ]\+//g'`

    # check if <config.h> is included
    if [ "$ret" != "#include<config.h>" ]; then

        return 1

    fi

    return 0

}

find_include() {

    mode=$2

    # search for first include
    first_inc=`cat $1 | grep -e "^[[:space:]]*#[[:space:]]*include" | head -n 1`

    # remove all tabs and spaces
    first_inc=`echo $first_inc | sed 's/[\t ]\+//g'`

    if [ -z $first_inc ]; then

        # neither <config.h>, "config.h" nor any include found
        # add <config.h> as first include at beginning of the file

        if [ $mode -eq 0 ]; then

            sed -i '1s/^/#include <config.h>\n\n/' $1

        else

            # dry run
            echo "Neither <config.h>, \"config.h\" nor any include found."
            echo "Insert <config.h> at the beginning of the file."

        fi

    elif [ "$first_inc" = "#include<config.h>" ]; then

        # <config.h> is already the first include
        # just remove possible duplicates

        if [ $mode -eq 0 ]; then

            mv $1 $1.tmp_config
            awk '/^[[:blank:]]*#[[:blank:]]*include[[:blank:]]*<config.h>[[:space:]]*/ \
                 {c++;if(c==2){sub("^[[:blank:]]*#[[:blank:]]*include[[:blank:]]*<config.h>[[:space:]]*","");c=1}}1' $1.tmp_config > $1
            rm $1.tmp_config

        else 

            # dry run
            echo "<config.h> is already the first include."
            echo "All right."

        fi

    elif [ "$first_inc" = "#include\"config.h\"" ]; then

        # "config.h" is the first include
        # just change it to <config.h>

        if [ $mode -eq 0 ]; then

            sed -i 's/^[ \t]*#[ \t]*include[ \t]*"config.h"/#include <config.h>/g' $1

            # just remove possible duplicates
            mv $1 $1.tmp_config
            awk '/^[[:blank:]]*#[[:blank:]]*include[[:blank:]]*<config.h>/ \
                 {c++;if(c==2){sub("^[[:blank:]]*#[[:blank:]]*include[[:blank:]]*<config.h>","");c=1}}1' $1.tmp_config > $1
            rm $1.tmp_config

        else 

            # dry run
            echo "Found \"config.h\" as the first include."
            echo "Replace it by <config.h>."

        fi

    else

        # neither <config.h> nor "config.h" found
        # add <config.h> as first include

        if [ $mode -eq 0 ]; then

            sed -i '0,/^[ \t]*#[ \t]*include/ s//#include <config.h>\n\n#include/' $1

            # just remove possible duplicates
            mv $1 $1.tmp_config
            awk '/^[[:blank:]]*#[[:blank:]]*include[[:blank:]]*<config.h>/ \
                 {c++;if(c==2){sub("^[[:blank:]]*#[[:blank:]]*include[[:blank:]]*<config.h>","");c=1}}1' $1.tmp_config > $1
            rm $1.tmp_config

        else 

            # dry run
            echo "Neither <config.h> nor \"config.h\" found."
            echo "Insert <config.h> as the first include."

        fi

    fi

}

# print helptext
print_help() {

    echo ""
    echo "Usage: `basename $0` [-c|-d] directory/file"
    echo ""
    echo "Options:"
    echo "  -c  check: list files that are invalid"
    echo "      (no <config.h> included)."
    echo "  -d  dry run: print what would be done and"
    echo "      do not modify anything."
    echo ""

}

## main
# parameter handling
ret=`getopt "cd" "$@"`

if [ $? -gt 0 ]; then

    print_help
    exit 1;

fi

set -- `echo $ret`

mode=0

while :
do

    case "$1" in
           -c) mode=1 ;;
           -d) mode=2 ;;
           -h) print_help
               exit 0 ;;
           --) break ;;

    esac
    shift

done

shift

if [ -z $* ]; then

    print_help
    exit 0

fi    

# go through directories/filenames
for i in $*
do

    # traverse directory and search for .c/.cpp files
    for file in $( find $i -regextype posix-egrep -regex '.*\.(c|cpp)$' )
    do

        if [ $mode -eq 1 ]; then

            # check mode
            find_config $file

            if [ $? -gt 0 ]; then

                # #include <config.h> does not exist
                echo "File "$file" not valid."

            fi
        
        else

            echo "Processing file \"$file\""
            find_include $file $mode
    
        fi

    done

done

exit 0

