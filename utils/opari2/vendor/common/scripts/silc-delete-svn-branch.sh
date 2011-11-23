#!/bin/sh

# Exit on errors
set -e

if [ $# -ne 1 ]; then
    echo "Please provide the name of the branch (i.e. branches/<name>) you want to delete as first and only parameter."
    exit
fi

SVN_ROOTS="common utility otf2 opari2 silc"

# exit if the branches you try to delete do not exist
for root in $SVN_ROOTS; do
    svn ls https://silc.zih.tu-dresden.de/svn/${root}-root/branches/${1} > /dev/null
    if [ $? -ne 0 ]; then
        echo "https://silc.zih.tu-dresden.de/svn/${root}-root/branches/${1} does not exist"
        exit
    fi
done

# delete branches
for root in $SVN_ROOTS; do
    svn delete https://silc.zih.tu-dresden.de/svn/${root}-root/branches/${1} -m "Deleting branch ${1}."
    echo "deleting https://silc.zih.tu-dresden.de/svn/${root}-root/branches/${1}"
done
