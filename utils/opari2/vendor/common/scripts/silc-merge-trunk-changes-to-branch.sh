#!/bin/sh

# Assume you are in the toplevel scorep directory of a branch created from trunk

top_dir=`pwd`
SVN_MERGE="svn merge --non-interactive"
SVN_COMMIT="svn ci -m "
dirs="$top_dir/vendor/common $top_dir/vendor/otf2/vendor/utility $top_dir/vendor/otf2 $top_dir/vendor/opari2 $top_dir"

try_commit()
{
    conflicts="$(svn stat | sed -ne 's/^\(C.\|.C\|......C\)......//g p')"

    if [ -z "$conflicts" ]; then
        for i in $dirs; do
            echo "Committing in '$i':"
            cd $i || exit 2
            $SVN_COMMIT "Merged changes from trunk to branch." || {
                echo "Commit failed in '$i'!"
                exit 1
            }
        done
        svn up
        echo "Merge completed. Your working copy is up to date. Please re-bootstrap, re-configure and run make && make check && make distcheck. Have fun."
    else
        echo "Following files are in conflicted state:"
        svn stat | grep '^\(C.\|.C\|......C\)'
        echo "Please resolve conflicts and call $0 --commit"
    fi
}

if [ -n "$1" ]; then
    if [ "--commit" = "$1" ]; then
        try_commit
        exit
    else
        echo "Unknown parameter \"$1\", use without parameter or with \"--commit\"."
        exit 1
    fi
fi


# check for up to date working copy
up_to_date=`svn stat -q | grep -v "^Performing"`
if [ ! -z "$up_to_date" ]; then
    echo "Working copy has modifications, please commit before merging:"
    svn stat -q
    exit
fi

# merge, don't check errors
for i in $dirs; do
    echo "Merging in '$i':"
    cd $i || exit 2
    $SVN_MERGE ^/trunk
done

try_commit
