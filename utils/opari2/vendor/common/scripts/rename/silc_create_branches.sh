#!/bin/sh

BASE=`dirname $0`

# include $DOMAIN
. $BASE/silc_global.sh

TARGET=$1

if [ $# -lt 1 ]; then
    echo ""
    echo "Creates a new branch as a copy of trunk. The branch is"
    echo "called TRY_JSPAZIER_rename# where # stands for the next"
    echo "free number. The branch is checked out and the renaming"
    echo "script is started. If you don't want to create a branch"
    echo "execute the renaming script directly."
    echo ""
    echo "Usage: `basename $0` dest"
    echo ""
    echo " dest  local directory where the checkout is placed"
    echo ""
    exit 0
fi

if test -z $TARGET; then
    echo "WARNING: no target given"
    exit 1
fi

if test -e $TARGET; then 
    echo "WARNING: target dir "$TARGET" exists"; 
    exit 1;
fi

OLD_DIR=`pwd`

REPOS="silc-root
       otf2-root
       opari2-root
       utility-root
       common-root"

BRANCH="TRY_JSPAZIER_rename"

TMP_DIR=/tmp/test-branch
TMP_FILES_DIR=/tmp

id=0

echo "get next id..."

for i in $REPOS; do

    ret=`svn ls "https://$DOMAIN/svn/$i/branches/"`

    for k in $ret; do

        tmp=`echo $k | grep -e "TRY_JSPAZIER_rename[0-9][0-9]*/$"`
    
        if [ $? -eq 0 ]; then

            string=`echo $k | sed -e 's/TRY_JSPAZIER_rename//' -e 's/\///'`
            
            if [ $string -gt $id ]; then

                id=$string

            fi

        fi

    done

done

id=`expr $id + 1`
echo "id= $id"

BRANCH=$BRANCH$id

# creating files to set svn:externals property later
echo "m4 https://$DOMAIN/svn/common-root/branches/$BRANCH/build-config/m4" > "$TMP_FILES_DIR/build-config"
echo "platforms https://$DOMAIN/svn/common-root/branches/$BRANCH/build-config/platforms" >> "$TMP_FILES_DIR/build-config"

echo "beautifier https://$DOMAIN/svn/common-root/branches/$BRANCH/beautifier" > "$TMP_FILES_DIR/tools"

echo "utility https://$DOMAIN/svn/utility-root/branches/$BRANCH" > "$TMP_FILES_DIR/otf2-vendor"

echo "otf2 https://$DOMAIN/svn/otf2-root/branches/$BRANCH" > "$TMP_FILES_DIR/silc-vendor"
echo "opari2 https://$DOMAIN/svn/opari2-root/branches/$BRANCH" >> "$TMP_FILES_DIR/silc-vendor"


echo "create branches $BRANCH"

for REP in $REPOS; do

    if [ -e $TMP_DIR ]; then

        rm -rf $TMP_DIR

    fi

    echo "svn copy \"https://$DOMAIN/svn/$REP/trunk\" \"https://$DOMAIN/svn/$REP/branches/$BRANCH\""
    svn copy "https://$DOMAIN/svn/$REP/trunk" "https://$DOMAIN/svn/$REP/branches/$BRANCH" "-m \"- Created new branch to test rename script\""

    svn -q --ignore-externals checkout "https://$DOMAIN/svn/$REP/branches/$BRANCH" $TMP_DIR

    cd $TMP_DIR

    if [ "$REP" = "common-root" ]; then

        echo "common-root"
        continue

    fi

    svn propset svn:externals -F "$TMP_FILES_DIR/build-config" build-config
    svn propset svn:externals -F "$TMP_FILES_DIR/tools" tools

    if [ "$REP" = "otf2-root" ]; then

        echo "otf2-root"

        svn propset svn:externals -F "$TMP_FILES_DIR/otf2-vendor" vendor
        svn commit -m "\"- set property 'svn:externals' to branch instead of trunk \""

        continue

    fi

    if [ "$REP" = "silc-root" ]; then

        echo "silc-root"

        svn propset svn:externals -F "$TMP_FILES_DIR/silc-vendor" vendor
        svn commit -m "\"- set property 'svn:externals' to branch instead of trunk \""

        continue

    fi

    svn commit -m "\"- set property 'svn:externals' to branch instead of trunk \""

done

cd $OLD_DIR

# clean up tmp dir and files
rm -rf $TMP_DIR
rm -f "$TMP_FILES_DIR/build-config"
rm -f "$TMP_FILES_DIR/tools"
rm -f "$TMP_FILES_DIR/silc-vendor"
rm -f "$TMP_FILES_DIR/otf2-vendor"

sh $BASE/silc_rename_script.sh "branches/$BRANCH" $TARGET


