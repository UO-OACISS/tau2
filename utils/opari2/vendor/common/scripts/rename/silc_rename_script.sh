#!/bin/sh

BASE=`dirname $0`

# include $DOMAIN
. $BASE/silc_global.sh

BRANCH=$1
TARGET=$2

if [ $# -lt 1 ]; then
    echo ""
    echo "Usage: `basename $0` src dest"
    echo ""
    echo " src   svn branch or trunk to checkout and rename"
    echo "           one of: common-root/trunk/ utility-root/trunk/ "
    echo "           otf2-root/trunk/ opari2-root/trunk/ silc-root/trunk/"
    echo " dest  local directory where the checkout is placed"
    echo ""
    echo "Example:"
    echo ""
    echo " `basename $0` trunk silc-trunk"
    echo ""
    exit 0
fi

if test -z $BRANCH; then
    echo "WARNING: no source (branch/trunk) given"
    exit 1
fi

if test -z $TARGET; then
    echo "WARNING: no target given"
    exit 1
fi

if test -e $TARGET; then 
    echo "WARNING: target dir "$TARGET" exists"; 
#    exit 1;
else
    echo "get current SVN from $BRANCH"
    echo svn checkout -q https://$DOMAIN/svn/$BRANCH $TARGET
    svn checkout -q https://$DOMAIN/svn/$BRANCH $TARGET
fi


MV='echo svn mv'
MV='svn mv'

find $TARGET -type d -iname "*silc*"

echo "1) rename dirs"


find $TARGET"/" -type d -name "*silc*" -exec sh $BASE/silc_rename_file.sh '{}' \;

## $MV $TARGET/include/silc $TARGET/include/scorep 
## $MV $TARGET/vendor/otf2/vendor/utility/include/silc_utility $TARGET/vendor/otf2/vendor/utility/include/scorep_utility


echo "2) rename files"


PATTERNS='"*silc*.h"
          "*silc*.c"
          "*silc*.cpp"
          "*silc*.m4"
          "*silc*.sh"
          "silc*.s"
          "silc*.conf"
          "*SILC*.h"
          "*SILC*.c"
          "*SILC*.cpp"
          "SILC*.inc"
          "SILC*.f"
          "SILC*.tmpl"
          "SILC*.xml"
          "SILC*.w"
          "SILC*.txt"
          "SILC*.png"
          "SILC*.eps"
          "SILC*.hpp"
          "Silc*.hpp"
          "silc*.hpp"
          "Silc*.cpp"'


for I in $PATTERNS; do

   I=`echo "$I" | tr -d "\""`
   find $TARGET"/" -type f -name "$I" -exec sh $BASE/silc_rename_file.sh '{}' \;

done


echo "3) change files"


PATTERNS2='"*.h"
           "*.c"
           "*.hpp"
           "*.cpp"
           "*.m4"
           "*.s"
           "*.conf"
           "*.cfg"
           "*.dox"
           "*.inc"
           "*.f"
           "*.tmpl"
           "*.xml"
           "*.w"
           "*.txt"
           "*.sh"
           "*.in"
           "*.out"
           "*.template"
           "*.l"
           "header"
           "configure.ac"
           "*.ac"
           "*.am"
           "Makefile"
           "Makefile_test"
           "Makefile_pgi"
           "*.awk"
           "VERSION"'


for I in $PATTERNS2; do

   I=`echo "$I" | tr -d "\""`
   find $TARGET"/" -type f -name "$I" -exec sh $BASE/silc_change_file.sh '{}' \;

done


echo "4) correct buffer overflows"

SRC=$TARGET"/src/measurement/scorep_runtime_management.c"
TMP=$SRC".bak"

if test -f $SRC; then

    echo $SRC
    cp $SRC $TMP
    sed -e "s/#define dir_name_size  32/#define dir_name_size  34/" -e "s/scorep_experiment_dir_name, 21/scorep_experiment_dir_name, 23/" $TMP >$SRC
fi


SRC=$TARGET"/src/measurement/SCOREP_Config.c"
TMP=$SRC".bak"

if test -f $SRC; then

    echo $SRC
    cp $SRC $TMP
    sed -e 's/char environment_variable_name\[ 7 + 2 \* 32 \]/char environment_variable_name\[ 9 + 2 \* 32 \]/' $TMP >$SRC
fi

echo "5) correct fortran test program"


SRC=$TARGET"/test/adapters/user/Fortran/user_f_test.f"
TMP=$SRC".bak"

if test -f $SRC; then

    echo $SRC
    cp $SRC $TMP
sed -e 's/call SCOREP_User_RegionBeginF( scorepufh, "ScorepTest", 1, "test.f", 19)/call SCOREP_User_RegionBeginF( scorepufh, "ScorepTest", 1,\
     +                               "test.f", 19)/' \
    -e 's/call SCOREP_User_RegionBeginF( region1, "Region1", 0, "test.f", 23)/call SCOREP_User_RegionBeginF( region1, "Region1", 0,\
     +                               "test.f", 23)/' \
    -e 's/call SCOREP_User_InitMetricF( local3, "local3", "s", 1, 0, local1 )/call SCOREP_User_InitMetricF( local3, "local3", "s", 1, 0,\
     +                              local1 )/' $TMP >$SRC
fi

echo "done"



