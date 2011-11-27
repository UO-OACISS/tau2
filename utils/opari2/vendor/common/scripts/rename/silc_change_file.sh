#!/bin/sh


SRC=$1
TMP=$SRC".old"

echo $SRC | grep -e 'install-pre-commit\.sh'

if [ $? -eq 0 ]; then

    exit 0

fi

echo $1
cp $SRC $TMP
sed -e "s/silc/scorep/g" -e "s/SILC/SCOREP/g" -e "s/Silc/Scorep/g" $TMP >$SRC

