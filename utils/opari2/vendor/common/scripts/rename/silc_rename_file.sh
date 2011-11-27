#!/bin/sh

MV='echo svn mv'
MV='svn mv'

SRC=$1
DIR=`dirname $1`
FILE=`basename $1`

NEW=`echo $FILE | sed -e "s/silc/scorep/g" -e "s/SILC/SCOREP/g" -e "s/Silc/Scorep/g"`

#echo $MV $SRC $DIR"/"$NEW
$MV $SRC $DIR"/"$NEW

