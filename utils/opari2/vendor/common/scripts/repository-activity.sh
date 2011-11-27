#!/bin/bash

## 
## This file is part of the Score-P software (http://www.score-p.org)
##
## Copyright (c) 2009-2011, 
##    RWTH Aachen, Germany
##    Gesellschaft fuer numerische Simulation mbH Braunschweig, Germany
##    Technische Universitaet Dresden, Germany
##    University of Oregon, Eugene, USA
##    Forschungszentrum Juelich GmbH, Germany
##    German Research School for Simulation Sciences GmbH, Juelich/Aachen, Germany
##    Technische Universitaet Muenchen, Germany
##
## See the COPYING file in the package base directory for details.
##

## file       repository-activity.sh
##            Generate two wiki-formatted tables containing the activity in
##            all Score-P related repositories als mail them to 
##            c.roessel@fz-juelich.de.
## maintainer Christian Roessel <c.roessel@fz-juelich.de>

mypid="$$"
mydir="/tmp/silc_repo_activity_${mypid}"
mkdir ${mydir}

repo[0]="https://silc.zih.tu-dresden.de/svn/silc-root"
repo[1]="https://silc.zih.tu-dresden.de/svn/otf2-root"
repo[2]="https://silc.zih.tu-dresden.de/svn/opari2-root"
repo[3]="https://silc.zih.tu-dresden.de/svn/utility-root"
repo[4]="https://silc.zih.tu-dresden.de/svn/common-root"
repo[5]="https://silc.zih.tu-dresden.de/svn/applications-root"
repo[6]="https://silc.zih.tu-dresden.de/svn/publications-root"

repo_name[0]="silc"
repo_name[1]="otf2"
repo_name[2]="opari2"
repo_name[3]="utility"
repo_name[4]="common"
repo_name[5]="applications"
repo_name[6]="publications"

count[0]=0 # silc
count[1]=0 # ...
count[2]=0
count[3]=0
count[4]=0
count[5]=0
count[6]=0 # publications

# save the logs in local files for further processing
for index in 0 1 2 3 4 5 6; do
    svn log ${repo[index]} > ${mydir}/${repo_name[index]}
done

developers=`cat ${mydir}/* | grep "^r[0-9]" | grep "|" | awk -F "|" '{a=$2 ;gsub(" ","",a); print a}' | sort -u`

last_month=`date --date='yesterday' +%Y-%m`

rm -f ${mydir}/counts_* ${mydir}/repository_activity*

for dev in $developers; do 
    # ever since counts
    for index in  0 1 2 3 4 5 6; do
        count[index]=0
    done

    sum=0
    for index in  0 1 2 3 4 5 6; do
        count[index]=`grep ${dev} ${mydir}/${repo_name[index]} | wc -l`
        sum=$((sum+${count[index]}))
    done
    echo $sum ${count[0]} ${count[1]} ${count[2]} ${count[3]} ${count[4]} ${count[5]} ${count[6]} ${dev} >> ${mydir}/counts_ever_since

    # last month counts
    for index in  0 1 2 3 4 5 6; do
        count[index]=0
    done

    sum=0
    for index in  0 1 2 3 4 5 6; do
        count[index]=`grep ${dev} ${mydir}/${repo_name[index]} | grep $last_month | wc -l`
        sum=$((sum+${count[index]}))
    done
    echo $sum ${count[0]} ${count[1]} ${count[2]} ${count[3]} ${count[4]} ${count[5]} ${count[6]} ${dev} >> ${mydir}/counts_month
done

echo "Report created by https://silc.zih.tu-dresden.de/svn/common-root/trunk/scripts/repository-activity.sh" > ${mydir}/repository_activity_${last_month}
echo >> ${mydir}/repository_activity_${last_month}
echo "= Repository activity ${last_month} =" >> ${mydir}/repository_activity_${last_month}
echo >> ${mydir}/repository_activity_${last_month}
echo "|| developer      || sum  || silc || otf2 || opari2 || utility || common || applications || publications ||" >> ${mydir}/repository_activity_${last_month}
cat ${mydir}/counts_month | sort -rn | awk '{if ($1 == 0) {next}; printf "|| %-14s || %4s || %4s || %4s || %6s || %7s || %6s || %12s || %12s ||\n", $9, $1, $2, $3, $4, $5, $6, $7, $8 }' >> ${mydir}/repository_activity_${last_month}
echo >> ${mydir}/repository_activity_${last_month}

echo "= Repository activity ever since =" > ${mydir}/repository_activity_ever_since
echo >> ${mydir}/repository_activity_ever_since
echo "|| developer      || sum  || silc || otf2 || opari2 || utility || common || applications || publications ||" >> ${mydir}/repository_activity_ever_since
cat ${mydir}/counts_ever_since | sort -rn | awk '{printf "|| %-14s || %4s || %4s || %4s || %6s || %7s || %6s || %12s || %12s ||\n", $9, $1, $2, $3, $4, $5, $6, $7, $8 }' >> ${mydir}/repository_activity_ever_since
echo >> ${mydir}/repository_activity_ever_since

cat ${mydir}/repository_activity_${last_month} ${mydir}/repository_activity_ever_since >> ${mydir}/repository_activity
mail -s "silc repository activity ${last_month}" c.roessel@fz-juelich.de < ${mydir}/repository_activity

rm -rf ${mydir}
