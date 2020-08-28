#!/bin/bash -e

set -x

T="$(date +%s)"

# Do this in the /tmp directory
basedir=/tmp/`whoami`
# The name of the working tau2 repo
repodir=tau2-for-cleaning
# The main TAU repo
origin=git.nic.uoregon.edu:/gitroot/tau2
# The mirror repo
mirror=git@github.com:UO-OACISS/tau2.git

# Go to the basedir
cd ${basedir}

# Get the BFG repo-cleaner, if necessary
jarurl=http://www.nic.uoregon.edu/~khuck/bfg-1.13.0.jar
if [ ! -f bfg-1.13.0.jar ] ; then
    if command -v wget >/dev/null ; then
        wget ${jarurl}
    else
        curl -O ${jarurl}
    fi
fi

# If the repo already exists, pull and update - it's faster than
# a clean checkout
if [ -d ${repodir} ] ; then
    cd ${repodir}
    git pull --rebase origin master
else
    git clone --branch master ${origin} ${repodir}
    cd ${repodir}
    git remote add mirror ${mirror}
fi

# Run the BFG! See https://github.com/rtyley/bfg-repo-cleaner for details.
before=`du -sh .`
java -jar ${basedir}/bfg-1.13.0.jar --strip-blobs-bigger-than 10M .
git reflog expire --expire=now --all && git gc --prune=now --aggressive
after=`du -sh .`

T="$(($(date +%s)-T))"
echo "Done!  Before: $before, After: $after"
printf "Time to clean TAU: %02d hours %02d minutes %02d seconds.\n" "$((T/3600))" "$((T/60%60))" "$((T%60))"

# Push updates to the mirror
git push -u mirror master

