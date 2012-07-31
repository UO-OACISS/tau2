#! /bin/sh

# this script expects an input that contains at least these lines specifying
# the package version, e.g.
# package.major=0
# package.minor=9
# package.bugfix=0
# package.suffix=

format="${2-}"

major=0
minor=0
bugfix=0
suffix=

errormsg=
atexit_error()
{
    if test -n "$errormsg"
    then
        printf "%s\n" "$errormsg"
    fi
}
trap atexit_error EXIT

errormsg="$(printf "Malformed VERSION file: %s" "$1")"
set -e
eval "$(sed -n -e 's/^package\.// p' "$1")"
errormsg=

if test -z "$format"
then
    suffix="${suffix:+-}${suffix}"
    if test "$bugfix" -eq 0
    then
        printf "%d.%d%s" "$major" "$minor" "$suffix"
    else
        printf "%d.%d.%d%s" "$major" "$minor" "$bugfix" "$suffix"
    fi
    if test -t 1
    then
        printf "\n"
    fi
else
    eval "$format"
fi
