#! /bin/sh

# this script expects an input that contains at least these lines specifying
# the package version, e.g.
# package.major=0
# package.bugfix=0
# package.suffix=
# The following line is optional:
# package.minor=9
# Note that these variable definitions must not be preceeded by whitespace.
# If package.minor is omitted, the output will be
#     major.bugfix-suffix
# Otherwise, if bugfix != 0, the script outputs
#     major.minor.bugfix-suffix
# If bugfix == 0, teh output will be
#     major.minor-suffix
# If suffix is not given, the trailing '-' will be omitted.

format="${2-}"

major=0
# if minor is not set in $1, than its the major.bugfix-suffix version scheme
minor=
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
    if test -z "$minor"
    then # major.bugfix scheme, use $bugfix, even if 0
        printf "%d.%d%s" "$major" "$bugfix" "$suffix"
    else # major.minor.bugfix scheme, use $bugfix if different from 0 only
        if test "$bugfix" -eq 0
        then
            printf "%d.%d%s" "$major" "$minor" "$suffix"
        else
            printf "%d.%d.%d%s" "$major" "$minor" "$bugfix" "$suffix"
        fi
    fi
    if test -t 1
    then
        printf "\n"
    fi
else
    eval "$format"
fi
