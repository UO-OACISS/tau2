#! /bin/sh

# this script expects an input that contains at least three lines specifying
# the library interface version, e.g.
# library.current=1
# library.revision=2
# library.age=4

format="${2-}"

current=0
revision=0
age=0

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
eval "$(sed -n -e 's/^library\.// p' "$1")"
errormsg=

if test -z "$format"
then
    printf "%d:%d:%d" "$current" "$revision" "$age"
    if test -t 1
    then
        printf "\n"
    fi
else
    eval "$format"
fi
