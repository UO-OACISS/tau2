#! /bin/bash -e

# returns true, if the variable (given as first argument) is unset or has
# the NULL value
isnull()
{
    eval "test \"\${$1:+set}\" != set"
}

argv0=$0

get_repos_list()
{
    # Hand-made list of repos based on the svn:externals dependencies
    case ${1-} in
    (silc|"") echo common utility opari2 otf2 silc;;
    (otf2)    echo common utility otf2;;
    (opari2)  echo common opari2;;
    (utility) echo common utility;;
    (common)  echo common;;
    esac
}

usage()
{
    printf "Usage: %s [--top TOP] [--top-rev REV] [--no-commit] SOURCE TARGET\n" "$argv0"
    test $# -gt 0 &&
    {
        printf "Try \`%s --help' for more information.\n" "$argv0"
        exit $1
    }
    printf "       %s --setup ROOT\n" "$argv0"
    printf "       %s --help\n" "$argv0"
    printf "Creates a branch or tag in the silc project.\n"
    printf "\n"
    printf "Needs the following set of variables defined:\n"
    for repos in $(get_repos_list)
    do
        REPOS=$(echo "$repos" | tr '[[:lower:]]' '[[:upper:]]')
        printf "    SILC_SVN_%-7s  Absolute path to a root checkout of the %s repository.\n" \
            $REPOS $repos
    done
    printf "\n"
    printf "You can use the --setup option to checkout all repositories into\n"
    printf "the directory pointed to by ROOT. It than echoes suitable assignments\n"
    printf "for all variables. You should than export these in your shell.\n"
    exit 0
}

die()
{
    printf "%s: " "$argv0" >&2
    printf "$@" >&2
    usage 1 >&2
}

if test $# -eq 1 && test "$1" = "--help"
then
    usage
fi

base_url=https://silc.zih.tu-dresden.de/svn

if test $# -ge 1 && test "$1" = "--setup"
then
    if test $# -ne 2
    then
        die "Wrong number of arguments for --setup.\n"
    fi

    root=$2
    mkdir -p "$root" ||
        die "Can't create directory for repositories.\n"

    cd "$root" ||
        die "Can't change to directory for repositories.\n"

    for repos in $(get_repos_list)
    do
        REPOS=$(echo "$repos" | tr '[[:lower:]]' '[[:upper:]]')

        svn co --ignore-externals $base_url/$repos-root >/dev/null ||
            die "Can't checkout %s repository.\n" $repos

        printf "SILC_SVN_%s=\"%s/%s-root\"\n" $REPOS "$PWD" $repos
    done

    exit 0
fi

top_repos=silc

if test $# -gt 2 && test "$1" = "--top"
then
    shift
    top_repos=$1
    shift
    if test "x$(get_repos_list "$top_repos")" = "x"
    then
        die "Unknown repository for --top flag: %s.\n" "$top_repos"
    fi
fi

top_rev=
if test $# -gt 2 && test "$1" = "--top-rev"
then
    shift
    top_rev=-r$1
    shift
fi

do_commit=
if test $# -gt 1 && test "$1" = "--no-commit"
then
    shift
    do_commit=:
fi

if test $# -ne 2
then
    die "Wrong number of arguments.\n"
fi

SOURCE=$1
TARGET=$2

case $SOURCE in
(*/*/*) die "Invalid source: %s.\n" "$SOURCE" ;;

(trunk)
    : accept
    EXTERNALS_SOURCE="$SOURCE"
;;

(branches/*|tags/*)
    : accept
    # SOURCE has at least one /, and should have at most one
    SOURCE_SUFFIX="${SOURCE#*/}"

    EXTERNALS_SOURCE="${SOURCE%%/*}/${top_repos}-$SOURCE_SUFFIX"
;;

(*) die "Invalid source: %s.\n" "$SOURCE" ;;
esac

case $TARGET in
(*/*/*) die "Invalid target: %s.\n" "$TARGET" ;;
(branches/*|tags/*) : accept ;;
(*) die "Invalid target: %s.\n" "$TARGET" ;;
esac

# TARGET has exactly one /
EXTERNALS_TARGET="${TARGET%%/*}/${top_repos}-${TARGET#*/}"

for repos in $(get_repos_list)
do
    REPOS=$(echo "$repos" | tr '[[:lower:]]' '[[:upper:]]')

    if isnull SILC_SVN_$REPOS
    then
        die "SILC_SVN_%s variable for %s repository not set.\n" $REPOS $repos
    fi

    case "$(eval "echo \"\$SILC_SVN_$REPOS\"")"
    in
    (/*) : full path ;;
    (*)
        die "SILC_SVN_%s variable does not hold an absolute path.\n" $REPOS
    ;;
    esac

    if eval "test ! -d \"\$SILC_SVN_$REPOS\""
    then
        die "SILC_SVN_%s variable does not point to a directory.\n" $REPOS
    fi

    # check if the target exists
    eval "cd \"\$SILC_SVN_$REPOS\""

    source="$EXTERNALS_SOURCE"
    target="$EXTERNALS_TARGET"
    # only the top repos get the desired target name, sub repos get an
    # generated name, based on the top repos one
    if test $repos = $top_repos
    then
        source=$SOURCE
        target=$TARGET
    fi

    if test -d "$target"
    then
        die "Target directory \"%s\" exists in %s repository\n" "$target" $repos
    fi
done

for repos in $(get_repos_list "$top_repos")
do
    REPOS=$(echo "$repos" | tr '[[:lower:]]' '[[:upper:]]')

    source="$EXTERNALS_SOURCE"
    source_rev=
    target="$EXTERNALS_TARGET"
    # only the top repos get the desired target name, sub repos get an
    # generated name, based on the top repos one
    if test $repos = $top_repos
    then
        source=$SOURCE
        source_rev=$top_rev
        target=$TARGET
    fi

    printf "Creating \"%s\" in the %s repository\n" "$target" $repos

    eval "cd \"\$SILC_SVN_$REPOS\""

    svn up --ignore-externals $source_rev "$source" >/dev/null ||
        die "Can't update \"%s\" in %s repository.\n" "$source" $repos

    svn cp "$source" "$target" >/dev/null ||
        die "Can't create \"%s\" in %s repository.\n" "$target" $repos

    cd "$target"
    # we process only externals under vendor
    if test -d vendor
    then
        svn propget --strict svn:externals vendor |
            while read external url
            do
                case "$external" in
                (otf2|opari2|utility|common)
                    printf "%-16s %s/%s-root/%s\n" \
                        "$external" \
                        "$base_url" \
                        "$external" \
                        "$EXTERNALS_TARGET"
                ;;
                (*)
                    echo "%s %s\n" "$external" "$url"
                ;;
                esac
            done >.svn/tmp/externals.prop
        svn propset --quiet svn:externals -F .svn/tmp/externals.prop vendor
        svn propget --strict svn:externals vendor
    fi

    $do_commit svn ci -m "Creating $target" ||
        die "Can't commit in %s repository.\n" $repos
done
