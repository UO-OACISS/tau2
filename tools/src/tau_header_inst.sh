#! /bin/sh

# name:          tau-instrument-headers
# synopsis:      Instrument header files with Opari
# authors:       Scott Biersdorff and Chris Spiel
# last revision: Sun Dec  2 09:22:57 UTC 2012


set -e                          # exit on any error
set -u                          # fail on any unset variable
set +x                          # trace execution


readonly GLOBAL__COMMAND_NAME=`basename $0`


# Verbosity level; zero means no messages
GLOBAL__VERBOSITY=0


# Write out a prefixed message to stderr
#
#        display_message PREFIX MESSAGE*
display_message ()
{
    local PREFIX=$1

    shift

    if [ -n "$@" ]
    then
        MESSAGE=": $@"
    else
        MESSAGE=
    fi

    echo "$GLOBAL__COMMAND_NAME: $PREFIX$MESSAGE" 1>&2
}


# Write info-class message if the current verbosity is at least
# VERBOSITY-LEVEL.
#
#        show_info VERBOSITY-LEVEL MESSAGE*
show_info ()
{
    local LEVEL=$1

    shift

    if [ $GLOBAL__VERBOSITY -ge $LEVEL ]
    then
        display_message info "$@"
    fi
}


# Write error message
#
#        show_error MESSAGE*
show_error ()
{
    display_message error "$@"
}


# Write error message and terminate with non-zero exit code.
#
#        raise_error MESSAGE*
raise_error ()
{
    show_error "$@"
    exit 1
}


# Find all necessary TAU and PDB tools.
#
#        find_tools
find_tools()
{
    if [ -z $TAU_MAKEFILE ]
    then
        raise_error "parameter \"TAU_MAKEFILE\" not set"
    fi

    local TAU_BINDIR=`dirname $TAU_MAKEFILE`/../bin
    if [ ! -d $TAU_BINDIR ]
    then
        raise_error "TAU binary directory ($TAU_BINDIR) as defined by path to TAU_MAKEFILE not found"
    fi

    GLOBAL__OPARI2=$TAU_BINDIR/opari2
    GLOBAL__TAU_INSTRUMENTOR=$TAU_BINDIR/tau_instrumentor

    if which pdbcomment
    then
        GLOBAL__PDBCOMMENT=`which pdbcomment`
    else
        GLOBAL__PDBCOMMENT=${PDBCOMMENT:-}
    fi

    return 0
}


# Check the availability of all necessary tools.
#
#        check_tools
check_tools ()
{
    local OK=true

    for X in GLOBAL__PDBCOMMENT GLOBAL__OPARI2 GLOBAL__TAU_INSTRUMENTOR
    do
        # IMPLEMENTATION NOTE
        #     Bash has indirect variable references and we would say
        #         tool=${!X}
        #     here.
        eval "TOOL=\$$X"

        if test -n "$TOOL" && type $TOOL > /dev/null 2>&1
        then
            show_info 1 "found tool ${X#GLOBAL__}"
            show_info 2 "  ->  \"$TOOL\""
        else
            OK=false
            show_error "tool ${X#GLOBAL__} (-> \"$TOOL\") not found"
        fi
    done

    $OK  ||  exit 1
}


# Run pdbcomment tool on given PDB file.
#
#         run_pdbcomment PDB COMMENT-OUTPUT
run_pdbcomment ()
{
    local PDB=$1
    local COMMENT_OUTPUT=$2

    show_info 2 "pdbcomment $PDB  =>  $COMMENT_OUTPUT"
    $GLOBAL__PDBCOMMENT -o $COMMENT_OUTPUT $PDB  ||  raise_error pdbcomment
}




# Run OPari2.
#
#         run_opari2 CHECKED-HEADER-FILE POMP-FILE
run_opari2 ()
{
    local CHK=$1
    local POMP=$2

    show_info 2 "opari2 $CHK  =>  $POMP"
    $GLOBAL__OPARI2 --c++ --nosrc $CHK $POMP  ||  raise_error opari2
}


# Fix a header file to make it includable multiple times.
#
#         fix_header HEADER_FILE FIXED_HEADER_FILE
fix_header ()
{
    local HEADER=$1
    local FIXED_HEADER=$2

    local TEMPORARY=`mktemp $GLOBAL__COMMAND_NAME.XXXXXXXXXX`
    test -w $TEMPORARY  ||  raise_error temporary header file
    mv $FIXED_HEADER $TEMPORARY

    show_info 2 "fix header file $HEADER  =>  $FIXED_HEADER"

    (
        TAG=`echo ${HEADER}_opari_inc | tr -c '[A-Za-z0-9_]' _ | tr '[a-z]' '[A-Z]'`

        cat <<END_OF_HEADER
#ifndef HEADER_${TAG}_INCLUDED
#define HEADER_${TAG}_INCLUDED
END_OF_HEADER

        cat $TEMPORARY

        cat <<END_OF_FOOTER
#endif /* HEADER_${TAG}_INCLUDED */
END_OF_FOOTER
    ) > $FIXED_HEADER

    rm --force $TEMPORARY
}


# Run TAU's instrumentor on POMP-FILE.
#
#         run_instrumentor PDB POMP-FILE INSTRUMENTED-POMP-FILE
run_instrumentor ()
{
    local PDB=$1
    local POMP=$2
    local INST=$3

    show_info 2 "tau_instrumentor $PDB $POMP  =>  $INST"
    $GLOBAL__TAU_INSTRUMENTOR $PDB $POMP -o $INST  ||  raise_error tau_instrumentor
}


# Instument a single HEADER using a PDB file.
#
#        instrument_header PDB HEADER
instrument_header ()
{
    local PDB=$1
    local HEADER=$2

    test -n "$PDB"  ||  raise_error "instrument_header: missing arguments"
    test -n "$HEADER"  ||  raise_error "instrument_header: missing header filename"
    test -z "${3:-}"  ||  raise_error "instrument_header: excessive argument(s)"

    test -r "$PDB"  ||  raise_error "instrument_header: cannot read PDB file \"$PDB\""
    test -r "$HEADER"  ||  raise_error "instrument_header: cannot read header file \"$HEADER\""

    local OUTPUT_DIR=`dirname $PDB`
    local PDB_FILE=`basename $PDB`
    local HEADER_FILE=`basename $HEADER`

    run_pdbcomment $PDB $OUTPUT_DIR/$PDB_FILE.comment
    run_opari2 $OUTPUT_DIR/$HEADER_FILE $OUTPUT_DIR/$HEADER_FILE.pomp
    fix_header $HEADER_FILE $OUTPUT_DIR/$HEADER_FILE.opari.inc

    run_instrumentor $PDB $OUTPUT_DIR/$HEADER_FILE.pomp $OUTPUT_DIR/$HEADER_FILE.pomp.inst

    rm --force $OUTPUT_DIR/$PDB.comment 

    return 0
}


# Instument multiple HEADER files using a single PDB file.
#
#        instrument_multiple_headers PDB HEADER*
instrument_multiple_headers ()
{
    local PDB=$1

    test -n "$PDB"  ||  raise_error "instrument_headers: missing arguments"

    shift

    for X in "$@"
    do
        show_info 1 "instrumenting $X"
        instrument_header $PDB $X
    done
}


########################################################################


# Print version information and exit.
#
#        show_version
show_version ()
{
    echo $GLOBAL__COMMAND_NAME 0.3
    echo "Copyright (C) 2010-2012 Chris Spiel, Scott Biersdorff"

    exit 0
}


# Print help message and exit.
#
#        show_help
show_help ()
{
    cat <<END_OF_HELP_TEXT
Usage: $GLOBAL__COMMAND_NAME [OPTIONS...] PDB-FILE HEADER...
Instrument (C++) HEADER files with OPari2 directives using PDB-FILE.

Note: environment variable TAU_MAKEFILE must be set!

Options:
      --instrumentor=TAU-INSTRUMENTOR    override tau_instrumentor
      --opari=OPARI                      override opari2
      --pdbcomment=PDB-COMMENT           override pdbcomment

      --check-tools                      only check for tools

  -h, --help                             display this help and exit
  -v, --verbose[=LEVEL]                  increase verbosity or set directly
                                         to LEVEL
  -V, --version                          output version information and exit

Environment:
  TAU_MAKEFILE                           take paths to TAU tools from Makefile
                                         path
  PDBCOMMENT                             set pdbcomment

The search order for TAU tools is
  1. The Sibling path of TAU_MAKEFILE and then
  2. the associated command-line option.
The pdbcomment tool is searched
  1. Along the current PATH and if it is not found
  2. the contents of environment variable PDBCOMMENT.
  3. Finally, command-line option "--pdbcomment" overrides any previous
     definition.

Examples:
  Each header file (e.g. "foo.hh") that should be processed must be
  included conditionally:

#if defined(USE_POMP_INSTRUMENTED_HEADER)
#include "foo.hh.pomp.inst"
#else
#include "foo.hh"
#endif

The rule in a Makefile could look like this:

foo.instrumented: foo.cc foo.hh
        \$(TAUCXX) -optKeepFiles \$(CXXFLAGS) -o \$@ \$<
        tau-instrument-headers \$(basename \$<).pomp.pdb tau-test-template.hh
        \$(TAUCXX) -DUSE_POMP_INSTRUMENTED_HEADER \$(CXXFLAGS) -o \$@ \$<

The "-optKeepFile" in the first call to the TAU-wrapped compiler is
important as tau-instrument-headers needs the program database
(*.pdb).  The second call sets the necessary define
"USE_POMP_INSTRUMENTED_HEADER" to include the instrumented header.

END_OF_HELP_TEXT

    exit 0
}


GLOBAL__NUMBER_OF_OPTIONS=0


# Parse all options and set the index of the first argument so that we
# know later on how many options to drop.
#
#         parse_options ARGV
parse_options ()
{
    local N=0
    local DO_CHECK_TOOLS=false

    while test -n "${1:-}"
    do
        case "$1" in
            --instrumentor)
                GLOBAL__TAU_INSTRUMENTOR=$2
                shift
                N=$((N + 2))
                ;;
            --instrumentor=*)
                GLOBAL__TAU_INSTRUMENTOR=${1#--instrumentor=}
                N=$((N + 1))
                ;;
            --opari)
                GLOBAL__OPARI2=$2
                shift
                N=$((N + 2))
                ;;
            --opari=*)
                GLOBAL__OPARI2=${1#--opari=}
                N=$((N + 1))
                ;;
            --pdbcomment)
                GLOBAL__PDBCOMMENT=$2
                shift
                N=$((N + 2))
                ;;
            --pdbcomment=*)
                GLOBAL__PDBCOMMENT=${1#--pdbcomment=}
                N=$((N + 1))
                ;;

            --check-tools)
                DO_CHECK_TOOLS=true
                N=$((N + 1))
                ;;
            --no-check-tools)
                DO_CHECK_TOOLS=false
                N=$((N + 1))
                ;;

            -h | --help)
                show_help
                N=$((N + 1))
                ;;
            -V | --version)
                show_version
                N=$((N + 1))
                ;;
            -v | --verbose)
                GLOBAL__VERBOSITY=$((GLOBAL__VERBOSITY + 1))
                N=$((N + 1))
                ;;
            -v*)
                GLOBAL__VERBOSITY=${1#-v}
                N=$((N + 1))
                ;;
            --verbose=*)
                GLOBAL__VERBOSITY=${1#--verbose=}
                N=$((N + 1))
                ;;

            --)
                break
                ;;

            --*)
                raise_error "unrecognized long option \"$1\""
                ;;
            -*)
                raise_error "unrecognized short option \"$1\""
                ;;

            *)
                break
                ;;
        esac
        shift
    done

    GLOBAL__NUMBER_OF_OPTIONS=$N

    if $DO_CHECK_TOOLS
    then
        check_tools
        exit 0
    fi

    return 0
}


main ()
{
    case "$1" in
        -h | --help | -V | --version) # skip tool check for these
            parse_options "$@"
            ;;

        --manual-page) # intentionally undocumented option
            help2man --no-info --name "semi-automatically instrument header files" $0
            # Preview the result with
            #     tau-instrument-headers --manual-page | man -l -
            exit 0
            ;;
    esac

    find_tools

    parse_options "$@"
    shift $GLOBAL__NUMBER_OF_OPTIONS

    check_tools
    instrument_multiple_headers "$@"

    exit 0
}


main "$@"


# Local Variables:
# mode: shell-script
# End:
