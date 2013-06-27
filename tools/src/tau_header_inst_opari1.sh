#!/bin/bash

# name:          tau-instrument-headers
# synopsis:      Instrument header files with Opari
# authors:       Scott Biersdorff and Chris Spiel
# last revision: Fri Jun 12 05:19:44 UTC 2009
# bash version:  3.2.39


readonly command_name=`basename $0`

# Verbosity level; zero means no messages
verbose=0


# Print version information and exit
#
#        show_version
function show_version
{
    echo $command_name 0.1
    exit 0
}


# Print help message and exit
#
#        show_help
function show_help
{
cat <<EOF
Usage: $command_name PDB-FILE HEADER*
Instrument HEADER files with Opari directives.

Options
  -h, --help       display this help and exit
  -V, --version    output version information and exit
  -v, --verbose    report what is going on; repeat to increase verbosity
EOF

    exit 0
}


# Write out a prefixed message to stderr
#
#        display_message PREFIX MESSAGE*
function display_message
{
    local prefix=$1

    shift

    if [ -n "$@" ]; then
        message=": $@"
    else
        message=
    fi

    echo "$command_name: $prefix$message" 1>&2
}


# Write info-class message
#
#        show_info MESSAGE*
function show_info
{
    display_message info "$@"
}


# Write error message
#
#        show_error MESSAGE*
function show_error
{
    display_message error "$@"
}


# Write error message and terminate with non-zero exit code.
#
#        raise_error MESSAGE*
function raise_error
{
    show_error "$@"
    exit 1
}


# Check the availability of all necessary tools.
#
#        check_tools
function check_tools
{
		if [ "x$TAU_MAKEFILE" != "x" ] ; then
			MAKEFILE=$TAU_MAKEFILE
		else
				echo $0: "ERROR: please set the environment variable TAU_MAKEFILE"
				exit 1
		fi
		
		cat <<EOF > /tmp/makefile.tau.$USER.$$
		include $MAKEFILE
		all:
			print-%:; @echo $($*)
		EOF
		PDTARCH=`make -s -f /tmp/makefile.tau.$USER.$$ print-PDTARCHDIR`
		/bin/rm -f /tmp/makefile.tau.$USER.$$

		readonly pdbcomment=@PDTROOTDIR@/$PDTARCH/bin/pdbcomment
		readonly tau_ompcheck=@TAUROOTDIR@/@ARCH@/bin/tau_ompcheck
		readonly opari=@TAUROOTDIR@/@ARCH@/bin/opari
		readonly tau_instrumentor=@TAUROOTDIR@/@ARCH@/bin/tau_instrumentor

    local ok=1

    for x in pdbcomment tau_ompcheck opari tau_instrumentor; do
        if type -t ${!x} >/dev/null; then
            :
        else
            show_error "tool \"$x\" (-> \"${!x}\") not found"
            ok=0
        fi
    done

    test $ok -eq 0 && exit 1
}


# Instument a single HEADER using a PDB file.
#
#        instrument_header PDB HEADER
function instrument_header
{
    local pdb=$1
    local header=$2

    test -n "$pdb" || raise_error "instrument_header: missing arguments"
    test -n "$header" || raise_error "instrument_header: missing header filename"
    test -z "$3" || raise_error "instrument_header: excessive argument(s)"

    local output_dir=`dirname $pdb`
    local pdb_file=`basename $pdb`
    local header_file=`basename $header`

    test $verbose -ge 2 && \
        show_info "pdbcomment $pdb  =>  $output_dir/$pdb_file.comment"
    $pdbcomment -o $output_dir/$pdb_file.comment $pdb || \
        raise_error pdbcomment

    test $verbose -ge 2 && \
        show_info "tau_ompcheck $output_dir/$pdb.comment $header  =>  $output_dir/$header_file.chk"
    $tau_ompcheck $output_dir/$pdb.comment $header -o $output_dir/$header_file.chk || \
        raise_error tau_ompcheck

    test $verbose -ge 2 && \
        show_info "opari $output_dir/$header_file.chk  =>  $output_dir/$header_file.chk.pomp"
    $opari -nosrc -table $output_dir/opari.tab.c $output_dir/$header_file.chk $output_dir/$header_file.chk.pomp || \
        raise_error opari
    mv $output_dir/$header_file.chk.opari.inc $output_dir/_$header_file.chk.opari.inc
    (
        tag=`echo ${header_file}_chk_opari_inc | tr -c '[A-Za-z0-9_]' _ | tr '[a-z]' '[A-Z]'`
        cat <<EOF_HEADER
#ifndef HEADER_${tag}_INCLUDED_
#define HEADER_${tag}_INCLUDED_
EOF_HEADER
        cat $output_dir/_$header_file.chk.opari.inc
        cat <<EOF_FOOTER
#endif /* HEADER_${tag}_INCLUDED_ */
EOF_FOOTER
    ) > $output_dir/$header_file.chk.opari.inc
    rm -f $output_dir/_$header_file.chk.opari.inc

    test $verbose -ge 2 && \
        show_info "tau_instrumentor $pdb $output_dir/$header_file.chk.pomp  =>  $output_dir/$header_file.chk.pomp.inst"
    $tau_instrumentor $pdb $output_dir/$header_file.chk.pomp -o $output_dir/$header_file.chk.pomp.inst || \
        raise_error tau_instrumentor

    rm -f $output_dir/$pdb.comment $output_dir/$header_file.chk

    return 0
}


# Instument multiple HEADER files using a single PDB file.
#
#        instrument_multiple_headers PDB HEADER*
function instrument_multiple_headers
{
    local pdb=$1

    test -n "$pdb" || raise_error "instrument_headers: missing arguments"

    shift

    for x in "$@"; do
        test $verbose -ge 1 && show_info "instrumenting $x"
        instrument_header $pdb $x
    done
}


function main
{
    while test -n "$1"; do
        case "$1" in
            -h | --help)
                show_help
                ;;
            -V | --version)
                show_version
                ;;
            -v | --verbose)
                : $((verbose++))
                ;;
            --)
                arguments="$arguments $@"
                break
                ;;
            -*)
                raise_error "unrecognized option \"$1\""
                ;;
            *)
                arguments="$arguments $1"
                ;;
        esac
        shift
    done

    check_tools
    instrument_multiple_headers $arguments

    exit 0
}


main $@
