## -*- mode: autoconf -*-

##
## This file is part of the Score-P software ecosystem (http://www.score-p.org)
##
## Copyright (c) 2009-2011,
## RWTH Aachen University, Germany
##
## Copyright (c) 2009-2011,
## Gesellschaft fuer numerische Simulation mbH Braunschweig, Germany
##
## Copyright (c) 2009-2015, 2020,
## Technische Universitaet Dresden, Germany
##
## Copyright (c) 2009-2011,
## University of Oregon, Eugene, USA
##
## Copyright (c) 2009-2013,
## Forschungszentrum Juelich GmbH, Germany
##
## Copyright (c) 2009-2011,
## German Research School for Simulation Sciences GmbH, Juelich/Aachen, Germany
##
## Copyright (c) 2009-2011,
## Technische Universitaet Muenchen, Germany
##
## This software may be modified and distributed under the terms of
## a BSD-style license.  See the COPYING file in the package base
## directory for details.
##


# _AFS_SUMMARY_DEFINE_FN
# ----------------------
# Internal initialization
m4_define([_AFS_SUMMARY_DEFINE_FN],
[AC_REQUIRE_SHELL_FN([afs_fn_summary],
    [AS_FUNCTION_DESCRIBE([afs_fn_summary], [DESCR VALUE INDENT [WRAP-MARGIN=_AFS_SUMMARY_WRAP_MARGIN]],
        [Produces a line-wrapped summary line with DESCR and VALUE, indented by INDENT
and wrapped at WRAP-MARGIN.])],
[dnl function body
    _afs_summary_wrap_width=${4-_AFS_SUMMARY_WRAP_MARGIN}
    _afs_summary_column_width=_AFS_SUMMARY_COLUMN_WIDTH
    _afs_summary_prefix="${3-}  ${1-}:"
    printf "%s" "${_afs_summary_prefix}"
    _afs_summary_padding="$(printf "%-${_afs_summary_column_width}s" "")"
    _afs_summary_value="$(echo "${2-}" | $SED -e 's/  */ /g' -e 's/^ //' -e 's/ $//')"
    AS_IF([test ${#_afs_summary_prefix} -ge ${#_afs_summary_padding}], [
        _afs_summary_nl=" \\$as_nl${_afs_summary_padding}"
    ], [
        as_fn_arith ${#_afs_summary_padding} - ${#_afs_summary_prefix} &&
            _afs_summary_nl="$(printf "%-${as_val}s" "")"
    ])
    _afs_summary_sep=""
    as_fn_arith ${#_afs_summary_padding} + 1 &&
        _afs_summary_column=$as_val
    while test -n "${_afs_summary_value}"
    do
        _afs_summary_entry="${_afs_summary_value%% *}"
        printf "%s" "${_afs_summary_nl}${_afs_summary_sep}${_afs_summary_entry}"

        case "${_afs_summary_value}" in
        (*" "*) _afs_summary_value="${_afs_summary_value#* }" ;;
        (*) _afs_summary_value="" ;;
        esac

        as_fn_arith ${_afs_summary_column} + ${#_afs_summary_entry} + ${#_afs_summary_sep} &&
            _afs_summary_column=$as_val
        AS_IF([test ${_afs_summary_column} -ge ${_afs_summary_wrap_width}], [
            _afs_summary_nl=" \\$as_nl${_afs_summary_padding}"
            _afs_summary_sep=""
            as_fn_arith ${#_afs_summary_padding} + 1 &&
                _afs_summary_column=$as_val
        ], [
            _afs_summary_sep=" "
            _afs_summary_nl=""
        ])
    done
    echo
AS_UNSET([_afs_summary_column_width])
AS_UNSET([_afs_summary_wrap_width])
AS_UNSET([_afs_summary_prefix])
AS_UNSET([_afs_summary_padding])
AS_UNSET([_afs_summary_value])
AS_UNSET([_afs_summary_nl])
AS_UNSET([_afs_summary_sep])
AS_UNSET([_afs_summary_column])
AS_UNSET([_afs_summary_entry])
AS_UNSET([_afs_summary_tag])
AS_UNSET([_afs_summary_tag_final])
])]dnl
) #_AFS_SUMMARY_DEFINE_FN

# _AFS_SUMMARY_INIT
# -----------------
AC_DEFUN_ONCE([_AFS_SUMMARY_INIT],
[AC_REQUIRE([AC_PROG_SED])]
[m4_define_default([_AFS_SUMMARY_COLUMN_WIDTH], 32)]dnl
[m4_define_default([_AFS_SUMMARY_WRAP_MARGIN], 128)]dnl
_AFS_SUMMARY_DEFINE_FN
)# _AFS_SUMMARY_INIT

# AFS_SUMMARY_INIT( [BUILD-MASTER] )
# ----------------------------------
# Initializes the summary system and adds the package header (possibly
# including the sub-build name) to it. It removes config.summary files
# from previous configure runs recursively, therefore you need to call
# AFS_SUMMARY_INIT before any sub-configures.
# The sub-build name is used from the `AFS_PACKAGE_BUILD` variable
# set by the AFS_PACKAGE_INIT macro.
#
# Provide BUILD-MASTER, if this config.summary should be cleaned up
# from duplicate lines the BUILD-MASTER/config.summary file has too.
#
# Autoconf variables:
# `_AFS_SUMMARY_COLUMN_WIDTH`:: The width of the description column.
#                               Defaults to 32.
# `_AFS_SUMMARY_WRAP_MARGIN`::  The wrap margin. Defaults to 128.
AC_DEFUN([AFS_SUMMARY_INIT],
[AC_REQUIRE([_AFS_SUMMARY_INIT])]dnl
[rm -f AC_PACKAGE_TARNAME.summary
LC_ALL=C find . -name 'config.summary*' -exec rm -f '{}' \;
m4_ifnblank(m4_normalize($1), AS_LN_S(m4_normalize($1)[/config.summary], [config.summary-master]))
m4_define([_AFS_SUMMARY_INDENT], [m4_ifndef([AFS_PACKAGE_BUILD], [], [  ])])dnl
m4_define([_AFS_SUMMARY_FILE], [config.summary])dnl
afs_fn_summary \
    "AC_PACKAGE_NAME m4_ifndef([AFS_PACKAGE_BUILD], AC_PACKAGE_VERSION, [(]AFS_PACKAGE_BUILD[)])" \
    "" \
    "_AFS_SUMMARY_INDENT" \
    >_AFS_SUMMARY_FILE
m4_pushdef([_AFS_SUMMARY_INDENT], _AFS_SUMMARY_INDENT[  ])dnl
])


# AFS_SUMMARY_SECTION_BEGIN( [DESCR, [VALUE]] )
# ---------------------------------------------
# Starts a new section, optionally with the given description.
# All summary lines after this call will be indented by 2 spaces.
# Close the section with 'AFS_SUMMARY_SECTION_END'.
AC_DEFUN([AFS_SUMMARY_SECTION_BEGIN], [
AC_REQUIRE([AFS_SUMMARY_INIT])dnl

m4_ifnblank($1, AFS_SUMMARY([$1], [$2]))
m4_pushdef([_AFS_SUMMARY_INDENT], _AFS_SUMMARY_INDENT[  ])dnl
])


# AFS_SUMMARY_SECTION_END
# -----------------------
# Close a previously opened section with 'AFS_SUMMARY_SECTION_BEGIN'.
AC_DEFUN([AFS_SUMMARY_SECTION_END], [
AC_REQUIRE([AFS_SUMMARY_INIT])dnl

m4_popdef([_AFS_SUMMARY_INDENT])dnl
])


# AFS_SUMMARY_PUSH
# ----------------
# Starts a new section (see 'AFS_SUMMARY_SECTION_BEGIN'), but without a
# section heading and it collects all subsequent summaries and sections in
# a hold space.
# All summary lines after this call will be indented by 2 spaces.
# Output the hold space with 'AFS_SUMMARY_POP'.
AC_DEFUN([AFS_SUMMARY_PUSH], [
AC_REQUIRE([AFS_SUMMARY_INIT])dnl

AFS_SUMMARY_SECTION_BEGIN
m4_pushdef([_AFS_SUMMARY_FILE], _AFS_SUMMARY_FILE[.x])dnl
: >_AFS_SUMMARY_FILE
])


# AFS_SUMMARY_POP( DESCR, VALUE )
# -------------------------------
# Close a previously opened section with 'AFS_SUMMARY_PUSH'. Outputs the
# section header with DESCR and VALUE, and then outputs the summary from the
# hold space.
AC_DEFUN([AFS_SUMMARY_POP], [
AC_REQUIRE([AFS_SUMMARY_INIT])dnl

m4_define([_afs_summary_tmp], _AFS_SUMMARY_FILE)dnl
m4_popdef([_AFS_SUMMARY_FILE])dnl
AFS_SUMMARY_SECTION_END

AFS_SUMMARY([$1], [$2])
cat _afs_summary_tmp >>_AFS_SUMMARY_FILE
rm _afs_summary_tmp
m4_undefine([_afs_summary_tmp])
])

# AFS_SUMMARY( DESCR, VALUE, [WRAP-MARGIN] )
# ------------------------------------------
# Generates a summary line with the given description and value.
# DESCR needs to be colon-free. DESCR and VALUE will be expanded in double quotes.
AC_DEFUN([AFS_SUMMARY], [
AC_REQUIRE([AFS_SUMMARY_INIT])dnl
m4_if(m4_index([$1], [:]), [-1],
[afs_fn_summary "_AS_QUOTE([$1])" "_AS_QUOTE([$2])" "_AFS_SUMMARY_INDENT" $3 >>_AFS_SUMMARY_FILE],
[m4_fatal([$0: description should not have a colon (:): ]$1)])])

# AFS_SUMMARY_VERBOSE( DESCR, VALUE )
# -----------------------------------
# Generates a summary line with the given description and value, but only
# if ./configure was called with --verbose
AC_DEFUN([AFS_SUMMARY_VERBOSE], [

AS_IF([test "x${verbose}" = "xyes"], [
    AFS_SUMMARY([$1], [$2])
])
])


# internal
AC_DEFUN([_AFS_SUMMARY_SHOW], [

AS_ECHO([""])
if test ${as__color-no} = yes ||
   { { test "${CLICOLOR-1}" != "0" && test -t 1; } || test "${CLICOLOR_FORCE:-0}" != "0"; }; then
    cat
else
    $SED -e 's/'"$(printf "\033")"'\@<:@@<:@0-9;@:>@*m//g'
fi <AC_PACKAGE_TARNAME.summary
])

# AFS_SUMMARY_COLLECT( [SHOW-COND] )
# --------------------------------
# Collectes the summary of all configures recursively into the file
# $PACKAGE.summary. If SHOW-COND is not given, or the expression
# evaluates to true the summary is also printed to stdout.
# Should be called after AC_OUTPUT.
AC_DEFUN([AFS_SUMMARY_COLLECT], [

(
    AS_ECHO(["Configure command:"])
    _afs_summary_column_width=]_AFS_SUMMARY_COLUMN_WIDTH[
    _afs_summary_prefix="  $as_myself "
    printf "%-${_afs_summary_column_width}s" "${_afs_summary_prefix}"
    _afs_summary_padding="$(printf "%-${_afs_summary_column_width}s" "")"
    AS_IF([test ${#_afs_summary_prefix} -gt ${_afs_summary_column_width}], [
        _afs_summary_sep="\\$as_nl${_afs_summary_padding}"
    ], [
        _afs_summary_sep=""
    ])

    eval "set x $ac_configure_args"
    shift
    AS_FOR([ARG], [_afs_summary_arg], [], [
        AS_CASE([${_afs_summary_arg}],
        [*\'*], [_afs_summary_arg="`$as_echo "${_afs_summary_arg}" | $SED "s/'/'\\\\\\\\''/g"`"])
        AS_ECHO_N(["${_afs_summary_sep}'${_afs_summary_arg}'"])
        _afs_summary_sep=" \\$as_nl${_afs_summary_padding}"
    ])
    AS_ECHO([""])

    AS_IF([test "x${MODULESHOME:+set}" = "xset"], [
        AS_ECHO([""])
        AS_ECHO(["Loaded modules:"])
        _afs_summary_sep=""
        AS_IF([test "x${LOADEDMODULES:+set}" = "xset"], [
            _afs_summary_prefix="  module load "
            printf "%-${_afs_summary_column_width}s" "${_afs_summary_prefix}"
            IFS=': ' eval 'set x $LOADEDMODULES'
            shift
            AS_FOR([MODULE], [_afs_summary_module], [], [
                AS_ECHO_N(["${_afs_summary_sep}${_afs_summary_module}"])
                _afs_summary_sep=" \\$as_nl${_afs_summary_padding}"
            ])
            AS_ECHO([""])
        ], [
            AS_ECHO(["  No modules loaded"])
        ])
    ])

    AS_ECHO([""])
    _afs_summary_sep="Configuration summary:"
    _afs_summary_sub="$(printf "\032")"
    LC_ALL=C find . -name config.summary |
        LC_ALL=C $AWK -F "config.summary" '{print $[]1}' |
        LC_ALL=C sort |
        LC_ALL=C $AWK '{print $[]0 "config.summary"}' |
        while read summary
    do
        AS_ECHO(["${_afs_summary_sep}"])
        if ! test -r $summary-master
        then
            touch $summary-master
        fi
        $SED -e :a -e '/\\$/N; s/\n/'"$_afs_summary_sub"'/; ta' <$summary        >$summary.sub
        $SED -e :a -e '/\\$/N; s/\n/'"$_afs_summary_sub"'/; ta' <$summary-master >$summary-master.sub
        LC_ALL=C $AWK '
function printsummary(ps_summary, ps_descr_pre, ps_descr_post) {
    match(ps_summary, ": *")
    ps_descr = substr(ps_summary, 1, RSTART)
    ps_gap = substr(ps_summary, RSTART+1, RLENGTH-1)
    ps_value = substr(ps_summary, RSTART+RLENGTH)
    match(ps_descr, "^ *")
    ps_indent = substr(ps_descr, 1, RLENGTH)
    ps_descr = substr(ps_descr, RLENGTH+1)
    ps_value_res = ""
    ps_pre  = ""
    ps_post = ""
    ps_firstword = 0
    while (ps_value != "") {
        ps_start = 1
        if (0 != match(ps_value, " *\\\\\032 *")) {
            ps_chunk = substr(ps_value, 1, RSTART - 1)
            ps_sep   = substr(ps_value, RSTART, RLENGTH)
            ps_value = substr(ps_value, RSTART + RLENGTH)
        } else {
            # last chunk
            ps_chunk = ps_value
            ps_sep   = ""
            ps_value = ""
        }
        if (!ps_firstword && length(ps_chunk) > 0) {
            if (match(ps_chunk, "^@<:@a-z@:>@*"))
                ps_word = substr(ps_chunk, RSTART, RLENGTH)
            else
                ps_word = ps_chunk
            if (ps_word == "yes")
                ps_pre  = "\033@<:@0;32m"
            else if (ps_word == "no")
                ps_pre  = "\033@<:@0;31m"
            else
                ps_pre  = "\033@<:@1;34m"
            ps_post = "\033@<:@m"
            ps_firstword = 1
        }
        ps_value_res = ps_value_res "" ps_pre "" ps_chunk "" ps_post "" ps_sep
    }
    print ps_indent "" ps_descr_pre "" ps_descr "" ps_descr_post "" ps_gap "" ps_value_res
}

BEGIN {
    sectionstack@<:@0@:>@ = ""
    sectionstackprinted@<:@0@:>@ = 0
    sectionestacklen = 0
    sectionoffset = -1
}

{
    if (ARGV@<:@1@:>@ == FILENAME) {
        colon = in@&t@dex($[]0, ":")
        refsections@<:@substr($[]0, 1, colon)@:>@ = substr($[]0, colon+1)
        next
    }
    match($[]0, "^ *")
    depth = RLENGTH / 2
    if (sectionoffset == -1) {
        sectionoffset = depth;
    }
    depth -= sectionoffset
    while (sectionestacklen < depth) {
        sectionstack@<:@sectionestacklen@:>@ = ""
        sectionstackprinted@<:@sectionestacklen@:>@ = 1
        sectionestacklen++
    }
    sectionstack@<:@depth@:>@ = $[]0
    sectionstackprinted@<:@depth@:>@ = 0
    sectionestacklen = depth + 1
    colon = in@&t@dex($[]0, ":")
    descr = substr($[]0, 1, colon)
    value = substr($[]0, colon+1)
    if (!(descr in refsections) || refsections@<:@descr@:>@ != value) {
        for (i = 1; i < sectionestacklen; i++) {
            if (!sectionstackprinted@<:@i-1@:>@) {
                printsummary(sectionstack@<:@i-1@:>@, "", "")
                sectionstackprinted@<:@i-1@:>@ = 1
            }
        }
        descr_pre = ""
        descr_post = ""
        if (descr in refsections) {
            # we are here, because the values differ
            # if the first word of the values are equal (yes, no, ...)
            # color as warning/yellow, else red/attention

            match(value, "^@<:@a-z@:>@*")
            word = substr(value, RSTART, RLENGTH)

            ref_value = refsections@<:@descr@:>@
            match(ref_value, "^@<:@a-z@:>@*")
            ref_word = substr(ref_value, RSTART, RLENGTH)

            if (word == ref_word)
                descr_pre = "\033@<:@0;33m"
            else
                descr_pre = "\033@<:@0;31m"
            descr_post = "\033@<:@m"
        }
        printsummary(descr "" value, descr_pre, descr_post)
        sectionstackprinted@<:@depth@:>@ = 1
    }
}
' \
            $summary-master.sub $summary.sub | $SED -e 's/'"$_afs_summary_sub"'/'"\\$as_nl"'/g'
        rm -f $summary.sub $summary-master $summary-master.sub
        _afs_summary_sep=""
    done
) >AC_PACKAGE_TARNAME.summary

m4_ifblank($1,
          [_AFS_SUMMARY_SHOW],
          [AS_IF([$1], [_AFS_SUMMARY_SHOW])])
])
