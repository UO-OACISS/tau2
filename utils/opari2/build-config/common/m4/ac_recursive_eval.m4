dnl copied from http://ac-archive.sourceforge.net/adl/relpaths.m4
dnl @category Misc
dnl @author Alexandre Duret-Lutz <duret_g@epita.fr>
dnl @version 2001-05-25
dnl @license GPLWithACException

dnl adl_RECURSIVE_EVAL(VALUE, RESULT)
dnl =================================
dnl Interpolate the VALUE in loop until it doesn't change,
dnl and set the result to $RESULT.
dnl WARNING: It's easy to get an infinite loop with some unsane input.
AC_DEFUN([adl_RECURSIVE_EVAL],
[_lcl_receval="$1"
$2=`(test "x$prefix" = xNONE && prefix="$ac_default_prefix"
     test "x$exec_prefix" = xNONE && exec_prefix="${prefix}"
     _lcl_receval_old=''
     while test "[$]_lcl_receval_old" != "[$]_lcl_receval"; do
       _lcl_receval_old="[$]_lcl_receval"
       eval _lcl_receval="\"[$]_lcl_receval\""
     done
     echo "[$]_lcl_receval")`])
