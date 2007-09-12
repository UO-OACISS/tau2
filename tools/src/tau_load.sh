#!/bin/sh
TAU_INSTALLATION_DIR=@TAUROOTDIR@
TAU_ARCHITECTURE_DIR=@ARCH@
TAU_LIB=libTAU.so

for arg in "$@"; do
  case $arg in
      -XrunTAU*)
	  myarg=`echo $arg | sed 's/-XrunTAU//'`
	  TAU_LIB=libTAU$myarg.so
	  ;;
      *)
	  ARGS="$ARGS $arg"
	  ;;
  esac  
done

TAU_LOADLIB=$TAU_INSTALLATION_DIR/$TAU_ARCHITECTURE_DIR/lib/$TAU_LIB
export LD_PRELOAD=$TAU_LOADLIB
$ARGS
