## -*- mode: makefile -*-

##
## This file is part of the Score-P software (http://www.score-p.org)
##
## Copyright (c) 2009-2011,
## RWTH Aachen, Germany
##
## Copyright (c) 2009-2011,
## Gesellschaft fuer numerische Simulation mbH Braunschweig, Germany
##
## Copyright (c) 2009-2011,
## Technische Universitaet Dresden, Germany
##
## Copyright (c) 2009-2011,
## University of Oregon, Eugene, USA
##
## Copyright (c) 2009-2011, 2013-2015,
## Forschungszentrum Juelich GmbH, Germany
##
## Copyright (c) 2009-2011,
## German Research School for Simulation Sciences GmbH, Juelich/Aachen, Germany
##
## Copyright (c) 2009-2011,
## Technische Universitaet Muenchen, Germany
##
## This software may be modified and distributed under the terms of
## a BSD-style license. See the COPYING file in the package base
## directory for details.
##

## file       build-frontend/Makefile.am


## Options passed to aclocal to specify local macro directories. Although
## ACLOCAL_AMFLAGS should be replaced by AC_CONFIG_MACRO_DIRS, aclocal-1.13.4
## doesn't honor the paths provided to AC_CONFIG_MACRO_DIRS. See
## https://debbugs.gnu.org/cgi/bugreport.cgi?bug=21352
ACLOCAL_AMFLAGS = -I ../build-config/m4 -I ../vendor/common/build-config/m4

include ../build-includes/common.am

AM_CPPFLAGS += -DFRONTEND_BUILD -DCROSS_BUILD

include ../build-includes/frontend-only.am

clean-local:
	rm -rf ../test/jacobi


if SCOREP_HAVE_F77
if HAVE_OPENMP_SUPPORT
installcheck-local:
	@$(am__cd) $(DESTDIR)$(docdir)/example/ && $(MAKE) $(AM_MAKEFLAGS)
	@$(am__cd) $(DESTDIR)$(docdir)/example_user_instrumentation/ && $(MAKE) $(AM_MAKEFLAGS)
endif
else
if HAVE_OPENMP_SUPPORT
installcheck-local:
	@$(am__cd) $(DESTDIR)$(docdir)/example/ && $(MAKE) $(AM_MAKEFLAGS) c
	@$(am__cd) $(DESTDIR)$(docdir)/example_user_instrumentation/ && $(MAKE) $(AM_MAKEFLAGS) c
endif
endif

install-example:
if HAVE_OPENMP_SUPPORT
	$(MKDIR_P) $(DESTDIR)$(docdir)/example
	$(INSTALL_DATA) $(abs_srcdir)/../doc/example/example.*            $(DESTDIR)$(docdir)/example/
	$(INSTALL_DATA) $(abs_srcdir)/../include/opari2/pomp2_lib.h       $(DESTDIR)$(docdir)/example/
	$(INSTALL_DATA) $(abs_builddir)/../doc/example/Makefile           $(DESTDIR)$(docdir)/example/
	$(INSTALL_DATA) $(abs_builddir)/../build-frontend/.libs/libpomp.* $(DESTDIR)$(docdir)/example/
	rm -f $(DESTDIR)$(docdir)/example/libpomp.la
endif
	$(MKDIR_P) $(DESTDIR)$(docdir)/example_user_instrumentation
	$(INSTALL_DATA) $(abs_srcdir)/../doc/example_user_instrumentation/example_user_instrumentation.*  $(DESTDIR)$(docdir)/example_user_instrumentation/
	$(INSTALL_DATA) $(abs_srcdir)/../include/opari2/pomp2_user_lib.h                                  $(DESTDIR)$(docdir)/example_user_instrumentation/
	$(INSTALL_DATA) $(abs_builddir)/../doc/example_user_instrumentation/Makefile                      $(DESTDIR)$(docdir)/example_user_instrumentation/
if HAVE_OPENMP_SUPPORT
	$(INSTALL_DATA) $(abs_builddir)/../build-frontend/.libs/libpomp.*                                 $(DESTDIR)$(docdir)/example_user_instrumentation/
endif
	rm -f $(DESTDIR)$(docdir)/example_user_instrumentation/libpomp.la

install-pomp2_region_info:
	$(MKDIR_P) $(DESTDIR)$(datadir)/@PACKAGE@/devel
	$(INSTALL_DATA) $(abs_srcdir)/../src/opari-lib-dummy/opari2_ctc_token.h $(DESTDIR)$(datadir)/@PACKAGE@/devel/
	$(INSTALL_DATA) $(abs_srcdir)/../src/opari-lib-dummy/opari2_region_info.h $(DESTDIR)$(datadir)/@PACKAGE@/devel/
	$(INSTALL_DATA) $(abs_srcdir)/../src/opari-lib-dummy/opari2_ctc_parser.h $(DESTDIR)$(datadir)/@PACKAGE@/devel/
	$(INSTALL_DATA) $(abs_srcdir)/../src/opari-lib-dummy/opari2_ctc_parser.c $(DESTDIR)$(datadir)/@PACKAGE@/devel/
	$(INSTALL_DATA) $(abs_srcdir)/../src/opari-lib-dummy/pomp2_region_info.h $(DESTDIR)$(datadir)/@PACKAGE@/devel/
	$(INSTALL_DATA) $(abs_srcdir)/../src/opari-lib-dummy/pomp2_region_info.c $(DESTDIR)$(datadir)/@PACKAGE@/devel/
	$(INSTALL_DATA) $(abs_srcdir)/../src/opari-lib-dummy/pomp2_user_region_info.h $(DESTDIR)$(datadir)/@PACKAGE@/devel/
	$(INSTALL_DATA) $(abs_srcdir)/../src/opari-lib-dummy/pomp2_user_region_info.c $(DESTDIR)$(datadir)/@PACKAGE@/devel/

install-data-local: install-example install-pomp2_region_info
