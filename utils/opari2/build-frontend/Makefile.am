## -*- mode: makefile -*-

## 
## This file is part of the Score-P software (http://www.score-p.org)
##
## Copyright (c) 2009-2011, 
##    RWTH Aachen, Germany
##    Gesellschaft fuer numerische Simulation mbH Braunschweig, Germany
##    Technische Universitaet Dresden, Germany
##    University of Oregon, Eugene, USA
##    Forschungszentrum Juelich GmbH, Germany
##    German Research School for Simulation Sciences GmbH, Juelich/Aachen, Germany
##    Technische Universitaet Muenchen, Germany
##
## See the COPYING file in the package base directory for details.
##

## file       build-frontend/Makefile.am 
## maintainer Christian Roessel <c.roessel@fz-juelich.de>


## ACLOCAL_AMFLAGS contains options to pass to aclocal when aclocal.m4
## is to be rebuilt by make. This line is also used by autoreconf to
## run aclocal with suitable options, or by autopoint and gettextize
## to locate the place where Gettext's macros should be installed. So
## even if you do not really care about the rebuild rules, you should
## define ACLOCAL_AMFLAGS.
## For some reason this can't be moved to common.am.
ACLOCAL_AMFLAGS = -I ../vendor/common/build-config/m4

include ../build-includes/common.am

AM_CPPFLAGS += -DFRONTEND_BUILD -DCROSS_BUILD

include ../build-includes/frontend-only.am

CONFIG_HEADER_FRONTEND = ../src/opari2_config_tool_frontend.h


clean-local:
	rm -rf ../test/jacobi
	rm -f $(CONFIG_HEADER_FRONTEND) $(builddir)/opari2_config.dat

create-config-tool-headers: $(CONFIG_HEADER_FRONTEND)

$(CONFIG_HEADER_FRONTEND): $(builddir)/opari2_config.dat
	@echo "Creating $(CONFIG_HEADER_FRONTEND)"
	@echo "#define NM      \"$(NM)\""               >  $(CONFIG_HEADER_FRONTEND)
	@echo "#define AWK     \"$(AWK)\""              >> $(CONFIG_HEADER_FRONTEND)
	@echo "#define SCRIPT  \"$(prefix)/bin/pomp2_parse_init_regions.awk\"" >> $(CONFIG_HEADER_FRONTEND)
	@echo "#define EGREP   \"$(EGREP)\""            >> $(CONFIG_HEADER_FRONTEND)
	@echo "#define VERSION \"$(PACKAGE_VERSION)\""  >> $(CONFIG_HEADER_FRONTEND)

$(builddir)/opari2_config.dat:
	@echo "Creating $(builddir)/opari2_config.dat"
	@echo NM=\"$(NM)\"                   >  $(builddir)/opari2_config.dat
	@echo AWK=\"$(AWK)\"                 >> $(builddir)/opari2_config.dat
	@echo OPARI_SCRIPT=\"$(PWD)/../build-frontend/pomp2_parse_init_regions.awk\"  >> $(builddir)/opari2_config.dat
	@echo VERSION=\"$(PACKAGE_VERSION)\" >> $(builddir)/opari2_config.dat
	@echo EGREP=\"$(EGREP)\"             >> $(builddir)/opari2_config.dat

if SCOREP_HAVE_F77
if OPENMP_SUPPORTED
installcheck-local:
	(cd $(DESTDIR)$(datadir)/@PACKAGE_NAME@/doc/example/ && $(MAKE) $(AM_MAKEFLAGS)) || exit 1;
	(cd $(DESTDIR)$(datadir)/@PACKAGE_NAME@/doc/example_user_instrumentation/ && $(MAKE) $(AM_MAKEFLAGS)) || exit 1;
endif
else
if OPENMP_SUPPORTED
installcheck-local:
	(cd $(DESTDIR)$(datadir)/@PACKAGE_NAME@/doc/example/ && $(MAKE) $(AM_MAKEFLAGS) c ) || exit 1;
	(cd $(DESTDIR)$(datadir)/@PACKAGE_NAME@/doc/example_user_instrumentation/ && $(MAKE) $(AM_MAKEFLAGS) c ) || exit 1;
endif
endif

if OPENMP_SUPPORTED
install-example:
	$(MKDIR_P) $(DESTDIR)$(datadir)/@PACKAGE_NAME@/doc/example
	$(INSTALL_DATA) $(abs_srcdir)/../doc/example/example.*            $(DESTDIR)$(datadir)/@PACKAGE_NAME@/doc/example/
	$(INSTALL_DATA) $(abs_srcdir)/../src/pomp-lib-dummy/pomp2_lib.h   $(DESTDIR)$(datadir)/@PACKAGE_NAME@/doc/example/
	$(INSTALL_DATA) $(abs_builddir)/../doc/example/Makefile           $(DESTDIR)$(datadir)/@PACKAGE_NAME@/doc/example/
	$(INSTALL_DATA) $(abs_builddir)/../build-frontend/.libs/libpomp.* $(DESTDIR)$(datadir)/@PACKAGE_NAME@/doc/example/
	rm -f $(DESTDIR)$(datadir)/@PACKAGE_NAME@/doc/example/libpomp.la
	$(MKDIR_P) $(DESTDIR)$(datadir)/@PACKAGE_NAME@/doc/example_user_instrumentation
	$(INSTALL_DATA) $(abs_srcdir)/../doc/example_user_instrumentation/example_user_instrumentation.*  $(DESTDIR)$(datadir)/@PACKAGE_NAME@/doc/example_user_instrumentation/
	$(INSTALL_DATA) $(abs_srcdir)/../src/pomp-lib-dummy/pomp2_lib.h                                   $(DESTDIR)$(datadir)/@PACKAGE_NAME@/doc/example_user_instrumentation/
	$(INSTALL_DATA) $(abs_builddir)/../doc/example_user_instrumentation/Makefile                      $(DESTDIR)$(datadir)/@PACKAGE_NAME@/doc/example_user_instrumentation/
	$(INSTALL_DATA) $(abs_builddir)/../build-frontend/.libs/libpomp.*                                 $(DESTDIR)$(datadir)/@PACKAGE_NAME@/doc/example_user_instrumentation/
	rm -f $(DESTDIR)$(datadir)/@PACKAGE_NAME@/doc/example_user_instrumentation/libpomp.la
endif

install-data-local: install-example
