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
## Copyright (c) 2009-2011, 2014-2015, 2017,
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

## file       Makefile.am

## The SUBDIRS variable holds a list of subdirectories automake (recursivly)
## must build.
SUBDIRS = $(subdirs) $(ax_config_subdirs)

## Options passed to aclocal to specify local macro directories. Although
## ACLOCAL_AMFLAGS should be replaced by AC_CONFIG_MACRO_DIRS, aclocal-1.13.4
## doesn't honor the paths provided to AC_CONFIG_MACRO_DIRS. See
## https://debbugs.gnu.org/cgi/bugreport.cgi?bug=21352
ACLOCAL_AMFLAGS = -I build-config/m4 -I vendor/common/build-config/m4

include vendor/common/build-config/Makefile.inc.am
include doc/Makefile.inc.am
include vendor/common/build-config/platforms.am

## Still, sometimes there are files that must be distributed, but which are not
## covered in the automatic rules. These files should be listed in the
## EXTRA_DIST variable. You can mention files from subdirectories in EXTRA_DIST.
##
## You can also mention a directory in EXTRA_DIST. In this case the entire
## directory will be recursively copied into the distribution. Please note that
## this will also copy everything in the directory, including CVS/RCS version
## control files. We recommend against using this feature.
EXTRA_DIST += $(srcdir)/src/config.h                                     \
              $(srcdir)/src/config-custom.h                              \
              $(srcdir)/test/replacePaths_c.awk                          \
              $(srcdir)/test/replacePaths_f77.awk                        \
              $(srcdir)/test/replacePaths_f90.awk                        \
              $(srcdir)/doc/example/example.c                            \
              $(srcdir)/doc/example/example.f90                          \
              $(srcdir)/doc/example_user_instrumentation/example_user_instrumentation.c   \
              $(srcdir)/doc/example_user_instrumentation/example_user_instrumentation.f90 \
              $(srcdir)/doc/doxygen-user/opari2_usage.dox.in             \
              $(srcdir)/doc/doxygen-user/opari2-config_usage.dox.in      \
              $(srcdir)/doc/doxygen-dev/doxygen-dev.cfg.in               \
              $(srcdir)/test/data                                        \
              $(srcdir)/THANKS                                           \
              $(srcdir)/OPEN_ISSUES

if ! CROSS_BUILD
EXTRA_DIST += $(srcdir)/build-frontend/configure.ac \
              $(srcdir)/build-frontend/aclocal.m4   \
              $(srcdir)/build-frontend/Makefile.am  \
              $(srcdir)/build-frontend/Makefile.in  \
              $(srcdir)/build-frontend/configure    \
              $(srcdir)/src/config-frontend.h.in
endif

doc_DATA = $(srcdir)/OPEN_ISSUES \
           $(srcdir)/ChangeLog

CLEANFILES += doc/doxygen-user/main.dox

.PHONY: doc/doxygen-user/main.dox remove-svn-files-from-dist

if SVN_CONTROLLED
doc/doxygen-user/main.dox: doc/doxygen-user/main_user_usage.dox doc/doxygen-user/main_user_linking.dox
	$(MKDIR_P) doc/doxygen-user
	cat $(srcdir)/doc/doxygen-user/main_user_introduction.dox.in \
            $(srcdir)/doc/doxygen-user/main_user_summary.dox.in \
	    $(srcdir)/doc/doxygen-user/main_user_install.dox.in \
	    doc/doxygen-user/main_user_usage.dox \
            $(srcdir)/doc/doxygen-user/main_user_ctc-decoding.dox.in \
            doc/doxygen-user/main_user_linking.dox \
            $(srcdir)/doc/doxygen-user/main_user_pomp-instrumentation.dox.in \
            $(srcdir)/doc/doxygen-user/main_user_example.dox.in \
            $(srcdir)/doc/doxygen-user/main_user_news.dox.in \
            $(srcdir)/doc/doxygen-user/main_user_finalization.dox.in \
	    > doc/doxygen-user/main.dox

## The developer documentation is compiled of the user documentation
## and additional information for developers
doc/doxygen-dev/main.dox: doc/doxygen-user/main_user_usage.dox doc/doxygen-user/main_user_linking.dox
	$(MKDIR_P) doc/doxygen-dev
	cat $(srcdir)/doc/doxygen-user/main_user_introduction.dox.in \
            $(srcdir)/doc/doxygen-dev/main_dev.dox.in \
            $(srcdir)/doc/doxygen-user/main_user_summary.dox.in \
	    $(srcdir)/doc/doxygen-user/main_user_install.dox.in \
	    doc/doxygen-user/main_user_usage.dox \
            $(srcdir)/doc/doxygen-user/main_user_ctc-decoding.dox.in \
            doc/doxygen-user/main_user_linking.dox \
            $(srcdir)/doc/doxygen-user/main_user_pomp-instrumentation.dox.in \
            $(srcdir)/doc/doxygen-user/main_user_example.dox.in \
            $(srcdir)/doc/doxygen-user/main_user_news.dox.in \
	    $(srcdir)/doc/doxygen-dev/main_dev_parser.dox.in \
	    $(srcdir)/doc/doxygen-dev/main_dev_new-paradigm.dox.in \
            $(srcdir)/doc/doxygen-user/main_user_finalization.dox.in \
	    > doc/doxygen-dev/main.dox

doc/doxygen-user/main_user_usage.dox:
	$(MKDIR_P) doc/doxygen-user
	cat $(srcdir)/doc/doxygen-user/main_user_usage.dox.in \
            $(srcdir)/doc/doxygen-user/opari2_usage.dox.in \
            $(srcdir)/doc/doxygen-user/main_user_usage-end.dox.in \
	    > doc/doxygen-user/main_user_usage.dox

doc/doxygen-user/main_user_linking.dox:
	$(MKDIR_P) doc/doxygen-user
	cat $(srcdir)/doc/doxygen-user/main_user_linking.dox.in \
            $(srcdir)/doc/doxygen-user/opari2-config_usage.dox.in \
	    > doc/doxygen-user/main_user_linking.dox
	echo "\endverbatim" >> doc/doxygen-user/main_user_linking.dox
else
doc/doxygen-user/main.dox:
#	touch doc/doxygen-user/main.dox

doc/doxygen-dev/main.dox:
#	touch doc/doxygen-dev/main.dox
endif

doxygen-dev: doc/doxygen-dev/main.dox
	$(MKDIR_P) doxygen-dev-doc/opari2/tags
	$(DOXYGEN) doc/doxygen-dev.cfg

remove-svn-files-from-dist:
	files=`find $(top_distdir) -type d | grep "\.svn"`; \
        for i in $$files; do rm -rf $$i; done

dist-hook: remove-svn-files-from-dist prohibit-svn-files-in-dist doc/doxygen-user/main.dox doxygen-user-generate
	rm -f $(top_distdir)/build-frontend/opari2_usage.h
	rm -f $(top_distdir)/build-frontend/opari2-config_usage.h

clean-local: doxygen-user-clean
	rm -rf test/tmp

install-exec-local:
	$(MKDIR_P) $(DESTDIR)$(libexecdir)
	$(INSTALL_SCRIPT) $(builddir)/build-frontend/pomp2-parse-init-regions.awk $(DESTDIR)$(libexecdir)/

install-data-local: doxygen-user-install

uninstall-hook: doxygen-uninstall common-uninstall
	rm -f  $(DESTDIR)$(libexecdir)/pomp2-parse-init-regions.awk
