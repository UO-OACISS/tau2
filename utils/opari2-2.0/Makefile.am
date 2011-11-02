
## The SUBDIRS variable holds a list of subdirectories automake (recursivly)
## must build.
SUBDIRS = $(subdirs) $(subdirs_custom)

## ACLOCAL_AMFLAGS contains options to pass to aclocal when aclocal.m4 is to be
## rebuilt by make. This line is also used by autoreconf to run aclocal with
## suitable options, or by autopoint and gettextize to locate the place where
## Gettext's macros should be installed. So even if you do not really care about
## the rebuild rules, you should define ACLOCAL_AMFLAGS.
ACLOCAL_AMFLAGS = -I vendor/common/build-config/m4

EXTRA_DIST =
CLEANFILES =

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
              $(srcdir)/build-config/VERSION                             \
              $(srcdir)/test/replacePaths_c.awk                          \
              $(srcdir)/test/replacePaths_f77.awk                        \
              $(srcdir)/test/replacePaths_f90.awk                        \
              $(srcdir)/doc/example/example.c                            \
	      $(srcdir)/doc/example/example.f                            \
	      $(srcdir)/src/opari/opari_user_2.dox                       \
	      $(srcdir)/test/data

include vendor/common/build-config/Makefile.inc.am

if ! CROSS_BUILD
EXTRA_DIST += $(srcdir)/build-frontend/configure.ac \
              $(srcdir)/build-frontend/aclocal.m4   \
              $(srcdir)/build-frontend/Makefile.am  \
              $(srcdir)/build-frontend/Makefile.in  \
              $(srcdir)/build-frontend/configure    \
              $(srcdir)/src/config-frontend.h.in
endif

DISTCLEANFILES = 

CLEANFILES += doc/opari2.dox

.PHONY: doc/opari2.dox remove-svn-files-from-dist

if SVN_CONTROLLED
doc/opari2.dox:
	cat $(srcdir)/src/opari/opari_user_1.dox \
            $(srcdir)/src/opari/opari_user_2.dox \
            $(srcdir)/src/opari/opari_user_3.dox > doc/opari2.dox
else
doc/opari2.dox:
	touch doc/opari2.dox
endif

remove-svn-files-from-dist:
	files=`find $(top_distdir) -type d | grep "\.svn"`; \
        for i in $$files; do rm -rf $$i; done

dist-hook: remove-svn-files-from-dist prohibit-svn-files-in-dist doc/opari2.dox doxygen-user-generate
	rm -f $(top_distdir)/build-frontend/opari_user_usage.h

clean-local: doxygen-user-clean

distclean-local:
	rm -rf test/

check-build: $(BUILT_SOURCES) all $(check_PROGRAMS)
	for dir in $(SUBDIRS); do                        \
            (cd $$dir && $(MAKE) $(AM_MAKEFLAGS) $@) || exit 1; \
        done

check-run: check-build
	for dir in $(SUBDIRS); do                        \
            (cd $$dir && $(MAKE) $(AM_MAKEFLAGS) $@) || exit 1; \
        done

install-exec-local:
	$(INSTALL_SCRIPT) $(builddir)/build-frontend/pomp2_parse_init_regions.awk $(DESTDIR)$(bindir)/

install-data-local: doxygen-user-install install-example

uninstall-local: doxygen-user-uninstall
	rm -rf $(DESTDIR)$(datadir)/@PACKAGE_NAME@
	rm -f  $(DESTDIR)$(bindir)/pomp2_parse_init_regions.awk

install-example:
	$(MKDIR_P) $(DESTDIR)$(datadir)/@PACKAGE_NAME@/doc/example
	$(INSTALL_DATA) $(abs_srcdir)/doc/example/example.*            $(DESTDIR)$(datadir)/@PACKAGE_NAME@/doc/example/
	$(INSTALL_DATA) $(abs_srcdir)/src/pomp-lib-dummy/pomp2_lib.h   $(DESTDIR)$(datadir)/@PACKAGE_NAME@/doc/example/
	$(INSTALL_DATA) $(abs_builddir)/doc/example/Makefile           $(DESTDIR)$(datadir)/@PACKAGE_NAME@/doc/example/
	$(INSTALL_DATA) $(abs_builddir)/build-frontend/.libs/libpomp.* $(DESTDIR)$(datadir)/@PACKAGE_NAME@/doc/example/
	rm -f $(DESTDIR)$(datadir)/@PACKAGE_NAME@/doc/example/libpomp.la
