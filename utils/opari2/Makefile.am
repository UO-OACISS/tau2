
## The SUBDIRS variable holds a list of subdirectories automake (recursivly)
## must build.
SUBDIRS = $(subdirs) $(subdirs_custom)

## ACLOCAL_AMFLAGS contains options to pass to aclocal when aclocal.m4 is to be
## rebuilt by make. This line is also used by autoreconf to run aclocal with
## suitable options, or by autopoint and gettextize to locate the place where
## Gettext's macros should be installed. So even if you do not really care about
## the rebuild rules, you should define ACLOCAL_AMFLAGS.
ACLOCAL_AMFLAGS = -I vendor/common/build-config/m4

include vendor/common/build-config/Makefile.inc.am
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
	      $(srcdir)/doc/example/example.f90                            \
              $(srcdir)/doc/example_user_instrumentation/example_user_instrumentation.c       \
	      $(srcdir)/doc/example_user_instrumentation/example_user_instrumentation.f90       \
	      $(srcdir)/doc/doxygen-user/opari2_usage.dox.in             \
	      $(srcdir)/test/data


if ! CROSS_BUILD
EXTRA_DIST += $(srcdir)/build-frontend/configure.ac \
              $(srcdir)/build-frontend/aclocal.m4   \
              $(srcdir)/build-frontend/Makefile.am  \
              $(srcdir)/build-frontend/Makefile.in  \
              $(srcdir)/build-frontend/configure    \
              $(srcdir)/src/config-frontend.h.in
endif

CLEANFILES += doc/doxygen-user/main.dox

.PHONY: doc/doxygen-user/main.dox remove-svn-files-from-dist

if SVN_CONTROLLED
doc/doxygen-user/main.dox:
	$(MKDIR_P) doc/doxygen-user
	cat $(srcdir)/doc/doxygen-user/main_1.dox.in \
            $(srcdir)/doc/doxygen-user/opari2_usage.dox.in \
            $(srcdir)/doc/doxygen-user/main_2.dox.in > doc/doxygen-user/main.dox
else
doc/doxygen-user/main.dox:
	$(MKDIR_P) doc/doxygen-user
	touch doc/doxygen-user/main.dox
endif

remove-svn-files-from-dist:
	files=`find $(top_distdir) -type d | grep "\.svn"`; \
        for i in $$files; do rm -rf $$i; done

dist-hook: remove-svn-files-from-dist prohibit-svn-files-in-dist doc/doxygen-user/main.dox doxygen-user-generate
	rm -f $(top_distdir)/build-frontend/opari2_usage.h

clean-local: doxygen-user-clean
	rm -rf test/tmp

install-exec-local:
	$(INSTALL_SCRIPT) $(builddir)/build-frontend/pomp2-parse-init-regions.awk $(DESTDIR)$(bindir)/

install-data-local: doxygen-user-install

uninstall-hook: common-uninstall
	rm -f  $(DESTDIR)$(bindir)/pomp2-parse-init-regions.awk
