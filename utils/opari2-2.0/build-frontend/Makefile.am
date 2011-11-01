# scorep-untilities--frontend

include ../build-includes/common.am

AM_CPPFLAGS += -DFRONTEND_BUILD -DCROSS_BUILD

include ../build-includes/frontend-only.am

CONFIG_HEADER_FRONTEND = ../src/opari2_config_tool_frontend.h

## ACLOCAL_AMFLAGS contains options to pass to aclocal when aclocal.m4 is to be
## rebuilt by make. This line is also used by autoreconf to run aclocal with
## suitable options, or by autopoint and gettextize to locate the place where
## Gettext's macros should be installed. So even if you do not really care about
## the rebuild rules, you should define ACLOCAL_AMFLAGS.
## Can't be moved to common.am
ACLOCAL_AMFLAGS = -I ../vendor/common/build-config/m4

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

##opari2-config-tool-local: $(builddir)/opari2_config.dat

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
endif
else
if OPENMP_SUPPORTED
installcheck-local:
	(cd $(DESTDIR)$(datadir)/@PACKAGE_NAME@/doc/example/ && $(MAKE) $(AM_MAKEFLAGS) c ) || exit 1;
endif

endif
