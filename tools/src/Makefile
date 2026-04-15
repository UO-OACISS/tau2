# tools/src/Makefile — Top-level build for TAU Java profiling tools
#
# Works with or without running tau configure
#
# Usage:  make [JAVA_RELEASE=8|17|21|none] [target]
# Targets: all  common  vis  vis-jogl2  perfdmf  paraprof  perfexplorer  clean

JAVA_RELEASE ?= 8

# Resolve TOOLS_SRC from this Makefile's location, regardless of $(PWD)
TOOLS_SRC := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))

# ── TAU tree detection ────────────────────────────────────────────────────────
# Module Makefiles include ../../../include/Makefile (via TAU_MAKEFILE) to pick
# up TAUROOT and CONFIG_ARCH for their install-cp step.  When building
# standalone that file does not exist; suppress the include and redirect
# installs to a local staging directory (tools/src/lib/) instead.
_TAU_INC := $(TOOLS_SRC)/../../include/Makefile
# _SUB_FLAGS: passed to all module sub-makes.
# _PERF_FLAGS: passed to perfexplorer only — deliberately omits TAUROOT and
#   CONFIG_ARCH because those are baked into perfexplorer/Makefile via sed
#   substitution and must not be overridden by the standalone values.
ifneq ($(wildcard $(_TAU_INC)),)
  _SUB_FLAGS  := TAU_MAKEFILE=$(_TAU_INC) JAVA_RELEASE=$(JAVA_RELEASE)
  _PERF_FLAGS := JAVA_RELEASE=$(JAVA_RELEASE)
else
  _SUB_FLAGS  := TAU_MAKEFILE=/dev/null TAUROOT=$(TOOLS_SRC) CONFIG_ARCH=. \
                 JAVA_RELEASE=$(JAVA_RELEASE)
  _PERF_FLAGS := JAVA_RELEASE=$(JAVA_RELEASE)
  $(shell mkdir -p $(TOOLS_SRC)/lib)
endif

CONTRIB      := $(TOOLS_SRC)/contrib
EXTDEPS_ROOT  ?= $(realpath $(TOOLS_SRC)/../..)
EXTDEPS_SRC   := $(EXTDEPS_ROOT)/external_dependencies
_EXTDEPS_DONE := $(CONTRIB)/.pe-extdeps-done

.PHONY: all common vis vis-jogl2 perfdmf paraprof perfexplorer extdeps clean

all: common vis vis-jogl2 perfdmf paraprof perfexplorer

common:
	$(MAKE) -C $(TOOLS_SRC)/common $(_SUB_FLAGS)

vis: common
	$(MAKE) -C $(TOOLS_SRC)/vis $(_SUB_FLAGS)

vis-jogl2: common
	$(MAKE) -C $(TOOLS_SRC)/vis-jogl2 $(_SUB_FLAGS)

perfdmf: common
	$(MAKE) -C $(TOOLS_SRC)/perfdmf $(_SUB_FLAGS)

paraprof: perfdmf vis vis-jogl2
	$(MAKE) -C $(TOOLS_SRC)/paraprof $(_SUB_FLAGS)

# Setup bypass of tau configure to process perfexplorer/Makefile.skel
$(TOOLS_SRC)/perfexplorer/Makefile: $(TOOLS_SRC)/perfexplorer/bin/Makefile.skel
	sed -e 's,@TAUROOT@,$(EXTDEPS_ROOT),' \
	    -e 's,@TAUARCH@,tools/src/perfexplorer/bin,' \
	    -e 's,@TAUSHELL@,$(SHELL),' \
	    $< > $@

# ── extdeps ───────────────────────────────────────────────────────────────────
# Downloads/extracts PerfExplorer's external JARs (JBoss Rules + Weka) from the
# external_dependencies/ archives into contrib/. 
# re-run 'make extdeps' to force a refresh.

EXT_TGZ_URL := http://tau.uoregon.edu/ext.tgz

extdeps: $(_EXTDEPS_DONE)

# Ensure external_dependencies/ exists, downloading ext.tgz if necessary.
$(EXTDEPS_SRC):
	@echo "external_dependencies not found at $(EXTDEPS_ROOT)."
	@echo "Downloading $(EXT_TGZ_URL)..."
	tmp=$$(mktemp -d) && \
	    wget -q -O $$tmp/ext.tgz $(EXT_TGZ_URL) && \
	    tar -xzf $$tmp/ext.tgz -C $(EXTDEPS_ROOT) && \
	    rm -rf $$tmp

$(_EXTDEPS_DONE): $(EXTDEPS_SRC)
	@[ -f $(EXTDEPS_SRC)/PE2_jars.tgz ] || \
	    { echo "ERROR: $(EXTDEPS_SRC)/PE2_jars.tgz not found"; exit 1; }
	@[ -f $(EXTDEPS_SRC)/weka-3-6-1.zip ] || \
	    { echo "ERROR: $(EXTDEPS_SRC)/weka-3-6-1.zip not found"; exit 1; }
	@echo "Extracting PerfExplorer external dependencies to $(CONTRIB)/..."
	tmp=$$(mktemp -d) && \
	    tar -xzf $(EXTDEPS_SRC)/PE2_jars.tgz -C $$tmp && \
	    cp $$tmp/jbossrules/*.jar $(CONTRIB)/ && \
	    cp $$tmp/jbossrules/lib/*.jar $(CONTRIB)/ && \
	    rm -rf $$tmp
	cd $(CONTRIB) && unzip -jo $(EXTDEPS_SRC)/weka-3-6-1.zip \
	    'weka-3-6-1/weka.jar' && mv weka.jar weka-3-6-1.jar
	touch $@

perfexplorer: perfdmf vis $(TOOLS_SRC)/perfexplorer/Makefile extdeps
	mkdir -p $(TOOLS_SRC)/perfexplorer/bin/lib
	$(MAKE) -C $(TOOLS_SRC)/perfexplorer $(_PERF_FLAGS)

clean:
	$(MAKE) -C $(TOOLS_SRC)/common   clean $(_SUB_FLAGS)
	$(MAKE) -C $(TOOLS_SRC)/vis      clean $(_SUB_FLAGS)
	$(MAKE) -C $(TOOLS_SRC)/vis-jogl2 clean $(_SUB_FLAGS)
	$(MAKE) -C $(TOOLS_SRC)/perfdmf  clean $(_SUB_FLAGS)
	$(MAKE) -C $(TOOLS_SRC)/paraprof clean $(_SUB_FLAGS)
	-$(MAKE) -C $(TOOLS_SRC)/perfexplorer clean 2>/dev/null
	rm -f $(TOOLS_SRC)/perfexplorer/Makefile
	rm -rf $(TOOLS_SRC)/perfexplorer/bin/lib
	# To also remove extracted external JARs from contrib/: rm -f $(_EXTDEPS_DONE)
