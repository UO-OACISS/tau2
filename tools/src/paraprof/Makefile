##########################
#Author: Robert Bell

#Makefile creates the ParaProf jar file.
##########################

CONFIG_ARCH=x86_64
TAUROOT=/home/amorris/tau2

BASE_SOURCES=ApplicationManager.java DerivedMetrics.java DerivedMetricPanel.java \
             ParaProfTreeCellRenderer.java \
             HistogramWindow.java ParaProfTrial.java \
             ParaProfApplication.java SystemEvents.java \
             CallPathDrawObject.java CallPathTextWindow.java HelpWindow.java PrefSpacingPanel.java \
             CallPathTextWindowPanel.java Preferences.java FunctionBarChartWindow.java \
             ColorChooser.java ParaProfExperiment.java \
             ColorPair.java LedgerWindow.java \
             ParaProfFileFilter.java LedgerDataElement.java \
	     DataSorter.java CallGraphWindow.java \
             PPThread.java LedgerWindowPanel.java PPFunctionProfile.java \
             UserEventWindow.java DBConfiguration.java ParaProfMetric.java \
	     PPUserEventProfile.java LoadTrialProgressWindow.java ColorBar.java FunctionOrdering.java \
             ParaProfManagerWindow.java StatWindow.java SnapshotThreadWindow.java LoadTrialWindow.java \
             ParaProfManagerTableModel.java StatWindowPanel.java \
             GlobalDataWindow.java ParaProfTreeNodeUserObject.java \
	     ParaProfUtils.java ParaProfException.java ParaProfErrorDialog.java \
	     ColorDefaultsWindow.java ColorMap.java ColorMapWindow.java PreferencesWindow.java \
	     Searcher.java SearchPanel.java FunctionSelectorDialog.java WindowPlacer.java \
	     DataSorterWrapper.java FunctionFilterDialog.java SnapshotBreakdownWindow.java SnapshotControlWindow.java

VIS_SOURCES = ThreeDeeWindow.java ThreeDeeSettings.java ThreeDeeControlPanel.java  
OTHER_SRC = $(wildcard src/enums/*.java) $(wildcard src/interfaces/*.java) $(wildcard src/treetable/*.java) $(wildcard src/barchart/*.java) $(wildcard src/util/*.java) $(wildcard src/script/*.java) $(wildcard src/sourceview/*.java) $(wildcard src/tablemodel/*.java)

BASE_SRC=$(addprefix src/, $(BASE_SOURCES))
VIS_SRC=$(addprefix src/, $(VIS_SOURCES))

NEW_SOURCES=$(BASE_SRC) $(VIS_SRC) $(OTHER_SRC) src/ParaProfImageOutput.java src/JVMDependent.java 

OLD_SOURCES=$(BASE_SRC) $(OTHER_SRC) ./wrapper/ParaProfImageOutput.java \
	    ./wrapper/ThreeDeeWindow.java ./wrapper/JVMDependent.java

TOOLS_SRC=$(TAUROOT)/tools/src

CC=javac
CC-OLD=javac

NOW = $(shell date)


CP_CORE = $(TOOLS_SRC)/perfdmf/bin/perfdmf.jar:$(TOOLS_SRC)/contrib/jargs.jar:$(TOOLS_SRC)/contrib/batik-combined.jar:$(TOOLS_SRC)/common/bin/tau-common.jar:$(TOOLS_SRC)/contrib/jfreechart-0.9.21.jar:$(TOOLS_SRC)/contrib/jcommon-0.9.6.jar:$(TOOLS_SRC)/contrib/jython.jar
CP13 = $(CP_CORE):$(TOOLS_SRC)/contrib/jgraph-1.3.jar
CP14 = $(CP_CORE):$(TOOLS_SRC)/contrib/jgraph.jar:$(TOOLS_SRC)/vis/bin/vis.jar

new: $(TOOLS_SRC)/paraprof/bin/paraprof.jar

old: ../bin/paraprof-1.3.jar

$(TOOLS_SRC)/paraprof/bin/paraprof.jar: $(NEW_SOURCES)
	rm -rf classes
	mkdir classes
	sed "s/XXXXX/$(NOW)/" src/ParaProf.java > wrapper/ParaProf.java
	$(CC) -d ./classes -classpath $(CP14) $(NEW_SOURCES) wrapper/ParaProf.java
	jar -cvf bin/paraprof.jar -C ./classes edu
	rm -f $(TAUROOT)/$(CONFIG_ARCH)/lib/paraprof.jar
	cp $(TOOLS_SRC)/paraprof/bin/paraprof.jar $(TAUROOT)/$(CONFIG_ARCH)/lib

../bin/paraprof-1.3.jar: $(OLD_SOURCES) classes
	rm -rf classes
	mkdir classes
	sed "s/XXXXX/$(NOW)/" src/ParaProf.java > wrapper/ParaProf.java
	$(CC) -d ./classes -classpath $(CP13) $(OLD_SOURCES) wrapper/ParaProf.java
	jar -cvf bin/paraprof-1.3.jar -C ./classes edu
	rm -f $(TAUROOT)/$(CONFIG_ARCH)/lib/paraprof-1.3.jar
	cp $(TOOLS_SRC)/paraprof/bin/paraprof-1.3.jar $(TAUROOT)/$(CONFIG_ARCH)/lib

clean:
	rm -rf classes wrapper/ParaProf.java bin/paraprof.jar bin/paraprof-1.3.jar
