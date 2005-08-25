##########################
#Author: Robert Bell

#Makefile creates the ParaProf jar file.
##########################

CONFIG_ARCH=i386_linux
TAUROOT=/home/amorris/tau2

BASE_SOURCES=ApplicationManager.java DerivedMetrics.java DerivedMetricPanel.java ParaProfTreeCellRenderer.java \
             HistogramWindow.java wrapper/ParaProf.java ParaProfTrial.java \
             HistogramWindowPanel.java ParaProfApplication.java SystemEvents.java \
             CallPathDrawObject.java CallPathTextWindow.java HelpWindow.java PrefSpacingPanel.java \
             CallPathTextWindowPanel.java Preferences.java FunctionBarChartWindow.java \
             ColorChooser.java ParaProfExperiment.java \
             ColorPair.java LedgerWindow.java ParaProfImageOptionsPanel.java \
             ParaProfImageFormatFileFilter.java LedgerDataElement.java DataSorter.java CallGraphWindow.java \
             PPThread.java LedgerWindowPanel.java PPFunctionProfile.java \
             UserEventWindow.java DBConfiguration.java ParaProfMetric.java \
	     PPUserEventProfile.java LoadTrialProgressWindow.java ColorBar.java \
             ParaProfManagerWindow.java StatWindow.java LoadTrialWindow.java \
             ParaProfManagerTableModel.java StatWindowPanel.java \
             GlobalDataWindow.java ParaProfTreeNodeUserObject.java \
	     ParaProfUtils.java ParaProfException.java ParaProfErrorDialog.java \
	     ColorDefaultsWindow.java ColorMap.java ColorMapWindow.java PreferencesWindow.java \
	     Searcher.java SearchPanel.java FunctionSelectorDialog.java

VIS_SOURCES = ThreeDeeWindow.java ThreeDeeSettings.java ThreeDeeControlPanel.java  
OTHER_SOURCES = $(wildcard enums/*.java) $(wildcard interfaces/*.java) $(wildcard treetable/*.java) $(wildcard barchart/*.java)

NEW_SOURCES=$(BASE_SOURCES) $(VIS_SOURCES) $(OTHER_SOURCES) ParaProfImageOutput.java JVMDependent.java

OLD_SOURCES=$(BASE_SOURCES) $(OTHER_SOURCES) ./wrapper/ParaProfImageOutput.java ./wrapper/ThreeDeeWindow.java ./wrapper/JVMDependent.java

TOOLS_SRC=$(TAUROOT)/tools/src

CC=javac
CC-OLD=javac

NOW = $(shell date)

new: $(TOOLS_SRC)/paraprof/bin/ParaProf.jar

old: ../bin/ParaProf-Old.jar

wrapper/ParaProf.java : ParaProf.java
	sed "s/XXXXX/$(NOW)/" ParaProf.java > wrapper/ParaProf.java

$(TOOLS_SRC)/paraprof/bin/ParaProf.jar: $(NEW_SOURCES)
	rm -f ./paraprof/*.class
	$(CC) -d ./classes -classpath $(TOOLS_SRC)/dms/dms.jar::$(TOOLS_SRC)/contrib/jgraph.jar:$(TOOLS_SRC)/contrib/junit.jar:$(TOOLS_SRC)/contrib/jargs.jar:$(TOOLS_SRC)/vis/bin/vis.jar $(NEW_SOURCES)
	cp ./resources/*.gif ./classes/edu/uoregon/tau/paraprof
	cd classes; jar -cvf ParaProf.jar edu
	mv ./classes/ParaProf.jar $(TOOLS_SRC)/paraprof/bin
	rm -f $(TAUROOT)/$(CONFIG_ARCH)/lib/ParaProf.jar
	cp $(TOOLS_SRC)/paraprof/bin/ParaProf.jar $(TAUROOT)/$(CONFIG_ARCH)/lib

../bin/ParaProf-Old.jar: $(OLD_SOURCES)
	rm -f ./paraprof/*.class
	$(CC) -d ./classes -classpath $(TOOLS_SRC)/dms/dms.jar:$(TOOLS_SRC)/contrib/junit.jar:$(TOOLS_SRC)/contrib/jargs.jar:$(TOOLS_SRC)/contrib/jgraph-1.3.jar $(OLD_SOURCES)
	cd classes; jar -cvf ParaProf-Old.jar edu
	mv ./classes/ParaProf-Old.jar $(TOOLS_SRC)/paraprof/bin
	rm -f $(TAUROOT)/$(CONFIG_ARCH)/lib/ParaProf-Old.jar
	cp $(TOOLS_SRC)/paraprof/bin/ParaProf-Old.jar $(TAUROOT)/$(CONFIG_ARCH)/lib

clean:
	rm -f classes/edu/uoregon/tau/paraprof/*.class
	rm -f $(TOOLS_SRC)/paraprof/bin/ParaProf.jar
	rm -f $(TOOLS_SRC)/paraprof/bin/ParaProf-Old.jar
