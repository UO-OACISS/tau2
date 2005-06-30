##########################
#Author: Robert Bell

#Makefile creates the ParaProf jar file.
##########################

CONFIG_ARCH=@TAUARCH@
TAUROOT=@TAUROOT@

BASE_SOURCES=ApplicationManager.java DerivedMetrics.java DerivedMetricPanel.java ParaProfTreeCellRenderer.java \
             HistogramWindow.java ParaProf.java ParaProfTrial.java FullDataWindowPanel.java \
             HistogramWindowPanel.java ParaProfApplication.java SystemEvents.java \
             CallPathDrawObject.java CallPathTextWindow.java HelpWindow.java PrefSpacingPanel.java \
             CallPathTextWindowPanel.java Preferences.java FunctionDataWindow.java \
             ColorChooser.java FunctionDataWindowPanel.java ParaProfExperiment.java ThreadDataWindow.java \
             ColorPair.java LedgerWindow.java ParaProfImageOptionsPanel.java \
             ParaProfImageFormatFileFilter.java LedgerDataElement.java DataSorter.java CallGraphWindow.java \
             PPThread.java ThreadDataWindowPanel.java LedgerWindowPanel.java PPFunctionProfile.java \
             UserEventWindow.java DBConfiguration.java ParaProfMetric.java \
	     UserEventWindowPanel.java PPUserEventProfile.java LoadTrialProgressWindow.java ColorBar.java \
             ParaProfManagerWindow.java StatWindow.java LoadTrialWindow.java \
             ParaProfManagerTableModel.java StatWindowPanel.java \
             FullDataWindow.java ParaProfTreeNodeUserObject.java \
	     ParaProfUtils.java ParaProfException.java ParaProfErrorDialog.java \
	     ColorDefaultsWindow.java ColorMap.java ColorMapWindow.java PreferencesWindow.java \
	     Searcher.java SearchPanel.java FunctionSelectorDialog.java

VIS_SOURCES = $(wildcard vis/*.java) ThreeDeeWindow.java ThreeDeeSettings.java ThreeDeeControlPanel.java  
OTHER_SOURCES = $(wildcard enums/*.java) $(wildcard interfaces/*.java) $(wildcard treetable/*.java)

NEW_SOURCES=$(BASE_SOURCES) $(VIS_SOURCES) $(OTHER_SOURCES) ParaProfImageOutput.java JVMDependent.java

OLD_SOURCES=$(BASE_SOURCES) $(OTHER_SOURCES) ./wrapper/ParaProfImageOutput.java ./wrapper/ThreeDeeWindow.java ./wrapper/JVMDependent.java

TOOLS_SRC=$(TAUROOT)/tools/src

CC=javac
CC-OLD=javac

NOW = $(shell date)

new: $(TOOLS_SRC)/paraprof/bin/ParaProf.jar

old: ../bin/ParaProf-Old.jar

$(TOOLS_SRC)/paraprof/bin/ParaProf.jar: $(NEW_SOURCES)
	/bin/mv ParaProf.java ParaProf.java.orig
	sed "s/XXXXX/$(NOW)/" ParaProf.java.orig > ParaProf.java
	rm -f ./paraprof/*.class
	$(CC) -d ./classes -classpath $(TOOLS_SRC)/dms/dms.jar:$(TOOLS_SRC)/contrib/jogl.jar:$(TOOLS_SRC)/contrib/jgraph.jar:$(TOOLS_SRC)/contrib/jatha.jar:$(TOOLS_SRC)/contrib/junit.jar:$(TOOLS_SRC)/contrib/jargs.jar $(NEW_SOURCES)
	cp ./resources/*.gif ./classes/edu/uoregon/tau/paraprof
	cd classes; jar -cvf ParaProf.jar edu
	mv ./classes/ParaProf.jar $(TOOLS_SRC)/paraprof/bin
	rm -f $(TAUROOT)/$(CONFIG_ARCH)/lib/ParaProf.jar
	cp $(TOOLS_SRC)/paraprof/bin/ParaProf.jar $(TAUROOT)/$(CONFIG_ARCH)/lib
	/bin/rm ParaProf.java
	/bin/mv ParaProf.java.orig ParaProf.java

../bin/ParaProf-Old.jar: $(OLD_SOURCES)
	/bin/mv ParaProf.java ParaProf.java.orig
	sed "s/XXXXX/$(NOW)/" ParaProf.java.orig > ParaProf.java
	rm -f ./paraprof/*.class
	$(CC) -d ./classes -classpath $(TOOLS_SRC)/dms/dms.jar:$(TOOLS_SRC)/contrib/jogl.jar:$(TOOLS_SRC)/contrib/jatha.jar:$(TOOLS_SRC)/contrib/junit.jar:$(TOOLS_SRC)/contrib/jargs.jar:$(TOOLS_SRC)/contrib/jgraph-1.3.jar $(OLD_SOURCES)
	cd classes; jar -cvf ParaProf-Old.jar edu
	mv ./classes/ParaProf-Old.jar $(TOOLS_SRC)/paraprof/bin
	rm -f $(TAUROOT)/$(CONFIG_ARCH)/lib/ParaProf-Old.jar
	cp $(TOOLS_SRC)/paraprof/bin/ParaProf-Old.jar $(TAUROOT)/$(CONFIG_ARCH)/lib
	/bin/rm ParaProf.java
	/bin/mv ParaProf.java.orig ParaProf.java

clean:
	rm -f classes/edu/uoregon/tau/paraprof/*.class
	rm -f $(TOOLS_SRC)/paraprof/bin/ParaProf.jar
	rm -f $(TOOLS_SRC)/paraprof/bin/ParaProf-Old.jar
