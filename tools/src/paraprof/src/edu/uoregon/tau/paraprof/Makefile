##########################
#Author: Robert Bell

#Makefile creates the ParaProf jar file.
##########################

CONFIG_ARCH=@TAUARCH@
TAUROOT=@TAUROOT@

BASE_SOURCES=ApplicationManager.java DerivedMetrics.java DerivedMetricPanel.java ParaProfTreeCellRenderer.java \
             HistogramWindow.java ParaProf.java ParaProfTrial.java StaticMainWindowPanel.java \
             HistogramWindowPanel.java ParaProfApplication.java SystemEvents.java \
             CallPathDrawObject.java CallPathTextWindow.java HelpWindow.java PrefSpacingPanel.java \
             CallPathTextWindowPanel.java Preferences.java FunctionDataWindow.java \
             ColorChooser.java FunctionDataWindowPanel.java ParaProfExperiment.java ThreadDataWindow.java \
             ColorPair.java LedgerWindow.java ParaProfImageInterface.java ParaProfImageOptionsPanel.java \
             ParaProfImageFormatFileFilter.java LedgerDataElement.java DataSorter.java CallGraphWindow.java \
             PPThread.java ThreadDataWindowPanel.java LedgerWindowPanel.java PPFunctionProfile.java \
             UserEventWindow.java DBConfiguration.java ParaProfMetric.java Preferences.java \
	     UserEventWindowPanel.java PPUserEventProfile.java LoadTrialProgressWindow.java ColorBar.java \
             ParaProfManagerWindow.java StatWindow.java LoadTrialWindow.java DataSourceThreadControl.java \
             ParaProfManagerTableModel.java StatWindowPanel.java ParaProfLisp.java ParaProfLispPrimitives.java \
             StaticMainWindow.java ParaProfTreeNodeUserObject.java Interpreter.java DisplayProperties.java \
	     ParaProfUtils.java ParaProfException.java ParaProfErrorDialog.java ParaProfObserver.java \
	     ColorDefaultsWindow.java ColorMap.java ColorMapWindow.java PreferencesWindow.java \
	     enums/CallGraphOption.java enums/SortType.java enums/UserEventValueType.java enums/ValueType.java

NEW_SOURCES=$(BASE_SOURCES) ParaProfImageOutput.java

OLD_SOURCES=$(BASE_SOURCES) ./wrapper/ParaProfImageOutput.java

TOOLS_SRC=$(TAUROOT)/tools/src

CC=javac
CC-OLD=javac

new: $(TOOLS_SRC)/paraprof/bin/ParaProf.jar

old: ../bin/ParaProf-Old.jar

$(TOOLS_SRC)/paraprof/bin/ParaProf.jar: $(NEW_SOURCES)
	rm -f ./paraprof/*.class
	$(CC) -d ./classes -classpath $(TOOLS_SRC)/dms/dms.jar:$(TOOLS_SRC)/contrib/jgraph.jar:$(TOOLS_SRC)/contrib/jatha.jar:$(TOOLS_SRC)/contrib/junit.jar:$(TOOLS_SRC)/contrib/jargs.jar $(NEW_SOURCES)
	cp ./resources/*.gif ./classes/edu/uoregon/tau/paraprof
	cd classes; jar -cvf ParaProf.jar edu
	mv ./classes/ParaProf.jar $(TOOLS_SRC)/paraprof/bin
	rm -f $(TAUROOT)/$(CONFIG_ARCH)/lib/ParaProf.jar
	cp $(TOOLS_SRC)/paraprof/bin/ParaProf.jar $(TAUROOT)/$(CONFIG_ARCH)/lib

../bin/ParaProf-Old.jar: $(OLD_SOURCES)
	rm -f ./paraprof/*.class
	$(CC) -d ./classes -classpath $(TOOLS_SRC)/dms/dms.jar:$(TOOLS_SRC)/contrib/jatha.jar:$(TOOLS_SRC)/contrib/junit.jar:$(TOOLS_SRC)/contrib/jargs.jar:$(TOOLS_SRC)/contrib/jgraph-1.3.jar $(OLD_SOURCES)
	cd classes; jar -cvf ParaProf-Old.jar edu
	mv ./classes/ParaProf-Old.jar $(TOOLS_SRC)/paraprof/bin
	rm -f $(TAUROOT)/$(CONFIG_ARCH)/lib/ParaProf-Old.jar
	cp $(TOOLS_SRC)/paraprof/bin/ParaProf-Old.jar $(TAUROOT)/$(CONFIG_ARCH)/lib

clean:
	rm -f classes/edu/uoregon/tau/paraprof/*.class
	rm -f $(TOOLS_SRC)/paraprof/bin/ParaProf.jar
	rm -f $(TOOLS_SRC)/paraprof/bin/ParaProf-Old.jar
