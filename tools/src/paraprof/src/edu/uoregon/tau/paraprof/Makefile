##########################
#Author: Robert Bell

#Makefile creates the ParaProf jar file.
##########################
CONFIG_ARCH=@TAUARCH@
TAUROOT=@TAUROOT@

BASE_SOURCES=ApplicationManager.java PPMLPanel.java ParaProfTreeCellRenderer.java StaticMainWindowData.java \
             BinWindow.java ParaProf.java ParaProfTrial.java StaticMainWindowPanel.java \
             BinWindowPanel.java ParaProfApplication.java SystemEvents.java \
             CallPathDrawObject.java CallPathTextWindow.java HelpWindow.java PrefSpacingPanel.java \
             CallPathTextWindowPanel.java Preferences.java MappingDataWindow.java \
             ColorChooser.java MappingDataWindowPanel.java ParaProfExperiment.java ThreadDataWindow.java \
             ColorPair.java MappingLedgerWindow.java ParaProfImageInterface.java ParaProfImageOptionsPanel.java ParaProfImageFormatFileFilter.java \
             SMWThread.java ThreadDataWindowPanel.java MappingLedgerWindowPanel.java SMWThreadDataElement.java \
             UserEventWindow.java DBConfiguration.java Metric.java SavedPreferences.java UserEventWindowPanel.java \
             ParaProfManager.java StatWindow.java LoadTrialPanel.java \
             ParaProfManagerTableModel.java StatWindowPanel.java ParaProfLisp.java ParaProfLispPrimitives.java \
             PPML.java StaticMainWindow.java ParaProfTreeNodeUserObject.java Interpreter.java DisplayProperties.java

NEW_SOURCES=$(BASE_SOURCES) ParaProfImageOutput.java

OLD_SOURCES=$(BASE_SOURCES) ./wrapper/FileList.java ./wrapper/ParaProfImageOutput.java

TOOLS_SRC=$(TAUROOT)/tools/src

CC=javac
CC-OLD=javac

new: $(TOOLS_SRC)/paraprof/bin/ParaProf.jar

old: ../bin/ParaProf-Old.jar

$(TOOLS_SRC)/paraprof/bin/ParaProf.jar: $(NEW_SOURCES)
	rm -f ./paraprof/*.class
	$(CC) -d ./classes -classpath $(TOOLS_SRC)/dms/dms.jar:$(TOOLS_SRC)/contrib/jatha.jar:$(TOOLS_SRC)/contrib/junit.jar:$(TOOLS_SRC)/contrib/jargs.jar $(NEW_SOURCES)
	cp ./resources/*.gif ./classes/edu/uoregon/tau/paraprof
	cd classes; jar -cvf ParaProf.jar edu
	mv ./classes/ParaProf.jar $(TOOLS_SRC)/paraprof/bin
	rm -f $(TAUROOT)/$(CONFIG_ARCH)/lib/ParaProf.jar
	cp $(TOOLS_SRC)/paraprof/bin/ParaProf.jar $(TAUROOT)/$(CONFIG_ARCH)/lib

../bin/ParaProf-Old.jar: $(OLD_SOURCES)
	rm -f ./paraprof/*.class
	$(CC) -d ./classes -classpath $(TOOLS_SRC)/dms/dms.jar:$(TOOLS_SRC)/contrib/jatha.jar:$(TOOLS_SRC)/contrib/junit.jar:$(TOOLS_SRC)/contrib/jargs.jar $(OLD_SOURCES)
	cd classes; jar -cvf ParaProf-Old.jar edu
	mv ./classes/ParaProf-Old.jar $(TOOLS_SRC)/paraprof/bin
	rm -f $(TAUROOT)/$(CONFIG_ARCH)/lib/ParaProf-Old.jar
	cp $(TOOLS_SRC)/paraprof/bin/ParaProf-Old.jar $(TAUROOT)/$(CONFIG_ARCH)/lib

clean:
	rm -f classes/edu/uoregon/tau/paraprof/*.class
	rm -f $(TOOLS_SRC)/paraprof/bin/ParaProf.jar
	rm -f $(TOOLS_SRC)/paraprof/bin/ParaProf-Old.jar
