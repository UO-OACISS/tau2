##########################
#Author: Robert Bell

#Makefile creates the ParaProf jar file.
##########################

BASE_SOURCES=ApplicationManager.java PPMLPanel.java ParaProfTreeCellRenderer.java StaticMainWindowData.java \
             BinWindow.java ParaProf.java ParaProfTrial.java StaticMainWindowPanel.java \
             BinWindowPanel.java ParaProfApplication.java PrefColorPanel.java SystemEvents.java \
             CallPathDrawObject.java CallPathTextWindow.java HelpWindow.java PrefSpacingPanel.java \
             CallPathTextWindowPanel.java Preferences.java MappingDataWindow.java \
             ColorChooser.java MappingDataWindowPanel.java ParaProfExperiment.java ThreadDataWindow.java \
             ColorPair.java MappingLedgerWindow.java ParaProfImageInterface.java ParaProfImageOptionsPanel.java ParaProfImageFormatFileFilter.java \
             SMWThread.java ThreadDataWindowPanel.java MappingLedgerWindowPanel.java SMWThreadDataElement.java \
             UserEventWindow.java DBConfiguration.java Metric.java SavedPreferences.java UserEventWindowPanel.java \
             ParaProfManager.java StatWindow.java \
             ParaProfManagerTableModel.java StatWindowPanel.java ParaProfLisp.java ParaProfLispPrimitives.java \
             PPML.java StaticMainWindow.java ParaProfTreeNodeUserObject.java

NEW_SOURCES=$(BASE_SOURCES) FileList.java ParaProfImageOutput.java

OLD_SOURCES=$(BASE_SOURCES) ./wrapper/FileList.java ./wrapper/ParaProfImageOutput.java

TOOLS_SRC=../../../../../..

CC=javac
CC-OLD=javac

new: ../bin/ParaProf.jar

old: ../bin/ParaProf-Old.jar

../bin/ParaProf.jar: $(NEW_SOURCES)
	rm -f ./paraprof/*.class
	$(CC) -d ./classes -classpath $(TOOLS_SRC)/dms/dms.jar:$(TOOLS_SRC)/contrib/jatha.jar:$(TOOLS_SRC)/contrib/junit.jar $(NEW_SOURCES)
	cp ./resources/*.gif ./classes/edu/uoregon/tau/paraprof
	cd classes; jar -cvf ParaProf.jar edu
	mv ./classes/ParaProf.jar $(TOOLS_SRC)/paraprof/bin

../bin/ParaProf-Old.jar: $(OLD_SOURCES)
	rm -f ./paraprof/*.class
	$(CC) -d ./ -classpath ../../dms/dms.jar $(OLD_SOURCES)
	jar -cvf ParaProf-Old.jar paraprof
	mv ParaProf-Old.jar ../bin

clean:
	rm -f classes/edu/uoregon/tau/paraprof/*
	rm -f ../bin/ParaProf.jar
	rm -f ../bin/ParaProf-Old.jar
