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
             PPML.java StaticMainWindow.java ParaProfTreeNodeUserObject.java XMLSupport.java

NEW_SOURCES=$(BASE_SOURCES) FileList.java ParaProfImageOutput.java

OLD_SOURCES=$(BASE_SOURCES) ./wrapper/FileList.java ./wrapper/ParaProfImageOutput.java

CC=javac
CC-OLD=javac

new: ../bin/ParaProf.jar

old: ../bin/ParaProf-Old.jar

../bin/ParaProf.jar: $(NEW_SOURCES)
	rm -f ./paraprof/*.class
	$(CC) -d ./ -classpath ../../dms/dms.jar:../../contrib/jatha.jar:../../contrib/junit.jar $(NEW_SOURCES)
	jar -cvf ParaProf.jar paraprof
	mv ParaProf.jar ../bin

../bin/ParaProf-Old.jar: $(OLD_SOURCES)
	rm -f ./paraprof/*.class
	$(CC) -d ./ -classpath ../../dms/dms.jar $(OLD_SOURCES)
	jar -cvf ParaProf-Old.jar paraprof
	mv ParaProf-Old.jar ../bin

clean:
	rm -f paraprof/*.class
	rm -f ../bin/ParaProf.jar
	rm -f ../bin/ParaProf-Old.jar
