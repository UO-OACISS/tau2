##########################
#Author: Robert Bell

#Makefile creates the ParaProf jar file.
##########################

BASE_SOURCES=ApplicationManager.java GlobalMappingElement.java PPMLPanel.java ParaProfTreeCellRenderer.java StaticMainWindowData.java \
             BinWindow.java GlobalSortedMappingElement.java ParaProf.java ParaProfTrial.java StaticMainWindowPanel.java \
             BinWindowPanel.java GlobalThreadDataElement.java ParaProfApplication.java PrefColorPanel.java SystemEvents.java \
             CallPathDrawObject.java CallPathTextWindow.java HelpWindow.java ParaProfDBSession.java PrefSpacingPanel.java TauOutputSession.java \
             CallPathTextWindowPanel.java LineData.java ParaProfDataSession.java Preferences.java TauPprofOutputSession.java \
             CallPathUtilFuncs.java MappingDataWindow.java ParaProfError.java SMWContext.java Thread.java \
             ColorChooser.java MappingDataWindowPanel.java ParaProfExperiment.java SMWServer.java ThreadDataWindow.java \
             ColorPair.java MappingLedgerWindow.java ParaProfImageInterface.java ParaProfImageOptionsPanel.java ParaProfImageFormatFileFilter.java \
             SMWThread.java ThreadDataWindowPanel.java Context.java MappingLedgerWindowPanel.java SMWThreadDataElement.java \
             UserEventWindow.java DBConfiguration.java Metric.java ParaProfIterator.java SavedPreferences.java UserEventWindowPanel.java \
             DynaprofOutputSession.java NCT.java ParaProfManager.java StatWindow.java UtilFncs.java \
             Node.java ParaProfManagerTableModel.java StatWindowPanel.java Mapping.java\
             GlobalMapping.java PPML.java ParaProfObserver.java StaticMainWindow.java ParaProfTreeNodeUserObject.java

NEW_SOURCES=$(BASE_SOURCES) FileList.java ParaProfImageOutput.java

OLD_SOURCES=$(BASE_SOURCES) ./wrapper/FileList.java ./wrapper/ParaProfImageOutput.java

CC=javac
CC-OLD=javac

new: ../bin/ParaProf.jar

old: ../bin/ParaProf-Old.jar

../bin/ParaProf.jar: $(NEW_SOURCES)
	rm -f ./paraprof/*.class
	$(CC) -d ./ -classpath ../../dms/dms.jar $(NEW_SOURCES)
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
