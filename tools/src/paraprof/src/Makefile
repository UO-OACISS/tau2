
#Author: Robert Bell

#Makefile creates the ParaProf jar file.

CLASSDIR = ./paraprof

CLASSFILES = $(CLASSDIR)/Application.class               $(CLASSDIR)/ApplicationManager.class \
             $(CLASSDIR)/BinWindow.class                 $(CLASSDIR)/BinWindowPanel.class \
             $(CLASSDIR)/ColorChooser.class              $(CLASSDIR)/ColorPair.class \
             $(CLASSDIR)/Experiment.class                $(CLASSDIR)/GlobalContext.class \
             $(CLASSDIR)/GlobalMapping.class             $(CLASSDIR)/GlobalMappingElement.class \
             $(CLASSDIR)/GlobalServer.class              $(CLASSDIR)/GlobalThread.class \
             $(CLASSDIR)/GlobalThreadDataElement.class   $(CLASSDIR)/HelpWindow.class \
             $(CLASSDIR)/LocalPrefWindow.class           $(CLASSDIR)/MappingDataWindow.class \
             $(CLASSDIR)/MappingDataWindowPanel.class    $(CLASSDIR)/MappingLedgerWindow.class \
             $(CLASSDIR)/MappingLedgerWindowPanel.class  $(CLASSDIR)/MeanDataWindow.class \
             $(CLASSDIR)/MeanDataWindowPanel.class       $(CLASSDIR)/MeanTotalStatWindow.class \
             $(CLASSDIR)/MeanTotalStatWindowPanel.class  $(CLASSDIR)/ParaProf.class \
             $(CLASSDIR)/ParaProfManager.class           $(CLASSDIR)/PrefColorPanel.class \
             $(CLASSDIR)/PrefSpacingPanel.class          $(CLASSDIR)/Preferences.class \
             $(CLASSDIR)/SMWContext.class                $(CLASSDIR)/SMWMeanDataElement.class \
             $(CLASSDIR)/SMWServer.class                 $(CLASSDIR)/SMWThread.class \
             $(CLASSDIR)/SMWThreadDataElement.class      $(CLASSDIR)/SavedPreferences.class \
             $(CLASSDIR)/StaticMainWindow.class          $(CLASSDIR)/StaticMainWindowData.class \
             $(CLASSDIR)/StaticMainWindowPanel.class     $(CLASSDIR)/SystemEvents.class \
             $(CLASSDIR)/ThreadDataWindow.class          $(CLASSDIR)/ThreadDataWindowPanel.class \
             $(CLASSDIR)/TotalStatUEWindow.class         $(CLASSDIR)/TotalStatUEWindowPanel.class \
             $(CLASSDIR)/TotalStatWindow.class           $(CLASSDIR)/TotalStatWindowPanel.class \
             $(CLASSDIR)/Trial.class                     $(CLASSDIR)/UserEventWindow.class \
             $(CLASSDIR)/UserEventWindowPanel.class      $(CLASSDIR)/UtilFncs.class \
             $(CLASSDIR)/Value.class


default: ParaProf.jar

ParaProf.jar:
	javac *.java
	mv *.class ./paraprof
	jar -cmvf manifest.txt ParaProf.jar ParaProf

clean:
	rm -f jRacy.jar paraprof/*.class
