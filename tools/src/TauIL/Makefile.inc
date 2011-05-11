TAU_HOME	= /home/users/sameer/tau2

TAUIL_HOME	= $(TAU_HOME)/tools/src/TauIL

TOOLDIR		= $(TAUIL_HOME)/tools
CUPDIR		= $(TOOLDIR)/CUP
JFLEXDIR	= $(TOOLDIR)/JFlex

PARAPROF_HOME	= $(TAU_HOME)/tools/src/paraprof/bin/ParaProf.jar
DMS_HOME	= $(TAU_HOME)/tools/src/perfdmf/bin/perfdmf.jar

SOURCEDIR	= $(TAUIL_HOME)/src
CLASSDIR	= $(TAUIL_HOME)/classes
LIBDIR		= $(TAUIL_HOME)/bin

SOURCEPATH	= $(SOURCEDIR)
CLASSPATH 	= $(CUPDIR):$(CLASSDIR):$(PARAPROF_HOME):$(DMS_HOME)

DOCDIR		= $(TAUIL_HOME)/docs
APIDIR		= $(DOCDIR)/api

JAVAC_OPT	= -d $(CLASSDIR) -classpath $(CLASSPATH) -sourcepath $(SOURCEPATH) -deprecation
JAVA_OPT	= -classpath $(CLASSPATH)
JAR_OPT		= -cvf
JDOC_OPT	= -d $(APIDIR) -sourcepath $(SOURCEPATH)

JAVAC		= javac $(JAVAC_OPT)
JAVA		= java $(JAVA_OPT)
JAR		= jar $(JAR_OPT)
JDOC		= javadoc $(JDOC_OPT)

RM		= rm -f
CP		= cp -f

CUPOPT		= -parser Parser

JFLEX		= $(JFLEXDIR)/bin/jflex
CUP		= $(JAVA) java_cup.Main $(CUPOPT)

CUPDEP		= java_cup.runtime

LEXER		= TauIL.lexer
PARSER		= TauIL.parser

PACKAGES	= $(CUPDEP) \
		  TauIL.util \
		  TauIL.error \
		  TauIL.absyn \
		  TauIL.interpreter \
		  $(LEXER) \
		  $(PARSER) \
		  TauIL

MAINJAR		= TauIL.jar

JARS		= java_cup.jar \
		  $(MAINJAR)




EMACS_CLEAN	= kill-emacs
.PHONEY : $(EMACS_CLEAN) first

first : default

kill-emacs :
	$(RM) ./*~ ./#*#
