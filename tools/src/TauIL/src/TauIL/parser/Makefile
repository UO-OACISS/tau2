include ../../../Makefile.inc

SPEC	= Parser.y
SOURCES = Parser.java sym.java

.PHONEY : default all generate clean clean-deps clean-emacs

default : all

all : $(SOURCES)
	$(JAVAC) $(SOURCES)

generate : $(SPEC)
	$(CUP) $(SPEC)

clean :

clean-deps : clean
	$(RM) $(SOURCES)

