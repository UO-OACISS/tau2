include ../../../Makefile.inc

SPEC	= Lexer.flex
DEPS	= Lexer.java
SOURCES	= $(DEPS) \
	  Token.java \
	  StringToken.java \
	  DoubleToken.java

.PHONEY : default all generate clean clean-deps clean-emacs

default : all

all : $(SOURCES)
	$(JAVAC) $(SOURCES)

generate : $(SPEC)
	$(JFLEX) $(SPEC)

clean :

clean-deps : clean
	$(RM) $(DEPS)

