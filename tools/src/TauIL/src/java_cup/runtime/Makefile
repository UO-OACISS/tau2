include ../../../Makefile.inc

SOURCES = lr_parser.java \
	  Scanner.java \
	  Symbol.java \
	  virtual_parse_stack.java 

.PHONEY : default all clean clean-emacs

default : all

all : $(SOURCES)
	$(JAVAC) $(SOURCES)

clean :

clean-emacs :
	$(RM) ./*~
