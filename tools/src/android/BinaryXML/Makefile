include ../../../../include/Makefile

INSTALLDEST = $(TAU_BIN_DIR)

CC = $(TAU_CC)

CFLAGS = -Wall

targets := bxml

bxml-dep := bxml.o

install: $(targets)
	@echo Installing $? in $(INSTALLDEST)
	if [ -d $(INSTALLDEST) ] ; then true; \
	else mkdir $(INSTALLDEST) ;fi
	$(TAU_INSTALL) $? $(INSTALLDEST)


bxml: $(bxml-dep)

clean:
	rm -f $(targets) *.o
