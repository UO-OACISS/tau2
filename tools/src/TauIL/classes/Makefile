include ../Makefile.inc

.PHONEY : default install build-jars clean clean-emacs

default :

install : build-jars install-jars

build-jars : $(JARS)

install-jars :
	$(CP) $(patsubst %.jar,./%.jar, ${JARS}) $(LIBDIR)

$(JARS) :
	$(JAR) $@ ./$(subst .jar,,$@)

clean :
	$(RM) -r $(patsubst %.jar,./%,${JARS})
	$(RM) $(patsubst %,./%,${JARS})

clean-emacs :
	$(RM) ./*~
