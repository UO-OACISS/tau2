include Makefile.inc

.PHONY : default all build install deps docs clean clean-deps clean-emacs

default : all

build : all
	$(MAKE) -C $(CLASSDIR) install

install : build

all : 
	$(MAKE) -C $(SOURCEDIR)

deps :
	$(MAKE) -C $(SOURCEDIR) deps

docs :
	$(JDOC) $(PACKAGES)

clean :
	$(MAKE) -C $(SOURCEDIR) clean
	$(MAKE) -C $(CLASSDIR) clean
	$(MAKE) -C tests clean
	$(RM) $(LIBDIR)/*.jar
	$(RM) -r $(APIDIR)/


clean-deps : clean
	$(MAKE) -C $(SOURCEDIR) clean-deps

clean-emacs : $(EMACS_CLEAN)
	$(MAKE) -C $(SOURCEDIR) clean-emacs
	$(MAKE) -C $(CLASSDIR) $(EMACS_CLEAN)
