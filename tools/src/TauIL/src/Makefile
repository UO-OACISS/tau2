include ../Makefile.inc

CLEAN_PACKAGES	= $(patsubst %,clean_%, ${PACKAGES})
CLEAN_EMACS	= $(patsubst %,emacs_%, ${PACKAGES})

.PHONY : default $(PACKAGES) deps generate_lexer generate_parser clean $(CLEAN_PACKAGES) clean-deps $(CLEAN_EMACS) clean-emacs

default : $(PACKAGES)

$(PACKAGES) :
	$(MAKE) -C $(subst .,/,$@)

deps : generate_parser generate_lexer

generate_lexer : 
	$(MAKE) -C $(subst .,/,${LEXER}) generate

generate_parser :
	$(MAKE) -C $(subst .,/,${PARSER}) generate

clean : $(CLEAN_PACKAGES)

$(CLEAN_PACKAGES) :
	$(MAKE) -C $(subst .,/,$(patsubst clean_%,%,$@)) clean

clean-deps : clean_$(CUPDEP)
	$(MAKE) -C $(subst .,/,${LEXER}) clean-deps
	$(MAKE) -C $(subst .,/,${PARSER}) clean-deps

clean-emacs : $(CLEAN_EMACS) $(EMACS_CLEAN)

$(CLEAN_EMACS) :
	$(MAKE) -C $(subst .,/,$(patsubst emacs_%,%,$@)) $(EMACS_CLEAN)

