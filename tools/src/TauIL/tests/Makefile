include ../Makefile.inc

.PHONY : default all test-lex test-reduce clean clean-emacs

default : all

all : test-lex test-reduce

test-lex :
	$(JAVA) $(LEXER).Lexer rule.example

test-reduce :
	$(JAVA) TauIL.jReduce -d -r rule.example -v

test-tool :
	$(JAVA) TauIL.Tool rule.example -i
	$(JAVA) TauIL.Tool rule.test -i > tool.out
	tau_reduce -v -r rules.prop -f pprof.dat -o tau2.sel > reduce.out
	sdiff tool.out reduce.out
	sdiff tau.sel tau2.sel

#	$(JAVA) TauIL.Tool
#	$(JAVA) TauIL.Tool -v
#	$(JAVA) TauIL.Tool -D debug=true -D coolbeans -D bug=no
#	$(JAVA) TauIL.Tool -D who=knew inst.sel
#	$(JAVA) TauIL.Tool inst.sel

clean :
	$(RM) *.sel *.out

clean-emacs : $(EMACS_CLEAN)

