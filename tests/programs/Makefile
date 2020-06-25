PROGRAMS ?= $(wildcard [a-z]*)
COMPILER ?= gcc

PHONY: all

# Called from bin/buildbot.sh
all:
	@for a in $(PROGRAMS); do \
		if [ -d $(a) ]; then \
			echo "processing folder $$a"; \
			$(MAKE) -C $$a all; \
		fi; \
	done;
	@echo "Done!"

# Called from bin/buildbot.sh
cleanall:
	@for a in $(PROGRAMS); do \
		if [ -d $(a) ]; then \
			echo "processing folder $$a"; \
			$(MAKE) -C $$a clean; \
		fi; \
	done;
	@echo "Done!"

test:
	@for a in $(PROGRAMS); do \
		if [ -d $(a) ]; then \
			echo "processing folder $$a"; \
			$(MAKE) -C $$a ; \
		fi; \
	done;
	@echo "Done!"

clean:
	../bin/buildbot.sh $(COMPILER) clean

