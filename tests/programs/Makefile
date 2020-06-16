PROGRAMS ?= $(wildcard [a-z]*)

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
	../bin/buildbot.sh test

clean:
	../bin/buildbot.sh clean

