
#***************************************************************************
#
#  Section 1: Directories.
#
#***************************************************************************
CLASS_DIR        = $(PERFDB_HOME)/classes
MAKE_DIR         = $(PERFDB_HOME)/make
JAR_DIR          = $(PERFDB_HOME)/jars

SOURCE_DIR       = $(PERFDB_HOME)/src
DESTINATION      = $(PERFDB_HOME)/classes


#***************************************************************************
#
#  Section 2. Tools and options.
#
#***************************************************************************

JAVA     = java
JAVAC    = javac
JAR      = jar
DEBUG    = jdb
DELETE   = rm -f
COPY     = cp
MAKEDIR  = mkdir.exe -p
PRINT    = @echo
CHMOD    = chmod.exe

SEP = :

EMPTY            =
SPACE            = $(EMPTY) $(EMPTY)

LOCAL_JARTMP     = $(patsubst %,$(JAR_DIR)/%,$(JARS))
LOCAL_JARLIST    = $(subst $(SPACE),$(SEP),$(LOCAL_JARTMP))

OTHER_JARTMP     = $(patsubst %,$(JAR_DIR)/%,$(JARS_3RDPARTY))
OTHER_JARLIST    = $(subst $(SPACE),$(SEP),$(OTHER_JARTMP))

SOURCEPATH       = $(SOURCE_DIR)
CLASSPATH        = $(CLASS_DIR)$(SEP)$(LOCAL_JARLIST)$(SEP)$(OTHER_JARLIST)

JAVAC_OPTIONS    = -g -d $(DESTINATION) -classpath $(CLASSPATH) -sourcepath $(SOURCEPATH) -deprecation
JAVA_OPTIONS     = -classpath $(CLASSPATH) 
JAR_OPTIONS      = -cvf

#***************************************************************************
#
#  Section 3. Rules and dependencies.
#
#  This section defines the exact rules for creating a target file from
#  a (set of) source file(s). 
#
#  target : depends
#    rule
#
#  target  - the parameter given to make: What to build
#  depends - file or other targets target depends on
#  rule    - how to create target
#  $(VAR)  - environment variable or variable defined above
#  $@      - Current target
#  $*      - Current target without extension
#  $< - Current dependency
#
#***************************************************************************

PACKAGE_LOC     = $(subst .,/,$(PACKAGE))
PACKAGE_DIR     = $(DESTINATION)/$(PACKAGE_LOC)
JAVA_FILES      = $(filter  %.java,$(SOURCE))
NONJAVA_FILES   = $(patsubst %.java,,$(SOURCE))
CLASS_FILES     = $(JAVA_FILES:%.java=$(PACKAGE_DIR)/%.class)
OTHER_FILES     = $(NONJAVA_FILES:%=$(PACKAGE_DIR)/%)
JAR_CONTENT_CMD = -C $(CLASS_DIR) .

# Make a list of all packages involved
ifdef PACKAGE
PACKAGE_LIST    = $(subst .,/,$(PACKAGE))
MAIN_CLASS      = $(MAIN)
MAIN_PACKAGE    = $(PACKAGE)
else
PACKAGE_LIST    = $(subst .,/,$(PACKAGES)) $(subst .,/,$(NODOC_PACKAGES))
endif

PLIST_CLEAN     = $(patsubst %,$(SOURCE_DIR)/%/.clean,$(PACKAGE_LIST))
PLIST_BUILD     = $(patsubst %,$(SOURCE_DIR)/%/.build,$(PACKAGE_LIST))


# Applied when make is called without targets. 
default : buildall jar


# Building a .class file from a .java file
# .java -> .class

$(PACKAGE_DIR)/%.class : $(SOURCE_DIR)/$(PACKAGE_LOC)/%.java
	$(JAVAC) $(JAVAC_OPTIONS) $< 

%.class : $(SOURCE_DIR)/$(PACKAGE_LOC)/%.java
	$(MAKE) -k $(PACKAGE_DIR)/$@

# Default behaviour within a package: Simply copy the object from src
# to classes. Note that the location of this rule is important. It must be after
# the package specifics.
$(PACKAGE_DIR)/% :: $(SOURCE_DIR)/$(PACKAGE_LOC)/%
#	$(MAKEDIR) $(PACKAGE_DIR)
	$(COPY) $< $@
	$(CHMOD) u+rw $@

all : $(CLASS_FILES) $(OTHER_FILES)

cleanall :
	$(DELETE) $(PACKAGE_DIR)/*.class $(OTHER_FILES) 


# Change ".clean" tag to "Makefile", thus call the package makefile which
# in turn recalls this makefile with target cleanall .
%.clean :
	$(MAKE) -k -f $(subst .clean,Makefile,$@) cleanall
	rm -f $(JAR_DIR)/perfdb.jar 


# for every package directory
clean : $(PLIST_CLEAN)
	$(PRINT) Done clean.


# Change ".build" tag to "Makefile", thus call the package makefile which
# in turn recalls this makefile with target all 
%.build :
	$(MAKE) -k -f $(subst .build,Makefile,$@) all


# for every package
buildall : $(PLIST_BUILD)		
	$(PRINT) Done build.


# Build a jar file. $* strips the last phony .JAR extension.
%.JAR :
	$(DELETE) $(JAR_DIR)/$*
	$(JAR) $(JAR_OPTIONS) \
		$(JAR_DIR)/$* $(JAR_CONTENT_CMD)


# Create given jar file by invoking its Makefile 
%.jar :
	$(MAKE) -k -f $(patsubst %,$(JAR_DIR)/Makefile.%,$@) $@.JAR


# Create all jar files 
jar : $(JARS)
	$(PRINT) Done jars.

# A combination of steps used for automatic building
complete : clean buildall jar 





