cc =  cc
CPP = CC
LIBS = -L/local/apps/xforms/FORMS -lforms -lX11 -lm
INCLUDE = -I/local/apps/xforms/FORMS -I/research/parallel/mikek/tau2/include
FLAVOR = -n32 -g
OBJS = csUI.o csUI_cb.o 
TARGET     = csUI
OPTIONS = -O -w
#DEFINES = -DTAU_STDCXXLIB -DDEBUG
DEFINES = -DDEBUG
############################################

all: csUI

csUI:	$(OBJS)
	$(CPP) $(FLAVOR) $(OPTIONS) -o $(TARGET) $(OBJS) $(LIBS) $(DEFINES)

.C.o:
	@echo -n "Building $<...."
	@$(CPP) $(FLAVOR) $(INCLUDE) $(OPTIONS)  -c $< -o $@ $(DEFINES)
	@echo "...Built"


#############################################

clean:
	@rm -f *.o
	@echo "Everything has been cleaned master."

god:
	@echo "You're already here my lord."
