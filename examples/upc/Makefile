UPC=tau_upc.sh -tau_makefile=../../include/Makefile -optTrackUPCR -optTauSelectFile=select.tau #
#-g -optCompInst
APP=CAMEL
# APP=upctest

$(APP): $(APP).upc
	$(UPC) $(INCLUDE)  $< -o $@

clean:
	/bin/rm -rf $(APP).o $(APP) *.trc *.edf profile* MULT*
