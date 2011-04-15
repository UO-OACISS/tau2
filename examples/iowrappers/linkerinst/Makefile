CC=g++
CXX=tau_cxx.sh -optQuiet
LD=$(CXX)
AR=ar
ARFLAGS=rcv
LDFLAGS=@bar_wrapper/link_options.tau
OBJS=bar_wrapper main.o foo.o libbar.a bar.o
app: $(OBJS) 
	$(LD) main.o foo.o -L. -lbar -o $@ $(LDFLAGS) 

bar_wrapper: bar.h libbar.a
	tau_gen_wrapper bar.h libbar.a

libwrapbar.a : libbar.a wrap.o
	$(AR) $(ARFLAGS) $@ wrap.o
libbar.a: bar.o
	$(AR) $(ARFLAGS) $@ $<
bar.o: bar.cpp
	echo "Compiling bar.cpp without any TAU instrumentation!"
	g++ -c bar.cpp
.cpp.o: 
	$(CXX) -c $<

clean:
	/bin/rm -rf $(OBJS) app
