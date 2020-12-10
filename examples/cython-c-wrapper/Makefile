LIB_DIR = lib

default: pyexamples

pyexamples: setup.py pyexamples.pyx $(LIB_DIR)/libexamples.so
	python3 setup.py build_ext --inplace && rm -f pyexamples.c && rm -Rf build

$(LIB_DIR)/libexamples.so:
	make -C $(LIB_DIR) libexamples.so
	ln -s $(LIB_DIR)/libexamples.so .

test: pyexamples
	tau_python -tau-python-interpreter=python3 -T python,serial ./main.py
	pprof -a

clean:
	rm -f *.o *.so profile.0.0.0
	make -C $(LIB_DIR) clean
