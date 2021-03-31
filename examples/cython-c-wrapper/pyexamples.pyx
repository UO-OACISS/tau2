cdef extern from "examples.h":
    void hello_from_c_function(const char *name)

def py_hello_from_c(name: bytes) -> None:
    hello_from_c_function(name)
