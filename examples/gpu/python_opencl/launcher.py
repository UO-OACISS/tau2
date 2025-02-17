import ctypes
import sys
import array

def do_launch():
    my_library = ctypes.CDLL("./libmatmult.so")

    my_library.entry.restype = ctypes.c_int
    my_library.entry.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]

    argc = len(sys.argv)
    list_of_strings = [s.encode('utf-8') for s in sys.argv]
    argv = (ctypes.c_char_p * argc)(*list_of_strings)
    result = my_library.entry(argc, argv)
    print ("matmult returned:", result)

def main():
    do_launch()

if __name__ == "__main__": main()
