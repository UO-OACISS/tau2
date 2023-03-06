import os, signal

def my_test_kill():
    print(os.getpid())
    os.kill(os.getpid(), signal.SIGSEGV)

def read_list(in_list):
    for elem in in_list:
        if elem > len(in_list):
            my_test_kill()

my_list = [0,1,2,8,4,5]
read_list(my_list)
