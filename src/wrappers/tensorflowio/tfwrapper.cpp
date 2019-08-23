/****************************************************************************
**			TAU Portable Profiling Package                                 **
**			http://www.cs.uoregon.edu/research/tau                         **
*****************************************************************************
**    Copyright 2019  	        				   	                       **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
**    Forschungszentrum Juelich, ParaTools, Inc.                           **
****************************************************************************/
/****************************************************************************
**	File 		: tfwrapper.cpp 			        	                   **
**	Description 	: I/O wrapper library for TensorFlow I/O			   **
**  Author      : Nicholas Chaimov                                         **
**	Contact		: tau-bugs@cs.uoregon.edu               	               **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau           **
**                                                                         **
****************************************************************************/

#define _GNU_SOURCE
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/mman.h>
#if !defined(__APPLE__)
#include <sys/sendfile.h>
#endif
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <stdio.h>
#include <dlfcn.h>
#include <cstring>
#include <string>
#include <memory>
#include <atomic>

#include <TAU.h>

using namespace std;

void * Tau_get_tf_library() {
   static void * handle = NULL;
   static std::atomic<bool> first_time(true);
   if(first_time) {
      handle = dlopen("libtensorflow_framework.so", RTLD_LAZY | RTLD_LOCAL);
      first_time = false;
   }
   return handle;
}

namespace absl {
    class string_view;
};

namespace tensorflow {

    using StringPiece = absl::string_view;

    class Status {
        public:
        struct State {
            int32_t code;
            string msg;
        };
        std::unique_ptr<State> state_;
    };

    class RandomAccessFile {
        public:
        RandomAccessFile() {}
        virtual ~RandomAccessFile() {};
        RandomAccessFile (const RandomAccessFile&) = delete;
        RandomAccessFile& operator= (const RandomAccessFile&) = delete;            
        virtual Status Read(unsigned long long offset, size_t n, StringPiece* result, char* scratch) const {};
    };

    class PosixFileSystem {
        public:
        PosixFileSystem() {}
        ~PosixFileSystem() {}
        Status NewRandomAccessFile(const string& filename, std::unique_ptr<RandomAccessFile>* result);
    };

    class PosixRandomAccessFile : public RandomAccessFile {
        private:
        string filename_;
        int fd_;

        public:
        PosixRandomAccessFile(const string& fname, int fd)
            : filename_(fname), fd_(fd) {};
        ~PosixRandomAccessFile() { close(fd_); }

        Status Read(unsigned long long offset, size_t n, StringPiece* result, char* scratch) const;
    };

    Status PosixFileSystem::NewRandomAccessFile(const string& fname, std::unique_ptr<RandomAccessFile>* result) {

        typedef Status (PosixFileSystem::*origFuncType)(const string& fname, std::unique_ptr<RandomAccessFile>* result);
        static origFuncType origFunc = NULL;
        if(origFunc == NULL) {
            void * handle = Tau_get_tf_library();
            void * tmpPtr = dlsym(handle, "_ZN10tensorflow15PosixFileSystem19NewRandomAccessFileERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPSt10unique_ptrINS_16RandomAccessFileESt14default_deleteISA_EE");
            memcpy(&origFunc, &tmpPtr, sizeof(void*));    
        }

        TAU_PROFILE_TIMER(t, "tensorflow::Status tensorflow::PosixFileSystem::NewRandomAccessFile(const string &, std::unique_ptr<tensorflow::RandomAccessFile>*)", " ", TAU_IO);
        TAU_PROFILE_START(t);
        Status s;
        s = (this->*origFunc)(fname, result);
        TAU_PROFILE_STOP(t);
        return s;
    }

    Status PosixRandomAccessFile::Read(unsigned long long offset, size_t n, StringPiece* result, char* scratch) const {
        typedef Status (PosixRandomAccessFile::*origFuncType)(unsigned long long offset, size_t n, StringPiece* result, char* scratch) const;
        static origFuncType origFunc = NULL;
        if(origFunc == NULL) {
            void * handle = Tau_get_tf_library();
            fprintf(stderr, ">>>>>>>>>>>>>>>>>> HANDLE is %p\n", handle);
            void * tmpPtr = dlsym(handle, "_ZNK10tensorflow21PosixRandomAccessFile4ReadEymPN4absl11string_viewEPc");
            fprintf(stderr, ">>>>>>>>>>>>>>>>>> symbol addr is %p\n", tmpPtr);
            memcpy(&origFunc, &tmpPtr, sizeof(void*));
        }
        TAU_PROFILE_TIMER(t, "tensorflow::Status tensorflow::PosixRandomAccessFile::Read(unsigned long long, size_t, StringPiece*, char*)", " ", TAU_IO);
        TAU_PROFILE_START(t);
        Status s;
        s = (this->*origFunc)(offset, n, result, scratch);
        TAU_PROFILE_STOP(t);
        return s;
    }

}
