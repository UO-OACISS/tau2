/* ../src/config-backend.h.  Generated from config-backend.h.in by configure.  */
/* ../src/config-backend.h.in.  Generated from configure.ac by autoheader.  */

/* Define if building universal (internal helper macro) */
/* #undef AC_APPLE_UNIVERSAL_BUILD */

/* Name of the sub-build. */
#define AFS_PACKAGE_BUILD "backend"

/* Symbol name of the sub-build in upper case. */
#define AFS_PACKAGE_BUILD_NAME BACKEND

/* Symbol name of the sub-build in lower case. */
#define AFS_PACKAGE_BUILD_name backend

/* The package name usable as a symbol in upper case. */
#define AFS_PACKAGE_NAME SCOREP

/* Relative path to the top-level source directory. */
#define AFS_PACKAGE_SRCDIR "../../build-backend/../"

/* The package name usable as a symbol in lower case. */
#define AFS_PACKAGE_name scorep

/* Try to use colorful output for tests. */
#define CUTEST_USE_COLOR 1

/* Define to dummy `main' function (if any) required to link to the Fortran
   libraries. */
/* #undef F77_DUMMY_MAIN */

/* Define to a macro mangling the given C identifier (in lower and upper
   case), which must not contain underscores, for linking with Fortran. */
#define F77_FUNC(name,NAME) name ## _

/* As F77_FUNC, but for C identifiers containing underscores. */
#define F77_FUNC_(name,NAME) name ## _

/* Define to 1 if your Fortran compiler doesn't accept -c and -o together. */
/* #undef F77_NO_MINUS_C_MINUS_O */

/* Define to dummy `main' function (if any) required to link to the Fortran
   libraries. */
/* #undef FC_DUMMY_MAIN */

/* Define if F77 and FC dummy `main' functions are identical. */
/* #undef FC_DUMMY_MAIN_EQ_F77 */

/* Define to a macro mangling the given C identifier (in lower and upper
   case), which must not contain underscores, for linking with Fortran. */
#define FC_FUNC(name,NAME) name ## _

/* As FC_FUNC, but for C identifiers containing underscores. */
#define FC_FUNC_(name,NAME) name ## _

/* Define to 1 if your Fortran compiler doesn't accept -c and -o together. */
/* #undef FC_NO_MINUS_C_MINUS_O */

/* Makes C variable alignment consistent with Fortran */
#define FORTRAN_ALIGNED __attribute__((aligned (16)))

/* Name of var after mangled by the Fortran compiler. */
#define FORTRAN_MANGLED(var) var ## _

/* Define to 1 if you have the <arpa/inet.h> header file. */
#define HAVE_ARPA_INET_H 1

/* Compiler constructor support */
#define HAVE_BACKEND_COMPILER_CONSTRUCTOR_SUPPORT 1

/* Define if the compiler instrumentation needs the symbol table. */
#define HAVE_BACKEND_COMPILER_INSTRUMENTATION_NEEDS_SYMBOL_TABLE 1

/* Defined if cuda is available. */
#define HAVE_BACKEND_CUDA_SUPPORT 0

/* Defined if CUDA version is greater or equal 6.0. */
#define HAVE_BACKEND_CUDA_VERSION_GREATER_EQUAL_60 0

/* Defined if CUPTI activity asynchronous buffer handling is available. */
#define HAVE_BACKEND_CUPTI_ASYNC_SUPPORT 0

/* Define if cplus_demangle is available. */
#define HAVE_BACKEND_DEMANGLE 1

/* Defined if dynamic linking via dlfcn.h is supported. */
#define HAVE_BACKEND_DLFCN_SUPPORT 1

/* Defined if GCC plug-in support is available. */
#define HAVE_BACKEND_GCC_PLUGIN_SUPPORT 0

/* Defined if getrusage() is available. */
#define HAVE_BACKEND_GETRUSAGE 1

/* Defined if the linker is GNU ld. */
#define HAVE_BACKEND_GNU_LINKER 1

/* Define if libbfd is available. */
#define HAVE_BACKEND_LIBBFD 1

/* Define if memory tracking is supported. */
#define HAVE_BACKEND_MEMORY_SUPPORT 1

/* Defined if metric perf support is available. */
#define HAVE_BACKEND_METRIC_PERF 1

/* Defined if MPI support is available. */
#define HAVE_BACKEND_MPI_SUPPORT 0

/* Define if nm is available. */
#define HAVE_BACKEND_NM 1

/* Defined if online access is possible. */
#define HAVE_BACKEND_ONLINE_ACCESS 0

/* OPARI2 revision used for version-dependent feature activation. */
/* #undef HAVE_BACKEND_OPARI2_REVISION */

/* Defined if openacc.h has been found and OpenACC enabled. */
#define HAVE_BACKEND_OPENACC 1

/* Defined if openacc.h and acc_prof.h have been found and OpenACC enabled */
#define HAVE_BACKEND_OPENACC_SUPPORT 0

/* Defined if OpenCL is available. */
#define HAVE_BACKEND_OPENCL_SUPPORT 1

/* Defined if OpenCL API version 1.0 is supported. */
#define HAVE_BACKEND_OPENCL_VERSION_1_0_SUPPORT 1

/* Defined if OpenCL API version 1.1 is supported. */
#define HAVE_BACKEND_OPENCL_VERSION_1_1_SUPPORT 1

/* Defined if OpenCL API version 1.2 is supported. */
#define HAVE_BACKEND_OPENCL_VERSION_1_2_SUPPORT 1

/* Defined if OpenCL API version 2.0 is supported. */
#define HAVE_BACKEND_OPENCL_VERSION_2_0_SUPPORT 1

/* Defined if libpapi is available. */
#define HAVE_BACKEND_PAPI 0

/* Defined if PMI is available. */
#define HAVE_BACKEND_PMI 0

/* Defined if sampling support is available. */
#define HAVE_BACKEND_SAMPLING_SUPPORT 0

/* Timer aix available */
/* #undef HAVE_BACKEND_SCOREP_TIMER_AIX */

/* Timer bgl available */
/* #undef HAVE_BACKEND_SCOREP_TIMER_BGL */

/* Timer bgp available */
/* #undef HAVE_BACKEND_SCOREP_TIMER_BGP */

/* Timer bgq available */
/* #undef HAVE_BACKEND_SCOREP_TIMER_BGQ */

/* Timer clock_gettime available */
#define HAVE_BACKEND_SCOREP_TIMER_CLOCK_GETTIME 1

/* The clk_id as string used in clock_gettime calls. */
#define HAVE_BACKEND_SCOREP_TIMER_CLOCK_GETTIME_CLK_ID_NAME "CLOCK_MONOTONIC_RAW"

/* Default timer */
#define HAVE_BACKEND_SCOREP_TIMER_DEFAULT "tsc"

/* Timer gettimeofday available */
#define HAVE_BACKEND_SCOREP_TIMER_GETTIMEOFDAY 1

/* Timer mingw available */
/* #undef HAVE_BACKEND_SCOREP_TIMER_MINGW */

/* Timer tsc available */
#define HAVE_BACKEND_SCOREP_TIMER_TSC 1

/* TSC timer */
#define HAVE_BACKEND_SCOREP_TIMER_TSC_NAME "X86_64"

/* Defined if SHMEM Profiling Interface support is available. */
#define HAVE_BACKEND_SHMEM_PROFILING_INTERFACE 0

/* Defined if SHMEM support is available. */
#define HAVE_BACKEND_SHMEM_SUPPORT 0

/* Defined if thread local storage support is available. */
#define HAVE_BACKEND_THREAD_LOCAL_STORAGE 1

/* Defined if unwinding support is available. */
#define HAVE_BACKEND_UNWINDING_SUPPORT 0

/* Define to 1 if you have the <bfd.h> header file. */
#define HAVE_BFD_H 1

/* Can link a close function */
#define HAVE_CLOSE 1

/* Define to 1 if you have the <CL/cl.h> header file. */
#define HAVE_CL_CL_H 1

/* Defined if configured to use Cobi. */
#define HAVE_COBI 0

/* Compiler constructor support */
#define HAVE_COMPILER_CONSTRUCTOR_SUPPORT 1

/* Define if the compiler instrumentation needs the symbol table. */
#define HAVE_COMPILER_INSTRUMENTATION_NEEDS_SYMBOL_TABLE 1

/* Define to 1 if you have the <ctype.h> header file. */
#define HAVE_CTYPE_H 1

/* Define to 1 if you have the <cuda.h> header file. */
/* #undef HAVE_CUDA_H */

/* Define to 1 if you have the <cuda_runtime_api.h> header file. */
/* #undef HAVE_CUDA_RUNTIME_API_H */

/* Defined if cuda is available. */
#define HAVE_CUDA_SUPPORT 0

/* Defined if CUDA version is greater or equal 6.0. */
#define HAVE_CUDA_VERSION_GREATER_EQUAL_60 0

/* Defined if CUPTI activity asynchronous buffer handling is available. */
#define HAVE_CUPTI_ASYNC_SUPPORT 0

/* Define to 1 if you have the <cupti.h> header file. */
/* #undef HAVE_CUPTI_H */

/* Define to 1 if you have the declaration of `close', and to 0 if you don't.
   */
#define HAVE_DECL_CLOSE 1

/* Define to 1 if you have the declaration of `fileno', and to 0 if you don't.
   */
#define HAVE_DECL_FILENO 1

/* Define to 1 if you have the declaration of `fseeko', and to 0 if you don't.
   */
#define HAVE_DECL_FSEEKO 1

/* Define to 1 if you have the declaration of `fseeko64', and to 0 if you
   don't. */
#define HAVE_DECL_FSEEKO64 0

/* Define to 1 if you have the declaration of `getcwd', and to 0 if you don't.
   */
#define HAVE_DECL_GETCWD 1

/* Define to 1 if you have the declaration of `gethostid', and to 0 if you
   don't. */
#define HAVE_DECL_GETHOSTID 1

/* Define to 1 if you have the declaration of `gethostname', and to 0 if you
   don't. */
#define HAVE_DECL_GETHOSTNAME 1

/* Define to 1 if you have the declaration of
   `PERF_COUNT_HW_STALLED_CYCLES_BACKEND', and to 0 if you don't. */
#define HAVE_DECL_PERF_COUNT_HW_STALLED_CYCLES_BACKEND 1

/* Define to 1 if you have the declaration of
   `PERF_COUNT_HW_STALLED_CYCLES_FRONTEND', and to 0 if you don't. */
#define HAVE_DECL_PERF_COUNT_HW_STALLED_CYCLES_FRONTEND 1

/* Define to 1 if you have the declaration of
   `PERF_COUNT_SW_ALIGNMENT_FAULTS', and to 0 if you don't. */
#define HAVE_DECL_PERF_COUNT_SW_ALIGNMENT_FAULTS 1

/* Define to 1 if you have the declaration of
   `PERF_COUNT_SW_EMULATION_FAULTS', and to 0 if you don't. */
#define HAVE_DECL_PERF_COUNT_SW_EMULATION_FAULTS 1

/* Define to 1 if you have the declaration of `read', and to 0 if you don't.
   */
#define HAVE_DECL_READ 1

/* Define to 1 if you have the declaration of `RTLD_LAZY', and to 0 if you
   don't. */
#define HAVE_DECL_RTLD_LAZY 1

/* Define to 1 if you have the declaration of `RTLD_LOCAL', and to 0 if you
   don't. */
#define HAVE_DECL_RTLD_LOCAL 1

/* Define to 1 if you have the declaration of `RTLD_NOW', and to 0 if you
   don't. */
#define HAVE_DECL_RTLD_NOW 1

/* Define if cplus_demangle is available. */
#define HAVE_DEMANGLE 1

/* Define to 1 if you have the <dlfcn.h> header file. */
#define HAVE_DLFCN_H 1

/* Defined if dynamic linking via dlfcn.h is supported. */
#define HAVE_DLFCN_SUPPORT 1

/* Can link a fileno function */
#define HAVE_FILENO 1

/* Can link a fseeko function */
#define HAVE_FSEEKO 1

/* Can link a fseeko64 function */
/* #undef HAVE_FSEEKO64 */

/* Defined if GCC plug-in support is available. */
#define HAVE_GCC_PLUGIN_SUPPORT 0

/* Can link a getcwd function */
#define HAVE_GETCWD 1

/* Can link a gethostid function */
#define HAVE_GETHOSTID 1

/* Can link a gethostname function */
#define HAVE_GETHOSTNAME 1

/* Define to 1 if you have the `gethrtime' function. */
/* #undef HAVE_GETHRTIME */

/* Defined if getrusage() is available. */
#define HAVE_GETRUSAGE 1

/* Defined if the linker is GNU ld. */
#define HAVE_GNU_LINKER 1

/* Define to 1 if hrtime_t is defined in <sys/time.h> */
/* #undef HAVE_HRTIME_T */

/* Define to 1 if you have the <intrinsics.h> header file. */
/* #undef HAVE_INTRINSICS_H */

/* Define to 1 if you have the <inttypes.h> header file. */
#define HAVE_INTTYPES_H 1

/* Define if libbfd is available. */
#define HAVE_LIBBFD 1

/* Defined if libcuda is available. */
/* #undef HAVE_LIBCUDA */

/* Defined if libcudart is available. */
/* #undef HAVE_LIBCUDART */

/* Defined if libcupti is available. */
/* #undef HAVE_LIBCUPTI */

/* Define to 1 if you have the `dl' library (-ldl). */
#define HAVE_LIBDL 1

/* Defined if libOpenCL is available. */
#define HAVE_LIBOPENCL 1

/* Defined if libpmi is available. */
/* #undef HAVE_LIBPMI */

/* Defined if librca is available. */
/* #undef HAVE_LIBRCA */

/* Defined if libunwind is available. */
/* #undef HAVE_LIBUNWIND */

/* Define to 1 if you have the <libunwind.h> header file. */
/* #undef HAVE_LIBUNWIND_H */

/* Define to 1 if you have the <linux/perf_event.h> header file. */
#define HAVE_LINUX_PERF_EVENT_H 1

/* Define to 1 if you have the `mach_absolute_time' function. */
/* #undef HAVE_MACH_ABSOLUTE_TIME */

/* Define to 1 if you have the <mach/mach_time.h> header file. */
/* #undef HAVE_MACH_MACH_TIME_H */

/* Define to 1 if you have the <memory.h> header file. */
#define HAVE_MEMORY_H 1

/* Define if memory tracking is supported. */
#define HAVE_MEMORY_SUPPORT 1

/* Defined if metric perf support is available. */
#define HAVE_METRIC_PERF 1

/* Defined to 1 if native MIC build exists */
/* #undef HAVE_MIC_SUPPORT */

/* Defined if MPI support is available. */
#define HAVE_MPI_SUPPORT 0

/* define if the compiler implements namespaces */
#define HAVE_NAMESPACES /**/

/* Define to 1 if you have the <netdb.h> header file. */
#define HAVE_NETDB_H 1

/* Define to 1 if you have the <netinet/in.h> header file. */
#define HAVE_NETINET_IN_H 1

/* Define if nm is available. */
#define HAVE_NM 1

/* Defined if online access is possible. */
#define HAVE_ONLINE_ACCESS 0

/* Defined if openacc.h has been found and OpenACC enabled. */
#define HAVE_OPENACC 1

/* Defined if openacc.h and acc_prof.h have been found and OpenACC enabled */
#define HAVE_OPENACC_SUPPORT 0

/* Define to 1 if you have the <OpenCL/opencl.h> header file. */
/* #undef HAVE_OPENCL_OPENCL_H */

/* Defined if OpenCL is available. */
#define HAVE_OPENCL_SUPPORT 1

/* Defined if OpenCL API version 1.0 is supported. */
#define HAVE_OPENCL_VERSION_1_0_SUPPORT 1

/* Defined if OpenCL API version 1.1 is supported. */
#define HAVE_OPENCL_VERSION_1_1_SUPPORT 1

/* Defined if OpenCL API version 1.2 is supported. */
#define HAVE_OPENCL_VERSION_1_2_SUPPORT 1

/* Defined if OpenCL API version 2.0 is supported. */
#define HAVE_OPENCL_VERSION_2_0_SUPPORT 1

/* Define if the used OTF2 library has zlib compression support. */
/* #undef HAVE_OTF2_COMPRESSION_ZLIB */

/* Define if the used OTF2 library has SIONlib support. */
/* #undef HAVE_OTF2_SUBSTRATE_SION */

/* Defined if libpapi is available. */
#define HAVE_PAPI 0

/* Set if we are building for the AIX platform */
/* #undef HAVE_PLATFORM_AIX */

/* Set if we are building for the ALTIX platform */
/* #undef HAVE_PLATFORM_ALTIX */

/* Set if we are building for the ARM platform */
/* #undef HAVE_PLATFORM_ARM */

/* Set if we are building for the BG/L platform */
/* #undef HAVE_PLATFORM_BGL */

/* Set if we are building for the BG/P platform */
/* #undef HAVE_PLATFORM_BGP */

/* Set if we are building for the BG/Q platform */
/* #undef HAVE_PLATFORM_BGQ */

/* Set if we are building for the Cray platform */
/* #undef HAVE_PLATFORM_CRAY */

/* Set if we are building for the Cray XC platform */
/* #undef HAVE_PLATFORM_CRAYXC */

/* Set if we are building for the Cray XE platform */
/* #undef HAVE_PLATFORM_CRAYXE */

/* Set if we are building for the Cray XK platform */
/* #undef HAVE_PLATFORM_CRAYXK */

/* Set if we are building for the Cray XT platform */
/* #undef HAVE_PLATFORM_CRAYXT */

/* Set if we are building for the FX10 platform */
/* #undef HAVE_PLATFORM_FX10 */

/* Set if we are building for the FX100 platform */
/* #undef HAVE_PLATFORM_FX100 */

/* Set if we are building for the K platform */
/* #undef HAVE_PLATFORM_K */

/* Set if we are building for the Linux platform */
#define HAVE_PLATFORM_LINUX 1

/* Set if we are building for the Mac platform */
/* #undef HAVE_PLATFORM_MAC */

/* Set if we are building for the Intel MIC platform */
/* #undef HAVE_PLATFORM_MIC */

/* Set if we are building for the MinGW platform */
/* #undef HAVE_PLATFORM_MINGW */

/* Set if we are building for the NEC SX platform */
/* #undef HAVE_PLATFORM_NECSX */

/* Set if we are building for the Solaris platform */
/* #undef HAVE_PLATFORM_SOLARIS */

/* Defined if PMI is available. */
#define HAVE_PMI 0

/* Define to 1 if you have the <pmi.h> header file. */
/* #undef HAVE_PMI_H */

/* Can link against popen and pclose */
#define HAVE_POPEN 1

/* Define if you have POSIX threads libraries and header files. */
/* #undef HAVE_PTHREAD */

/* Have PTHREAD_PRIO_INHERIT. */
#define HAVE_PTHREAD_PRIO_INHERIT 1

/* Can link a read function */
#define HAVE_READ 1

/* Define to 1 if you have the `readlink' function. */
#define HAVE_READLINK 1

/* Define to 1 if you have the `read_real_time' function. */
/* #undef HAVE_READ_REAL_TIME */

/* Defined if RUSAGE_THREAD is available. */
#define HAVE_RUSAGE_THREAD 1

/* Defined if struct member sigaction.sa_sigaction and type siginfo_t are
   available. */
/* #undef HAVE_SAMPLING_SIGACTION */

/* Defined if sampling support is available. */
#define HAVE_SAMPLING_SUPPORT 0

/* Define to 1 to enable additional debug output and checks. */
/* #undef HAVE_SCOREP_DEBUG */

/* Define to 1 to disable assertions (like NDEBUG). */
#define HAVE_SCOREP_NO_ASSERT 0

/* Define to 1 if PDT support available */
#define HAVE_SCOREP_PDT 1

/* If set, time measurements for Score-P's SCOREP_InitMeasurement() and
   scorep_finalize() functions are performed. */
/* #undef HAVE_SCOREP_RUNTIME_MANAGEMENT_TIMINGS */

/* Defined if SHMEM Profiling Interface support is available. */
#define HAVE_SHMEM_PROFILING_INTERFACE 0

/* Defined if SHMEM support is available. */
#define HAVE_SHMEM_SUPPORT 0

/* Define to 1 if you have the `sigaction' function. */
#define HAVE_SIGACTION 1

/* Define to 1 if you have the <signal.h> header file. */
#define HAVE_SIGNAL_H 1

/* define if the compiler has stringstream */
#define HAVE_SSTREAM /**/

/* Define to 1 if you have the <stdint.h> header file. */
#define HAVE_STDINT_H 1

/* Define to 1 if you have the <stdio.h> header file. */
#define HAVE_STDIO_H 1

/* Define to 1 if you have the <stdlib.h> header file. */
#define HAVE_STDLIB_H 1

/* Define to 1 if you have the <strings.h> header file. */
#define HAVE_STRINGS_H 1

/* Define to 1 if you have the <string.h> header file. */
#define HAVE_STRING_H 1

/* define if the compiler has strstream */
#define HAVE_STRSTREAM /**/

/* Define to 1 if you have the <sys/mman.h> header file. */
#define HAVE_SYS_MMAN_H 1

/* Define to 1 if you have the <sys/socket.h> header file. */
#define HAVE_SYS_SOCKET_H 1

/* Define to 1 if you have the <sys/stat.h> header file. */
#define HAVE_SYS_STAT_H 1

/* Define to 1 if you have the <sys/syscall.h> header file. */
#define HAVE_SYS_SYSCALL_H 1

/* Define to 1 if you have the <sys/time.h> header file. */
#define HAVE_SYS_TIME_H 1

/* Define to 1 if you have the <sys/types.h> header file. */
#define HAVE_SYS_TYPES_H 1

/* Defined if thread local storage support is available. */
#define HAVE_THREAD_LOCAL_STORAGE 1

/* Define to 1 if you have the `time_base_to_time' function. */
/* #undef HAVE_TIME_BASE_TO_TIME */

/* Define to 1 if you have the <unistd.h> header file. */
#define HAVE_UNISTD_H 1

/* Defined if unwinding support is available. */
#define HAVE_UNWINDING_SUPPORT 0

/* Define if you have the UNICOS _rtc() intrinsic. */
/* #undef HAVE__RTC */

/* Define to the sub-directory where libtool stores uninstalled libraries. */
#define LT_OBJDIR ".libs/"

/* Define to 1 if your C compiler doesn't accept -c and -o together. */
/* #undef NO_MINUS_C_MINUS_O */

/* Name of package */
#define PACKAGE "scorep"

/* Define to the address where bug reports for this package should be sent. */
#define PACKAGE_BUGREPORT "support@score-p.org"

/* The #include argument used to include this packages error codes header. */
#define PACKAGE_ERROR_CODES_HEADER <SCOREP_ErrorCodes.h>

/* Define to the full name of this package. */
#define PACKAGE_NAME "Score-P"

/* Define to the full name and version of this package. */
#define PACKAGE_STRING "Score-P 3.0"

/* Define to the one symbol short name of this package. */
#define PACKAGE_TARNAME "scorep"

/* Define to the home page for this package. */
#define PACKAGE_URL ""

/* Define to the version of this package. */
#define PACKAGE_VERSION "3.0"

/* Define to necessary symbol if this constant uses a non-standard name on
   your system. */
/* #undef PTHREAD_CREATE_JOINABLE */

/* Toplevel src directory */
#define SCOREP_ABS_TOPLEVEL_SRCDIR "/home/wspear/bin/tau2/scorep-3.0"

/* First guess, use the maximum of size(void*) and sizeof(double) as alignment
   for SCOREP_Allocator. */
#define SCOREP_ALLOCATOR_ALIGNMENT 8

/* Backend nm. */
/* #undef SCOREP_BACKEND_NM */

/* Defines path to Cobi. */
#define SCOREP_COBI_PATH ""

/* Revision of common repository */
#define SCOREP_COMMON_REVISION "2147"

/* Specifies how the compiler supports a constructor in C. */
#define SCOREP_COMPILER_CONSTRUCTOR_MODE SCOREP_COMPILER_CONSTRUCTOR_MODE_ATTRIBUTE

/* Attribute mode for compiler constructor in C. */
#define SCOREP_COMPILER_CONSTRUCTOR_MODE_ATTRIBUTE 0

/* Pragma mode for compiler constructor in C. */
#define SCOREP_COMPILER_CONSTRUCTOR_MODE_PRAGMA 1

/* Revision of Score-P */
#define SCOREP_COMPONENT_REVISION "11303"

/* Default name of the machine Score-P is running on. */
#define SCOREP_DEFAULT_MACHINE_NAME "Linux"

/* Defined if we are working from svn. */
/* #undef SCOREP_IN_DEVELOPEMENT */

/* Defined if we are working from a make dist generated tarball. */
#define SCOREP_IN_PRODUCTION /**/

/* Name of the platform Score-P was built on. */
#define SCOREP_PLATFORM_NAME "Linux"

/* Defined to RUSAGE_THREAD, if it is available, else to RUSAGE_SELF. */
#define SCOREP_RUSAGE_SCOPE RUSAGE_THREAD

/* Defined if we are building shared libraries. See also SCOREP_STATIC_BUILD
   */
#define SCOREP_SHARED_BUILD /**/

/* Defined if we are building static libraries. See also SCOREP_SHARED_BUILD
   */
#define SCOREP_STATIC_BUILD /**/

/* Set specifier to mark a variable as thread-local storage (TLS) */
#define SCOREP_THREAD_LOCAL_STORAGE_SPECIFIER __thread

/* The clk_id used in clock_gettime calls. */
#define SCOREP_TIMER_CLOCK_GETTIME_CLK_ID CLOCK_MONOTONIC_RAW

/* The size of `double', as computed by sizeof. */
#define SIZEOF_DOUBLE 8

/* The size of `int64_t', as computed by sizeof. */
#define SIZEOF_INT64_T 8

/* The size of `void *', as computed by sizeof. */
#define SIZEOF_VOID_P 8

/* Define to 1 if you have the ANSI C header files. */
#define STDC_HEADERS 1

/* Version number of package */
#define VERSION "3.0"

/* Define WORDS_BIGENDIAN to 1 if your processor stores words with the most
   significant byte first (like Motorola and SPARC, unlike Intel). */
#if defined AC_APPLE_UNIVERSAL_BUILD
# if defined __BIG_ENDIAN__
#  define WORDS_BIGENDIAN 1
# endif
#else
# ifndef WORDS_BIGENDIAN
/* #  undef WORDS_BIGENDIAN */
# endif
#endif

/* Define to 1 if `lex' declares `yytext' as a `char *' by default, not a
   `char[]'. */
#define YYTEXT_POINTER 1
