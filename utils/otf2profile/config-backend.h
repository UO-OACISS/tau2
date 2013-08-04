/* ../src/config-backend.h.  Generated from config-backend.h.in by configure.  */
/* ../src/config-backend.h.in.  Generated from configure.ac by autoheader.  */

/* Define if building universal (internal helper macro) */
/* #undef AC_APPLE_UNIVERSAL_BUILD */

/* Try to use colorful output for tests. */
#define CUTEST_USE_COLOR 1

/* Name of var after mangled by the Fortran compiler. */
#define FORTRAN_MANGLED(var) var ## _

/* Defined to 1 if the clock_gettime() function is available. */
#define HAVE_CLOCK_GETTIME 1

/* Define to 1 if you have the declaration of `fseeko', and to 0 if you don't.
   */
#define HAVE_DECL_FSEEKO 0

/* Define to 1 if you have the declaration of `fseeko64', and to 0 if you
   don't. */
#define HAVE_DECL_FSEEKO64 0

/* Define to 1 if you have the declaration of `getcwd', and to 0 if you don't.
   */
#define HAVE_DECL_GETCWD 1

/* Define to 1 if you have the declaration of `gethostname', and to 0 if you
   don't. */
#define HAVE_DECL_GETHOSTNAME 0

/* Define to 1 if you have the <dlfcn.h> header file. */
#define HAVE_DLFCN_H 1

/* Can link a fseeko function */
#define HAVE_FSEEKO 1

/* Can link a fseeko64 function */
/* #undef HAVE_FSEEKO64 */

/* Can link a getcwd function */
#define HAVE_GETCWD 1

/* Can link a gethostname function */
#define HAVE_GETHOSTNAME 1

/* Define to 1 if the getpid() function is available. */
#define HAVE_GETPID 1

/* Define to 1 if the gettimeofday() function is available. */
#define HAVE_GETTIMEOFDAY 1

/* Define to 1 if you have the <inttypes.h> header file. */
#define HAVE_INTTYPES_H 1

/* Define to 1 if you have the <memory.h> header file. */
#define HAVE_MEMORY_H 1

/* Define to 1 to enable internal debug messages (like NDEBUG). */
/* #undef HAVE_OTF2_DEBUG */

/* Define to 1 to disable assertions (like NDEBUG). */
#define HAVE_OTF2_NO_ASSERT 0

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

/* Set if we are building for the Cray XT platform */
/* #undef HAVE_PLATFORM_CRAYXT */

/* Set if we are building for the Linux platform */
#define HAVE_PLATFORM_LINUX 1

/* Set if we are building for the Mac platform */
/* #undef HAVE_PLATFORM_MAC */

/* Set if we are building for the NEC SX platform */
/* #undef HAVE_PLATFORM_NECSX */

/* Set if we are building for the Solaris platform */
/* #undef HAVE_PLATFORM_SOLARIS */

/* Defined if libsion SERIAL is available. */
/* #undef HAVE_SION_SERIAL */

/* Define to 1 if you have the <stdint.h> header file. */
#define HAVE_STDINT_H 1

/* Define to 1 if you have the <stdlib.h> header file. */
#define HAVE_STDLIB_H 1

/* Define to 1 if you have the <strings.h> header file. */
#define HAVE_STRINGS_H 1

/* Define to 1 if you have the <string.h> header file. */
#define HAVE_STRING_H 1

/* Define to 1 if the sysinfo() function is available. */
#define HAVE_SYSINFO 1

/* Define to 1 if you have the <sys/stat.h> header file. */
#define HAVE_SYS_STAT_H 1

/* Define to 1 if you have the <sys/sysinfo.h> header file. */
#define HAVE_SYS_SYSINFO_H 1

/* Define to 1 if you have the <sys/types.h> header file. */
#define HAVE_SYS_TYPES_H 1

/* Define to 1 if you have the <unistd.h> header file. */
#define HAVE_UNISTD_H 1

/* Interface version number */
#define LIBRARY_INTERFACE_VERSION "2:3:0"

/* Define to the sub-directory in which libtool stores uninstalled libraries.
   */
#define LT_OBJDIR ".libs/"

/* Define to 1 if your C compiler doesn't accept -c and -o together. */
/* #undef NO_MINUS_C_MINUS_O */

/* Name of package */
#define PACKAGE "otf2"

/* Define to the address where bug reports for this package should be sent. */
#define PACKAGE_BUGREPORT "scorep-bugs@groups.tu-dresden.de"

/* Build dir */
#define PACKAGE_BUILDDIR "/usr/local/packages/notransfer/src/otf2-1.1.1/build-backend"

/* The #include argument used to include this packages error codes header. */
#define PACKAGE_ERROR_CODES_HEADER <otf2/OTF2_ErrorCodes.h>

/* Define to the full name of this package. */
#define PACKAGE_NAME "OTF2"

/* Source dir */
#define PACKAGE_SRCDIR "/usr/local/packages/notransfer/src/otf2-1.1.1/build-backend/.."

/* Define to the full name and version of this package. */
#define PACKAGE_STRING "OTF2 1.1.1"

/* The package name usable as a symbol. */
#define PACKAGE_SYM otf2

/* The package name usable as a symbol in all caps. */
#define PACKAGE_SYM_CAPS OTF2

/* Define to the one symbol short name of this package. */
#define PACKAGE_TARNAME "otf2"

/* Define to the home page for this package. */
#define PACKAGE_URL ""

/* Define to the version of this package. */
#define PACKAGE_VERSION "1.1.1"

/* The clock used in clock_gettime calls. */
#define SCOREP_CLOCK_GETTIME_CLOCK CLOCK_MONOTONIC_RAW

/* Revision of common repository */
#define SCOREP_COMMON_REVISION "1063"

/* Revision of ${PACKAGE_NAME} */
#define SCOREP_COMPONENT_REVISION "2900"

/* Defined if we are working from svn. */
/* #undef SCOREP_IN_DEVELOPEMENT */

/* Defined if we are working from a make dist generated tarball. */
#define SCOREP_IN_PRODUCTION /**/

/* Defined if we are building shared libraries. See also SCOREP_STATIC_BUILD
   */
/* #undef SCOREP_SHARED_BUILD */

/* Defined if we are building static libraries. See also SCOREP_SHARED_BUILD
   */
#define SCOREP_STATIC_BUILD /**/

/* Define to 1 if you have the ANSI C header files. */
#define STDC_HEADERS 1

/* Version number of package */
#define VERSION "1.1.1"

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
