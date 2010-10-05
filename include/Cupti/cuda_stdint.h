#ifndef __cuda_stdint_h__
#define __cuda_stdint_h__

// Compiler-specific treatment for C99's stdint.h and inttypes.h
//
// By default, this header will use the standard headers (so it
// is your responsibility to make sure they are available), except
// on MSVC before Visual Studio 2010, when they were not provided.
// To support old MSVC, a few of the commonly-used definitions are
// provided here.  If more definitions are needed, add them here,
// or replace these definitions with a complete implementation,
// such as the ones available from Google, Boost, or MSVC10.  You
// can prevent the definition of any of these types (in order to
// use your own) by #defining CU_STDINT_TYPES_ALREADY_DEFINED.

#if !defined(CU_STDINT_TYPES_ALREADY_DEFINED)

#if defined(_MSC_VER) && (_MSC_VER < 1600)

// These definitions can be used with MSVC 8 and 9,
// which don't ship with stdint.h:

typedef            char    int8_t;
typedef unsigned   char   uint8_t;

typedef            short  int16_t;
typedef unsigned   short uint16_t;

typedef            long   int32_t;
typedef unsigned   long  uint32_t;

typedef          __int64  int64_t;
typedef unsigned __int64 uint64_t;

#elif defined(__DJGPP__)

// These definitions can be used when compiling
// C code with DJGPP, which only provides stdint.h
// when compiling C++ code with TR1 enabled.

typedef               char    int8_t;
typedef unsigned      char   uint8_t;

typedef               short  int16_t;
typedef unsigned      short uint16_t;

typedef               long   int32_t;
typedef unsigned      long  uint32_t;

typedef          long long   int64_t;
typedef unsigned long long  uint64_t;

#else

// Use standard headers, as specified by C99 and C++ TR1.
// Known to be provided by:
// - gcc/glibc, supported by all versions of glibc
// - djgpp, supported since 2001
// - MSVC, supported by Visual Studio 2010 and later

#include <stdint.h>
#include <inttypes.h>

#endif

#endif // !defined(CU_STDINT_TYPES_ALREADY_DEFINED)


#endif // file guard
