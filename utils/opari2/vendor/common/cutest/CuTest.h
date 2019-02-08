#ifndef CU_TEST_H
#define CU_TEST_H

#include <setjmp.h>
#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/* CuString */

char*
CuStrAlloc( int size );
char*
CuStrCopy( const char* old );

#define CU_ALLOC( TYPE )          ( ( TYPE* )calloc( 1, sizeof( TYPE ) ) )

#define HUGE_STRING_LEN 8192
#define STRING_MAX              256
#define STRING_INC              256

typedef struct
{
    int   length;
    int   size;
    char* buffer;
} CuString;

void
CuStringInit( CuString* str );
CuString*
CuStringNew( void );
void
CuStringClear( CuString* str );
void
CuStringFree( CuString* str );
void
CuStringReset( CuString* str );
void
CuStringAppend( CuString*   str,
                const char* text );
void
CuStringAppendChar( CuString* str,
                    char      ch );
void
CuStringAppendFormat( CuString*   str,
                      const char* format,
                      ... );
void
CuStringAppendVFormat( CuString*   str,
                       const char* format,
                       va_list     argp );
void
CuStringInsert( CuString*   str,
                const char* text,
                int         pos );
void
CuStringResize( CuString* str,
                int       newSize );

/* CuTest */

typedef struct CuTest CuTest;

typedef void ( * TestFunction )( CuTest* );
typedef void ( * TestAllreduce )( int* );

struct CuTest
{
    const char*    name;
    TestFunction   function;
    void*          userArg;
    TestAllreduce  testAllreduce;
    int            failed;
    int            failedLocally;
    int            ran;
    const char*    message;
    jmp_buf*       jumpBuf;
    struct CuTest* next;
};

void
CuTestInit( CuTest*      t,
            const char*  name,
            TestFunction function );
CuTest*
CuTestNew( const char*  name,
           TestFunction function );
void
CuTestClear( CuTest* t );
void
CuTestFree( CuTest* t );
void
CuTestRun( CuTest* tc );

/* Internal versions of assert functions -- use the public versions */
void
CuFail_Line( CuTest*     tc,
             const char* file,
             int         line,
             const char* message2,
             const char* message );
void
CuAssert_Line( CuTest*     tc,
               const char* file,
               int         line,
               const char* message,
               int         success );
void
CuAssertStrEquals_LineMsg( CuTest*     tc,
                           const char* file,
                           int         line,
                           const char* message,
                           const char* expected,
                           const char* actual );
void
CuAssertIntEquals_LineMsg( CuTest*     tc,
                           const char* file,
                           int         line,
                           const char* message,
                           int         expected,
                           int         actual );

void
CuAssertIntNotEquals_LineMsg( CuTest*     tc,
                              const char* file,
                              int         line,
                              const char* message,
                              int         notExpected,
                              int         actual );
void
CuAssertDblEquals_LineMsg( CuTest*     tc,
                           const char* file,
                           int         line,
                           const char* message,
                           double      expected,
                           double      actual,
                           double      delta );
void
CuAssertPtrEquals_LineMsg( CuTest*     tc,
                           const char* file,
                           int         line,
                           const char* message,
                           void*       expected,
                           void*       actual );

/* public assert functions */

#define __CuStringify( x ) #x
#define _CuStringify( x ) __CuStringify( x )
#define CuFail( tc, ms )                        CuFail_Line(  ( tc ), __FILE__, __LINE__, NULL, ( ms ) )
#define CuAssert( tc, ms, cond )                CuAssert_Line( ( tc ), __FILE__, __LINE__, ms ": " _CuStringify( cond ), ( cond ) )
#define CuAssertTrue( tc, cond )                CuAssert_Line( ( tc ), __FILE__, __LINE__, "assert failed", ( cond ) )

#define CuAssertStrEquals( tc, ex, ac )           CuAssertStrEquals_LineMsg( ( tc ), __FILE__, __LINE__, NULL, ( ex ), ( ac ) )
#define CuAssertStrEqualsMsg( tc, ms, ex, ac )    CuAssertStrEquals_LineMsg( ( tc ), __FILE__, __LINE__, ( ms ), ( ex ), ( ac ) )
#define CuAssertIntNotEquals( tc, nex, ac )        CuAssertIntNotEquals_LineMsg( ( tc ), __FILE__, __LINE__, NULL, ( nex ), ( ac ) )
#define CuAssertIntNotEqualsMsg( tc, ms, nex, ac ) CuAssertIntNotEquals_LineMsg( ( tc ), __FILE__, __LINE__, ( ms ), ( nex ), ( ac ) )
#define CuAssertIntEquals( tc, ex, ac )           CuAssertIntEquals_LineMsg( ( tc ), __FILE__, __LINE__, NULL, ( ex ), ( ac ) )
#define CuAssertIntEqualsMsg( tc, ms, ex, ac )    CuAssertIntEquals_LineMsg( ( tc ), __FILE__, __LINE__, ( ms ), ( ex ), ( ac ) )
#define CuAssertDblEquals( tc, ex, ac, dl )       CuAssertDblEquals_LineMsg( ( tc ), __FILE__, __LINE__, NULL, ( ex ), ( ac ), ( dl ) )
#define CuAssertDblEqualsMsg( tc, ms, ex, ac, dl ) CuAssertDblEquals_LineMsg( ( tc ), __FILE__, __LINE__, ( ms ), ( ex ), ( ac ), ( dl ) )
#define CuAssertPtrEquals( tc, ex, ac )           CuAssertPtrEquals_LineMsg( ( tc ), __FILE__, __LINE__, NULL, ( ex ), ( ac ) )
#define CuAssertPtrEqualsMsg( tc, ms, ex, ac )    CuAssertPtrEquals_LineMsg( ( tc ), __FILE__, __LINE__, ( ms ), ( ex ), ( ac ) )

#define CuAssertPtrNotNull( tc, p )        CuAssert_Line( ( tc ), __FILE__, __LINE__, "null pointer unexpected", ( p != NULL ) )
#define CuAssertPtrNotNullMsg( tc, msg, p ) CuAssert_Line( ( tc ), __FILE__, __LINE__, ( msg ), ( p != NULL ) )

/* CuSuite */

#define SUITE_ADD_TEST( SUITE, TEST )            CuSuiteAdd( SUITE, CuTestNew( _CuStringify( TEST ), TEST ) )
#define SUITE_ADD_TEST_NAME( SUITE, TEST, NAME ) CuSuiteAdd( SUITE, CuTestNew( NAME, TEST ) )

typedef struct
{
    const char*   name;
    int           count;
    CuTest*       head;
    CuTest**      tail;
    int           failCount;

    int           currentRank;
    int           masterRank;
    TestAllreduce testAllreduce;
} CuSuite;


void
CuUseColors( void );

void
CuSuiteInit( const char*   name,
             CuSuite*      testSuite,
             int           currentRank,
             TestAllreduce testAllreduce );
CuSuite*
CuSuiteNew( const char* name );
CuSuite*
CuSuiteNewParallel( const char*   name,
                    int           currentRank,
                    TestAllreduce testAllreduce );
void
CuSuiteClear( CuSuite* testSuite );
void
CuSuiteFree( CuSuite* testSuite );
void
CuSuiteAdd( CuSuite* testSuite,
            CuTest*  testCase );
void
CuSuiteAddSuite( CuSuite* testSuite,
                 CuSuite* testSuite2 );
void
CuSuiteRun( CuSuite* testSuite );
void
CuSuiteSummary( CuSuite*  testSuite,
                CuString* summary );

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* CU_TEST_H */
