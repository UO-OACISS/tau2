#include <config.h>

#include <assert.h>
#include <setjmp.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>

#if defined( CUTEST_USE_COLOR ) && CUTEST_USE_COLOR
#include <unistd.h>
#endif

#include "CuTest.h"

/*-------------------------------------------------------------------------*
* CuStr
*-------------------------------------------------------------------------*/

char*
CuStrAlloc( int size )
{
    char* newStr = ( char* )malloc( sizeof( char ) * ( size ) );
    return newStr;
}

char*
CuStrCopy( const char* old )
{
    int   len    = strlen( old );
    char* newStr = CuStrAlloc( len + 1 );
    strcpy( newStr, old );
    return newStr;
}

/*-------------------------------------------------------------------------*
* CuString
*-------------------------------------------------------------------------*/

void
CuStringInit( CuString* str )
{
    str->length      = 0;
    str->size        = STRING_MAX;
    str->buffer      = ( char* )malloc( sizeof( char ) * str->size );
    str->buffer[ 0 ] = '\0';
}

CuString*
CuStringNew( void )
{
    CuString* str = ( CuString* )malloc( sizeof( CuString ) );
    CuStringInit( str );
    return str;
}

void
CuStringClear( CuString* str )
{
    if ( str )
    {
        free( str->buffer );
        str->buffer = NULL;
    }
}

void
CuStringFree( CuString* str )
{
    CuStringClear( str );
    free( str );
}

void
CuStringReset( CuString* str )
{
    str->length      = 0;
    str->buffer[ 0 ] = '\0';
}

void
CuStringResize( CuString* str,
                int       newSize )
{
    str->buffer = ( char* )realloc( str->buffer, sizeof( char ) * newSize );
    str->size   = newSize;
}

void
CuStringAppend( CuString*   str,
                const char* text )
{
    int length;

    if ( text == NULL )
    {
        text = "NULL";
    }

    length = strlen( text );
    if ( str->length + length + 1 >= str->size )
    {
        CuStringResize( str, str->length + length + 1 + STRING_INC );
    }
    str->length += length;
    strcat( str->buffer, text );
}

void
CuStringAppendChar( CuString* str,
                    char      ch )
{
    char text[ 2 ];
    text[ 0 ] = ch;
    text[ 1 ] = '\0';
    CuStringAppend( str, text );
}

void
CuStringAppendFormat( CuString*   str,
                      const char* format,
                      ... )
{
    va_list argp;
    va_start( argp, format );
    CuStringAppendVFormat( str, format, argp );
    va_end( argp );
}

void
CuStringAppendVFormat( CuString*   str,
                       const char* format,
                       va_list     argp )
{
    char buf[ HUGE_STRING_LEN ];
    vsprintf( buf, format, argp );
    CuStringAppend( str, buf );
}

void
CuStringInsert( CuString*   str,
                const char* text,
                int         pos )
{
    int length = strlen( text );
    if ( pos > str->length )
    {
        pos = str->length;
    }
    if ( str->length + length + 1 >= str->size )
    {
        CuStringResize( str, str->length + length + 1 + STRING_INC );
    }
    memmove( str->buffer + pos + length, str->buffer + pos, ( str->length - pos ) + 1 );
    str->length += length;
    memcpy( str->buffer + pos, text, length );
}

/*-------------------------------------------------------------------------*
* CuTest
*-------------------------------------------------------------------------*/
static void
CuTestAllreduceInternal( int* success )
{
    return;
}

void
CuTestInit( CuTest*      t,
            const char*  name,
            TestFunction function )
{
    t->name          = CuStrCopy( name );
    t->failed        = 0;
    t->ran           = 0;
    t->message       = NULL;
    t->function      = function;
    t->userArg       = NULL;
    t->testAllreduce = CuTestAllreduceInternal;
    t->jumpBuf       = NULL;
    t->next          = NULL;
}

CuTest*
CuTestNew( const char*  name,
           TestFunction function )
{
    CuTest* tc = CU_ALLOC( CuTest );
    CuTestInit( tc, name, function );
    return tc;
}

void
CuTestClear( CuTest* t )
{
    if ( t )
    {
        free( ( char* )t->name );
        t->name = NULL;
    }
}

void
CuTestFree( CuTest* t )
{
    CuTestClear( t );
    free( t );
}

void
CuTestRun( CuTest* tc )
{
    jmp_buf buf;
    tc->jumpBuf = &buf;
    if ( setjmp( buf ) == 0 )
    {
        tc->ran = 1;
        ( tc->function )( tc );
    }
    tc->jumpBuf = 0;
}

static void
CuFailInternal( CuTest*     tc,
                const char* file,
                int         line,
                CuString*   string )
{
    char buf[ HUGE_STRING_LEN ];

    sprintf( buf, "%s:%d: ", file, line );
    CuStringInsert( string, buf, 0 );

    tc->failed  = 1;
    tc->message = string->buffer;
    if ( tc->jumpBuf != 0 )
    {
        longjmp( *( tc->jumpBuf ), 1 );
    }
}

void
CuFail_Line( CuTest*     tc,
             const char* file,
             int         line,
             const char* message2,
             const char* message )
{
    CuString string;

    CuStringInit( &string );
    if ( message2 != NULL )
    {
        CuStringAppend( &string, message2 );
        CuStringAppend( &string, ": " );
    }
    CuStringAppend( &string, message );
    CuFailInternal( tc, file, line, &string );
    CuStringClear( &string );
}

void
CuAssert_Line( CuTest*     tc,
               const char* file,
               int         line,
               const char* message,
               int         success )
{
    tc->failedLocally = !success;

    tc->testAllreduce( &success );
    if ( success )
    {
        return;
    }

    CuFail_Line( tc, file, line, NULL, message );
}

void
CuAssertStrEquals_LineMsg( CuTest*     tc,
                           const char* file,
                           int         line,
                           const char* message,
                           const char* expected,
                           const char* actual )
{
    CuString string;
    int      success = 0;

    if ( ( expected == NULL && actual == NULL ) ||
         ( expected != NULL && actual != NULL &&
           strcmp( expected, actual ) == 0 ) )
    {
        success = 1;
    }

    tc->failedLocally = !success;

    tc->testAllreduce( &success );
    if ( success )
    {
        return;
    }

    CuStringInit( &string );
    if ( message != NULL )
    {
        CuStringAppend( &string, message );
        CuStringAppend( &string, ": " );
    }
    CuStringAppend( &string, "expected <" );
    CuStringAppend( &string, expected );
    CuStringAppend( &string, "> but was <" );
    CuStringAppend( &string, actual );
    CuStringAppend( &string, ">" );
    CuFailInternal( tc, file, line, &string );
    CuStringClear( &string );
}

void
CuAssertIntEquals_LineMsg( CuTest*     tc,
                           const char* file,
                           int         line,
                           const char* message,
                           int         expected,
                           int         actual )
{
    char buf[ STRING_MAX ];
    int  success = 0;

    if ( expected == actual )
    {
        success = 1;
    }

    tc->failedLocally = !success;

    tc->testAllreduce( &success );
    if ( success )
    {
        return;
    }

    sprintf( buf, "expected <%d> but was <%d>", expected, actual );
    CuFail_Line( tc, file, line, message, buf );
}


void
CuAssertIntNotEquals_LineMsg( CuTest*     tc,
                              const char* file,
                              int         line,
                              const char* message,
                              int         notExpected,
                              int         actual )
{
    char buf[ STRING_MAX ];
    int  success = 0;

    if ( notExpected != actual )
    {
        success = 1;
    }

    tc->failedLocally = !success;

    tc->testAllreduce( &success );
    if ( success )
    {
        return;
    }

    sprintf( buf, "Not expected <%d> but was <%d>", notExpected, actual );
    CuFail_Line( tc, file, line, message, buf );
}

void
CuAssertDblEquals_LineMsg( CuTest*     tc,
                           const char* file,
                           int         line,
                           const char* message,
                           double      expected,
                           double      actual,
                           double      delta )
{
    char buf[ STRING_MAX ];
    int  success = 0;

    if ( fabs( expected - actual ) <= delta )
    {
        success = 1;
    }

    tc->failedLocally = !success;

    tc->testAllreduce( &success );
    if ( success )
    {
        return;
    }

    sprintf( buf, "expected <%lf> but was <%lf>", expected, actual );
    CuFail_Line( tc, file, line, message, buf );
}

void
CuAssertPtrEquals_LineMsg( CuTest*     tc,
                           const char* file,
                           int         line,
                           const char* message,
                           void*       expected,
                           void*       actual )
{
    char buf[ STRING_MAX ];
    int  success = 0;

    if ( expected == actual )
    {
        success = 1;
    }

    tc->failedLocally = !success;

    tc->testAllreduce( &success );
    if ( success )
    {
        return;
    }

    sprintf( buf, "expected pointer <%p> but was <%p>", expected, actual );
    CuFail_Line( tc, file, line, message, buf );
}


/*-------------------------------------------------------------------------*
* CuSuite
*-------------------------------------------------------------------------*/

static char* color_red = "";
static char* color_grn = "";
static char* color_yel = "";
static char* color_std = "";

void
CuUseColors( void )
{
    if ( 0 != strcmp( getenv( "AM_COLOR_TESTS" ) ? getenv( "AM_COLOR_TESTS" ) : "", "no" )
         && 0 != strcmp( getenv( "TERM" ) ? getenv( "TERM" ) : "", "dumb" )
         && ( ( getenv( "AM_COLOR_TESTS" )
                && 0 == strcmp( getenv( "AM_COLOR_TESTS" ), "always" ) )
#if defined( CUTEST_USE_COLOR ) && CUTEST_USE_COLOR
              || isatty( STDOUT_FILENO )
#endif
              || 0 == system( "test -t 1" ) ) )
    {
        color_red = "\033[31m";
        color_grn = "\033[32m";
        color_yel = "\033[33m";
        color_std = "\033[m";
    }
}

void
CuSuiteInit( const char*   name,
             CuSuite*      testSuite,
             int           currentRank,
             TestAllreduce testAllreduce )
{
    testSuite->name        = CuStrCopy( name );
    testSuite->count       = 0;
    testSuite->failCount   = 0;
    testSuite->head        = NULL;
    testSuite->tail        = &testSuite->head;
    testSuite->currentRank = currentRank;
    testSuite->masterRank  = currentRank;

    if ( testAllreduce )
    {
        testSuite->testAllreduce = testAllreduce;
        /* determine the lowest rank and use this as the master rank */
        testSuite->testAllreduce( &testSuite->masterRank );
    }
    else
    {
        testSuite->testAllreduce = CuTestAllreduceInternal;
    }
}

CuSuite*
CuSuiteNew( const char* name )
{
    CuSuite* testSuite = CU_ALLOC( CuSuite );
    CuSuiteInit( name, testSuite, 0, NULL );
    return testSuite;
}

CuSuite*
CuSuiteNewParallel( const char*   name,
                    int           currentRank,
                    TestAllreduce testAllreduce )
{
    CuSuite* testSuite = CU_ALLOC( CuSuite );
    CuSuiteInit( name, testSuite, currentRank, testAllreduce );
    return testSuite;
}

void
CuSuiteClear( CuSuite* testSuite )
{
    if ( testSuite )
    {
        CuTest* test = testSuite->head;
        while ( test )
        {
            CuTest* next = test->next;
            CuTestFree( test );
            test = next;
        }

        free( ( char* )testSuite->name );
        testSuite->name = NULL;
    }
}

void
CuSuiteFree( CuSuite* testSuite )
{
    CuSuiteClear( testSuite );
    free( testSuite );
}

void
CuSuiteAdd( CuSuite* testSuite,
            CuTest*  testCase )
{
    testCase->testAllreduce = testSuite->testAllreduce;

    *testSuite->tail = testCase;
    testSuite->tail  = &testCase->next;
    testSuite->count++;
}

void
CuSuiteAddSuite( CuSuite* testSuite,
                 CuSuite* testSuite2 )
{
    if ( testSuite2->count )
    {
        CuSuiteAdd( testSuite, testSuite2->head );
        /* need to adjust count in testSuite */
        testSuite->count += testSuite2->count - 1;
    }
    CuSuiteClear( testSuite2 );
}

void
CuSuiteRun( CuSuite* testSuite )
{
    int     i        = 1;
    CuTest* testCase = testSuite->head;

    if ( testSuite->masterRank == testSuite->currentRank )
    {
        printf( "%s%s:%s\n", color_yel, testSuite->name, color_std );
    }

    while ( testCase )
    {
        CuTestRun( testCase );

        if ( testCase->failed )
        {
            testSuite->failCount++;

            int isSerialTest = testSuite->testAllreduce == CuTestAllreduceInternal;
            if ( isSerialTest )
            {
                printf( " %sFAIL%s %d: %s: %s%s%s\n", color_red, color_std,
                        i, testCase->name,
                        color_red, testCase->message, color_std );
            }
            else
            {
                if ( testCase->failedLocally )
                {
                    printf( " %sFAIL on Rank %d%s %d: %s: %s%s%s\n",
                            color_red, testSuite->currentRank, color_std,
                            i, testCase->name,
                            color_red, testCase->message, color_std );
                }

                int fakeBarrier = 0;
                testSuite->testAllreduce( &fakeBarrier );
            }
            break;
        }
        else
        {
            if ( testSuite->masterRank == testSuite->currentRank )
            {
                printf( "   %sok%s %d: %s\n", color_grn, color_std,
                        i, testCase->name );
            }
        }

        testCase = testCase->next;
        i++;
    }
}

void
CuSuiteSummary( CuSuite*  testSuite,
                CuString* details )
{
    int         runCount = 0;
    const char* testWord;

    if ( testSuite->masterRank != testSuite->currentRank )
    {
        return;
    }

    if ( testSuite->failCount == 0 )
    {
        testWord = testSuite->count == 1 ? "test" : "tests";
        CuStringAppendFormat( details, "%sOK%s (%d %s)\n",
                              color_grn, color_std,
                              testSuite->count, testWord );
    }
    else
    {
        for ( CuTest* testCase = testSuite->head;
              testCase;
              testCase = testCase->next )
        {
            if ( !testCase->ran )
            {
                continue;
            }
            runCount++;
        }
        testWord = testSuite->failCount == 1 ? "test" : "tests";
        CuStringAppendFormat( details, "%sFAIL%s (%d %s)",
                              color_red, color_std,
                              testSuite->failCount, testWord );
        if ( runCount - testSuite->failCount )
        {
            testWord = ( runCount - testSuite->failCount ) == 1 ? "test" : "tests";
            CuStringAppendFormat( details, " OK (%d %s)", runCount - testSuite->failCount, testWord );
        }
        if ( testSuite->count - runCount )
        {
            testWord = ( testSuite->count - runCount ) == 1 ? "test" : "tests";
            CuStringAppendFormat( details, " LEFT (%d %s)", testSuite->count - runCount, testWord );
        }
        CuStringAppendFormat( details, "\n" );
    }
}
