/*
 * This file is part of the SCOREP project (http://www.scorep.de)
 *
 * Copyright (c) 2009-2011,
 *    RWTH Aachen University, Germany
 *    Gesellschaft fuer numerische Simulation mbH Braunschweig, Germany
 *    Technische Universitaet Dresden, Germany
 *    University of Oregon, Eugene USA
 *    Forschungszentrum Juelich GmbH, Germany
 *    Technische Universitaet Muenchen, Germany
 *
 * See the COPYING file in the package base directory for details.
 *
 */

#include <config.h>
#include <stdio.h>
#include <stdint.h>
#include <malloc.h>
#include <assert.h>
#include <string.h>

#include <sys/time.h>
#include <time.h>

#include "zlib.h"


#define NUM_RECORDS            1e6
#define BYTES_PER_RECORD       15
#define TIMESTAMP_START        1e6
#define TIMESTAMP_INCR         10073
#define REGIONID( i )            ( ( i ) % 71 )
#define COMPRESSIONLEVEL       4

/* forward declarations */

int32_t
test_ascii_write( char*    buffer,
                  uint32_t wlen,
                  uint32_t num );
int32_t
test_ascii_read( char*    buffer,
                 uint32_t num );
inline char*
ascii_write_uint32( char*    p,
                    uint32_t value );
inline char*
ascii_write_uint64( char*    p,
                    uint64_t value );
inline char*
ascii_write_keyword( char* p,
                     char  keyword );
inline char*
ascii_write_newline( char* p );
inline char*
ascii_read_uint32( char*     p,
                   uint32_t* value );
inline char*
ascii_read_uint64( char*     p,
                   uint64_t* value );
inline char*
ascii_read_keyword( char* p,
                    char* value );
inline char*
ascii_read_newline( char* p );

int32_t
test_binA_write( char*    buffer,
                 uint32_t wlen,
                 uint32_t num );
int32_t
test_binA_read( char*    buffer,
                uint32_t num );
inline char*
binA_write_uint8( char*   pos,
                  uint8_t value );
inline char*
binA_write_uint32( char*    pos,
                   uint32_t value );
inline char*
binA_write_uint64( char*    pos,
                   uint64_t value );
inline char*
binA_read_uint8( char*    pos,
                 uint8_t* value );
inline char*
binA_read_uint32( char*     pos,
                  uint32_t* value );
inline char*
binA_read_uint64( char*     pos,
                  uint64_t* value );

int32_t
test_binB_write( char*    buffer,
                 uint32_t wlen,
                 uint32_t num );
int32_t
test_binB_read( char*    buffer,
                uint32_t num );
inline char*
binB_write_uint8( char*   pos,
                  uint8_t value );
inline char*
binB_write_uint32( char*    pos,
                   uint32_t value );
inline char*
binB_write_uint64( char*    pos,
                   uint64_t value );
inline char*
binB_read_uint8( char*    pos,
                 uint8_t* value );
inline char*
binB_read_uint32( char*     pos,
                  uint32_t* value );
inline char*
binB_read_uint64( char*     pos,
                  uint64_t* value );

int32_t
test_binC_write( char*    buffer,
                 uint32_t wlen,
                 uint32_t num );
int32_t
test_binC_read( char*    buffer,
                uint32_t num );
inline char*
binC_write_uint8( char*   pos,
                  uint8_t value );
inline char*
binC_write_uint32( char*    pos,
                   uint32_t value );
inline char*
binC_write_uint64( char*    pos,
                   uint64_t value );
inline char*
binC_read_uint8( char*    pos,
                 uint8_t* value );
inline char*
binC_read_uint32( char*     pos,
                  uint32_t* value );
inline char*
binC_read_uint64( char*     pos,
                  uint64_t* value );

int32_t
test_binD_write( char*    buffer,
                 uint32_t wlen,
                 uint32_t num );
int32_t
test_binD_read( char*    buffer,
                uint32_t num );
inline char*
binD_write_uint8( char*   pos,
                  uint8_t value );
inline char*
binD_write_uint32( char*    pos,
                   uint32_t value );
inline char*
binD_write_uint64( char*    pos,
                   uint64_t value );
inline char*
binD_read_uint8( char*    pos,
                 uint8_t* value );
inline char*
binD_read_uint32( char*     pos,
                  uint32_t* value );
inline char*
binD_read_uint64( char*     pos,
                  uint64_t* value );

int
call_record( char     keyword,
             uint64_t timestamp,
             uint32_t region_id );

uint64_t timestamp_check = 0;

/* main */

int
main( int          argc,
      const char** args )
{
    printf( "test encoding\n" );

    int      i, j;
    uint32_t num        = NUM_RECORDS;
    uLongf   buffersize = BYTES_PER_RECORD * num;
    int32_t  wlen, rlen;
    uLongf   wlen2;
    uLongf   zbuffersize = buffersize;
    int      ret;

    char*    buffer = ( char* )malloc( buffersize * sizeof( char* ) );
    assert( NULL != buffer );
    char*    zbuffer = ( char* )malloc( zbuffersize * sizeof( char* ) );
    assert( NULL != zbuffer );

    struct timeval timeA, timeB;
    double         write_duration, read_duration, zip_duration, unzip_duration;

    fprintf( stdout, "number of records %i\n\n", num );


    for ( i = 0; i < 2; ++i )
    {
        /* ascii */

        gettimeofday( &timeA, NULL );
        wlen = test_ascii_write( buffer, buffersize, num );
        gettimeofday( &timeB, NULL );
        write_duration = timeB.tv_sec - timeA.tv_sec + 0.000001 *
                         ( timeB.tv_usec - timeA.tv_usec );

        assert( wlen >= 0 );

        gettimeofday( &timeA, NULL );
        rlen = test_ascii_read( buffer, num );
        gettimeofday( &timeB, NULL );
        read_duration = timeB.tv_sec - timeA.tv_sec + 0.000001 *
                        ( timeB.tv_usec - timeA.tv_usec );

        assert( rlen >= 0 );

        gettimeofday( &timeA, NULL );
        zbuffersize = buffersize;
        ret         = compress2( zbuffer, &zbuffersize, buffer, wlen,
                                 COMPRESSIONLEVEL );
        assert( Z_OK == ret );
        gettimeofday( &timeB, NULL );
        zip_duration = timeB.tv_sec - timeA.tv_sec + 0.000001 *
                       ( timeB.tv_usec - timeA.tv_usec );

        gettimeofday( &timeA, NULL );
        wlen2 = buffersize;
        ret   = uncompress( buffer, &wlen2, zbuffer, buffersize );
        assert( Z_OK == ret );
        assert( wlen == wlen2 );
        gettimeofday( &timeB, NULL );
        unzip_duration = timeB.tv_sec - timeA.tv_sec + 0.000001 *
                         ( timeB.tv_usec - timeA.tv_usec );

        fprintf(
            stdout,
            "ASCII  write:   \t"
            "size %9.3f MB, \t"
            "time %7.3f s, \t"
            "%6.2f B/R \t"
            "%7.3f MR/s \n"
            "       read:    \t"
            "size %9.3f MB, \t"
            "time %7.3f s, \t"
            "%6.2f B/R \t"
            "%7.3f MR/s\n"
            "       compr L%1u: \t"
            "size %9.3f MB, \t"
            "time %7.3f s, \t"
            "%6.2f B/R \t"
            "%7.3f MR/s\n"
            "       uncompr: \t"
            "size %9.3f MB, \t"
            "time %7.3f s, \t"
            "%6.2f B/R \t"
            "%7.3f MR/s\n\n",
            wlen / 1024.0 / 1024.0, write_duration, wlen / ( float )num,
            num / ( float )write_duration / 1e6,
            rlen / 1024.0 / 1024.0, read_duration, rlen / ( float )num,
            num / ( float )read_duration / 1e6,
            COMPRESSIONLEVEL,
            zbuffersize / 1024.0 / 1024.0, zip_duration, zbuffersize /
            ( float )num, num / ( float )zip_duration / 1e6,
            wlen2 / 1024.0 / 1024.0, unzip_duration, wlen2 / ( float )num,
            num / ( float )unzip_duration / 1e6 );

        /* binA */

        gettimeofday( &timeA, NULL );
        wlen = test_binA_write( buffer, buffersize, num );
        gettimeofday( &timeB, NULL );
        write_duration = timeB.tv_sec - timeA.tv_sec + 0.000001 *
                         ( timeB.tv_usec - timeA.tv_usec );

        assert( wlen >= 0 );

        gettimeofday( &timeA, NULL );
        rlen = test_binA_read( buffer, num );
        gettimeofday( &timeB, NULL );
        read_duration = timeB.tv_sec - timeA.tv_sec + 0.000001 *
                        ( timeB.tv_usec - timeA.tv_usec );
        assert( rlen >= 0 );

        gettimeofday( &timeA, NULL );
        zbuffersize = buffersize;
        ret         = compress2( zbuffer, &zbuffersize, buffer, wlen,
                                 COMPRESSIONLEVEL );
        assert( Z_OK == ret );
        gettimeofday( &timeB, NULL );
        zip_duration = timeB.tv_sec - timeA.tv_sec + 0.000001 *
                       ( timeB.tv_usec - timeA.tv_usec );

        gettimeofday( &timeA, NULL );
        wlen2 = buffersize;
        ret   = uncompress( buffer, &wlen2, zbuffer, buffersize );
        assert( Z_OK == ret );
        assert( wlen == wlen2 );
        gettimeofday( &timeB, NULL );
        unzip_duration = timeB.tv_sec - timeA.tv_sec + 0.000001 *
                         ( timeB.tv_usec - timeA.tv_usec );

        fprintf(
            stdout,
            "binA   write:   \t"
            "size %9.3f MB, \t"
            "time %7.3f s, \t"
            "%6.2f B/R \t"
            "%7.3f MR/s \n"
            "       read:    \t"
            "size %9.3f MB, \t"
            "time %7.3f s, \t"
            "%6.2f B/R \t"
            "%7.3f MR/s\n"
            "       compr L%1u: \t"
            "size %9.3f MB, \t"
            "time %7.3f s, \t"
            "%6.2f B/R \t"
            "%7.3f MR/s\n"
            "       uncompr: \t"
            "size %9.3f MB, \t"
            "time %7.3f s, \t"
            "%6.2f B/R \t"
            "%7.3f MR/s\n\n",
            wlen / 1024.0 / 1024.0, write_duration, wlen / ( float )num,
            num / ( float )write_duration / 1e6,
            rlen / 1024.0 / 1024.0, read_duration, rlen / ( float )num,
            num / ( float )read_duration / 1e6,
            COMPRESSIONLEVEL,
            zbuffersize / 1024.0 / 1024.0, zip_duration, zbuffersize /
            ( float )num, num / ( float )zip_duration / 1e6,
            wlen2 / 1024.0 / 1024.0, unzip_duration, wlen2 / ( float )num,
            num / ( float )unzip_duration / 1e6 );

        /* binB */

        gettimeofday( &timeA, NULL );
        wlen = test_binB_write( buffer, buffersize, num );
        gettimeofday( &timeB, NULL );
        write_duration = timeB.tv_sec - timeA.tv_sec + 0.000001 *
                         ( timeB.tv_usec - timeA.tv_usec );

        assert( wlen >= 0 );

        gettimeofday( &timeA, NULL );
        rlen = test_binB_read( buffer, num );
        gettimeofday( &timeB, NULL );
        read_duration = timeB.tv_sec - timeA.tv_sec + 0.000001 *
                        ( timeB.tv_usec - timeA.tv_usec );
        assert( rlen >= 0 );

        gettimeofday( &timeA, NULL );
        zbuffersize = buffersize;
        ret         = compress2( zbuffer, &zbuffersize, buffer, wlen,
                                 COMPRESSIONLEVEL );
        assert( Z_OK == ret );
        gettimeofday( &timeB, NULL );
        zip_duration = timeB.tv_sec - timeA.tv_sec + 0.000001 *
                       ( timeB.tv_usec - timeA.tv_usec );

        gettimeofday( &timeA, NULL );
        wlen2 = buffersize;
        ret   = uncompress( buffer, &wlen2, zbuffer, buffersize );
        assert( Z_OK == ret );
        assert( wlen == wlen2 );
        gettimeofday( &timeB, NULL );
        unzip_duration = timeB.tv_sec - timeA.tv_sec + 0.000001 *
                         ( timeB.tv_usec - timeA.tv_usec );

        fprintf(
            stdout,
            "binB   write:   \t"
            "size %9.3f MB, \t"
            "time %7.3f s, \t"
            "%6.2f B/R \t"
            "%7.3f MR/s \n"
            "       read:    \t"
            "size %9.3f MB, \t"
            "time %7.3f s, \t"
            "%6.2f B/R \t"
            "%7.3f MR/s\n"
            "       compr L%1u: \t"
            "size %9.3f MB, \t"
            "time %7.3f s, \t"
            "%6.2f B/R \t"
            "%7.3f MR/s\n"
            "       uncompr: \t"
            "size %9.3f MB, \t"
            "time %7.3f s, \t"
            "%6.2f B/R \t"
            "%7.3f MR/s\n\n",
            wlen / 1024.0 / 1024.0, write_duration, wlen / ( float )num,
            num / ( float )write_duration / 1e6,
            rlen / 1024.0 / 1024.0, read_duration, rlen / ( float )num,
            num / ( float )read_duration / 1e6,
            COMPRESSIONLEVEL,
            zbuffersize / 1024.0 / 1024.0, zip_duration, zbuffersize /
            ( float )num, num / ( float )zip_duration / 1e6,
            wlen2 / 1024.0 / 1024.0, unzip_duration, wlen2 / ( float )num,
            num / ( float )unzip_duration / 1e6 );

        /* binC */

        gettimeofday( &timeA, NULL );
        wlen = test_binC_write( buffer, buffersize, num );
        gettimeofday( &timeB, NULL );
        write_duration = timeB.tv_sec - timeA.tv_sec + 0.000001 *
                         ( timeB.tv_usec - timeA.tv_usec );

        assert( wlen >= 0 );

        gettimeofday( &timeA, NULL );
        rlen = test_binC_read( buffer, num );
        gettimeofday( &timeB, NULL );
        read_duration = timeB.tv_sec - timeA.tv_sec + 0.000001 *
                        ( timeB.tv_usec - timeA.tv_usec );
        assert( rlen >= 0 );

        gettimeofday( &timeA, NULL );
        zbuffersize = buffersize;
        ret         = compress2( zbuffer, &zbuffersize, buffer, wlen,
                                 COMPRESSIONLEVEL );
        assert( Z_OK == ret );
        gettimeofday( &timeB, NULL );
        zip_duration = timeB.tv_sec - timeA.tv_sec + 0.000001 *
                       ( timeB.tv_usec - timeA.tv_usec );

        gettimeofday( &timeA, NULL );
        wlen2 = buffersize;
        ret   = uncompress( buffer, &wlen2, zbuffer, buffersize );
        assert( Z_OK == ret );
        assert( wlen == wlen2 );
        gettimeofday( &timeB, NULL );
        unzip_duration = timeB.tv_sec - timeA.tv_sec + 0.000001 *
                         ( timeB.tv_usec - timeA.tv_usec );

        fprintf(
            stdout,
            "binC   write:   \t"
            "size %9.3f MB, \t"
            "time %7.3f s, \t"
            "%6.2f B/R \t"
            "%7.3f MR/s \n"
            "       read:    \t"
            "size %9.3f MB, \t"
            "time %7.3f s, \t"
            "%6.2f B/R \t"
            "%7.3f MR/s\n"
            "       compr L%1u: \t"
            "size %9.3f MB, \t"
            "time %7.3f s, \t"
            "%6.2f B/R \t"
            "%7.3f MR/s\n"
            "       uncompr: \t"
            "size %9.3f MB, \t"
            "time %7.3f s, \t"
            "%6.2f B/R \t"
            "%7.3f MR/s\n\n",
            wlen / 1024.0 / 1024.0, write_duration, wlen / ( float )num,
            num / ( float )write_duration / 1e6,
            rlen / 1024.0 / 1024.0, read_duration, rlen / ( float )num,
            num / ( float )read_duration / 1e6,
            COMPRESSIONLEVEL,
            zbuffersize / 1024.0 / 1024.0, zip_duration, zbuffersize /
            ( float )num, num / ( float )zip_duration / 1e6,
            wlen2 / 1024.0 / 1024.0, unzip_duration, wlen2 / ( float )num,
            num / ( float )unzip_duration / 1e6 );

        /* binD */

        gettimeofday( &timeA, NULL );
        wlen = test_binD_write( buffer, buffersize, num );
        gettimeofday( &timeB, NULL );
        write_duration = timeB.tv_sec - timeA.tv_sec + 0.000001 *
                         ( timeB.tv_usec - timeA.tv_usec );

        assert( wlen >= 0 );

        gettimeofday( &timeA, NULL );
        rlen = test_binD_read( buffer, num );
        gettimeofday( &timeB, NULL );
        read_duration = timeB.tv_sec - timeA.tv_sec + 0.000001 *
                        ( timeB.tv_usec - timeA.tv_usec );
        assert( rlen >= 0 );

        gettimeofday( &timeA, NULL );
        zbuffersize = buffersize;
        ret         = compress2( zbuffer, &zbuffersize, buffer, wlen,
                                 COMPRESSIONLEVEL );
        assert( Z_OK == ret );
        gettimeofday( &timeB, NULL );
        zip_duration = timeB.tv_sec - timeA.tv_sec + 0.000001 *
                       ( timeB.tv_usec - timeA.tv_usec );

        gettimeofday( &timeA, NULL );
        wlen2 = buffersize;
        ret   = uncompress( buffer, &wlen2, zbuffer, buffersize );
        assert( Z_OK == ret );
        assert( wlen == wlen2 );
        gettimeofday( &timeB, NULL );
        unzip_duration = timeB.tv_sec - timeA.tv_sec + 0.000001 *
                         ( timeB.tv_usec - timeA.tv_usec );

        fprintf(
            stdout,
            "binD   write:   \t"
            "size %9.3f MB, \t"
            "time %7.3f s, \t"
            "%6.2f B/R \t"
            "%7.3f MR/s \n"
            "       read:    \t"
            "size %9.3f MB, \t"
            "time %7.3f s, \t"
            "%6.2f B/R \t"
            "%7.3f MR/s\n"
            "       compr L%1u: \t"
            "size %9.3f MB, \t"
            "time %7.3f s, \t"
            "%6.2f B/R \t"
            "%7.3f MR/s\n"
            "       uncompr: \t"
            "size %9.3f MB, \t"
            "time %7.3f s, \t"
            "%6.2f B/R \t"
            "%7.3f MR/s\n\n",
            wlen / 1024.0 / 1024.0, write_duration, wlen / ( float )num,
            num / ( float )write_duration / 1e6,
            rlen / 1024.0 / 1024.0, read_duration, rlen / ( float )num,
            num / ( float )read_duration / 1e6,
            COMPRESSIONLEVEL,
            zbuffersize / 1024.0 / 1024.0, zip_duration, zbuffersize /
            ( float )num, num / ( float )zip_duration / 1e6,
            wlen2 / 1024.0 / 1024.0, unzip_duration, wlen2 / ( float )num,
            num / ( float )unzip_duration / 1e6 );

        fprintf( stdout, "\n" );
    }

    return 0;
}


/* ASCII WRITING */
/* ASCII encoding with one record per line,
   with hex numbers and without leading zeros. */

int32_t
test_ascii_write( char*    buffer,
                  uint32_t wlen,
                  uint32_t num )
{
    uint32_t i;
    char*    pos = buffer;

    uint64_t timestamp;
    uint32_t region_id;

    timestamp = TIMESTAMP_START;

    for ( i = 0; i < num; ++i )
    {
        assert( pos + 100 < buffer + wlen );

        timestamp += TIMESTAMP_INCR;
        region_id  = REGIONID( i );

        pos = ascii_write_uint64( pos, timestamp );
        pos = ascii_write_newline( pos );
        pos = ascii_write_keyword( pos, 'E' );
        pos = ascii_write_uint32( pos, region_id );
        pos = ascii_write_newline( pos );
    }

    return pos - buffer;
}


/* write uint32_t as ascii hexadecimal */
inline char*
ascii_write_uint32( char*    p,
                    uint32_t value )
{
    static char dig[ 16 ] = { '0', '1', '2', '3', '4', '5', '6', '7',
                              '8', '9', 'a', 'b', 'c', 'd', 'e', 'f' };

    int         s = 28;
    uint32_t    v = 0;


    /* skip leading zeros */
    while ( ( 0 == v ) && ( s >= 0 ) )
    {
        v  = ( value >> s ) & 0xf;
        s -= 4;
    }

    *p = dig[ v ];
    p++;

    while ( s >= 0 )
    {
        v  = ( value >> s ) & 0xf;
        s -= 4;

        *p = dig[ v ];
        p++;
    }

    return p;
}


/* write uint64_t as ascii hexadecimal */
inline char*
ascii_write_uint64( char*    p,
                    uint64_t value )
{
    static char dig[ 16 ] = { '0', '1', '2', '3', '4', '5', '6', '7',
                              '8', '9', 'a', 'b', 'c', 'd', 'e', 'f' };

    int         s = 60;
    uint32_t    v = 0;


    /* skip leading zeros */
    while ( ( 0 == v ) && ( s >= 0 ) )
    {
        v  = ( uint32_t )( ( value >> s ) & 0xf );
        s -= 4;
    }

    *p = dig[ v ];
    p++;

    while ( s >= 0 )
    {
        v  = ( uint32_t )( ( value >> s ) & 0xf );
        s -= 4;

        *p = dig[ v ];
        p++;
    }

    return p;
}


/* write keyword as ascii char */
inline char*
ascii_write_keyword( char* p,
                     char  keyword )
{
    *p = keyword;
    return p + 1;
}


/* write newline as ascii char */
inline char*
ascii_write_newline( char* p )
{
    *p = '\n';
    return p + 1;
}


/* ASCII READING */
/* ASCII decoding from one record per line,
   with hex numbers and without leading zeros. */

int32_t
test_ascii_read( char*    buffer,
                 uint32_t num )
{
    uint32_t i;
    char*    pos = buffer;

    uint32_t region_id;
    uint64_t timestamp;
    char     keyword;

    timestamp_check = TIMESTAMP_START + TIMESTAMP_INCR;

    for ( i = 0; i < num; ++i )
    {
        pos = ascii_read_uint64( pos, &timestamp );
        pos = ascii_read_newline( pos );
        pos = ascii_read_keyword( pos, &keyword );
        pos = ascii_read_uint32( pos, &region_id );
        pos = ascii_read_newline( pos );

        call_record( keyword, timestamp, region_id );
    }

    return pos - buffer;
}


/* rebuild uint32_t from hexadecimal values */
inline char*
ascii_read_uint32( char*     p,
                   uint32_t* value )
{
    uint32_t v = 0;

    while ( 1 )
    {
        if ( '0' <= *p && *p <= '9' )
        {
            v = ( v << 4 ) | ( *p - '0' );
            p++;
        }
        else if ( 'a' <= *p && *p <= 'f' )
        {
            v = ( v << 4 ) | ( *p - ( 'a' - 10 ) );
            p++;
        }
        else
        {
            *value = v;
            return p;
        }
    }
}


/* rebuild uint64_t from hexadecimal values */
inline char*
ascii_read_uint64( char*     p,
                   uint64_t* value )
{
    uint64_t v = 0;

    while ( 1 )
    {
        if ( '0' <= *p && *p <= '9' )
        {
            v = ( v << 4 ) | ( *p - '0' );
            p++;
        }
        else if ( 'a' <= *p && *p <= 'f' )
        {
            v = ( v << 4 ) | ( *p - ( 'a' - 10 ) );
            p++;
        }
        else
        {
            *value = v;
            return p;
        }
    }
}


/* read keyword */
inline char*
ascii_read_keyword( char* p,
                    char* value )
{
    *value = *p;
    return p + 1;
}


/* skip newline */
inline char*
ascii_read_newline( char* p )
{
    return p + 1;
}



/* binA WRITING */
/* Binary encoding with byte-wise writing to non-aligned buffer positions.
   Note that writing  in little endian order (on little endian platform)
   improves zlib compression notably! */

int32_t
test_binA_write( char*    buffer,
                 uint32_t wlen,
                 uint32_t num )
{
    uint32_t i;
    char*    pos = buffer;

    uint64_t timestamp;
    uint32_t region_id;
    uint8_t  length;

    timestamp = TIMESTAMP_START;

    for ( i = 0; i < num; ++i )
    {
        assert( pos + 100 < buffer + wlen );

        timestamp += TIMESTAMP_INCR;
        region_id  = REGIONID( i );
        length     = 1 + 1 + 8 + 4;

        pos = binA_write_uint8( pos, 42 );
        pos = binA_write_uint8( pos, length );
        pos = binA_write_uint64( pos, timestamp );
        pos = binA_write_uint32( pos, region_id );
    }

    return pos - buffer;
}


/* write uint8_t */
inline char*
binA_write_uint8( char*   pos,
                  uint8_t value )
{
    *pos = value;
    return pos + 1;
}


/* write uint32_t */
inline char*
binA_write_uint32( char*    pos,
                   uint32_t value )
{
    /* little endian ordering */
    /* endian-safe and alignment independent */
    /* writing in this order is the same as memcpy on little endian platform,
       this imporives zlib compression for any strange reason */

    pos[ 0 ] = ( value >>  0 ) & 0xff;
    pos[ 1 ] = ( value >>  8 ) & 0xff;
    pos[ 2 ] = ( value >> 16 ) & 0xff;
    pos[ 3 ] = ( value >> 24 ) & 0xff;

    return pos + 4;
}


/* write uint64_t */
inline char*
binA_write_uint64( char*    pos,
                   uint64_t value )
{
    /* little endian ordering */
    /* endian-safe and alignment independent */
    /* writing in this order is the same as memcpy on little endian platform,
       this imporives zlib compression for any strange reason */

    pos[ 0 ] = ( value >>  0 ) & 0xff;
    pos[ 1 ] = ( value >>  8 ) & 0xff;
    pos[ 2 ] = ( value >> 16 ) & 0xff;
    pos[ 3 ] = ( value >> 24 ) & 0xff;
    pos[ 4 ] = ( value >> 32 ) & 0xff;
    pos[ 5 ] = ( value >> 40 ) & 0xff;
    pos[ 6 ] = ( value >> 48 ) & 0xff;
    pos[ 7 ] = ( value >> 56 ) & 0xff;

    return pos + 8;
}


/* binA READING */
/* Binary encoding with byte-wise writing to non-aligned buffer positions.
   Note that writing  in little endian order (on little endian platform)
   improves zlib compression notably! */

int32_t
test_binA_read( char*    buffer,
                uint32_t num )
{
    uint32_t i;
    char*    pos = buffer;

    uint64_t timestamp;
    uint32_t region_id;
    uint8_t  length;
    uint8_t  token;

    timestamp_check = TIMESTAMP_START + TIMESTAMP_INCR;

    for ( i = 0; i < num; ++i )
    {
        pos = binA_read_uint8( pos, &token );
        pos = binA_read_uint8( pos, &length );
        pos = binA_read_uint64( pos, &timestamp );
        pos = binA_read_uint32( pos, &region_id );

        call_record( token, timestamp, region_id );
    }

    return pos - buffer;
}


/* read uint8_t */
inline char*
binA_read_uint8( char*    pos,
                 uint8_t* value )
{
    *value = *pos;
    return pos + 1;
}


/* read uint32_t */
/* little endian ordering */
inline char*
binA_read_uint32( char*     pos,
                  uint32_t* value )
{
    uint8_t  i;
    uint32_t v = 0;

    v |= ( uint8_t )pos[ 3 ]; /* type cast to unsigned necessary */
    v  = ( v << 8 );
    v |= ( uint8_t )pos[ 2 ];
    v  = ( v << 8 );
    v |= ( uint8_t )pos[ 1 ];
    v  = ( v << 8 );
    v |= ( uint8_t )pos[ 0 ];

    *value = v;
    return pos + 4;
}


/* read uint64_t */
/* little endian ordering */
inline char*
binA_read_uint64( char*     pos,
                  uint64_t* value )
{
    uint8_t  i;
    uint64_t v = 0;

    v |= ( uint8_t )pos[ 7 ]; /* type cast to unsigned necessary */
    v  = ( v << 8 );
    v |= ( uint8_t )pos[ 6 ];
    v  = ( v << 8 );
    v |= ( uint8_t )pos[ 5 ];
    v  = ( v << 8 );
    v |= ( uint8_t )pos[ 4 ];
    v  = ( v << 8 );
    v |= ( uint8_t )pos[ 3 ];
    v  = ( v << 8 );
    v |= ( uint8_t )pos[ 2 ];
    v  = ( v << 8 );
    v |= ( uint8_t )pos[ 1 ];
    v  = ( v << 8 );
    v |= ( uint8_t )pos[ 0 ];

    *value = v;
    return pos + 8;
}


/* binB WRITING */
/* Binary encoding with memcpy writing to non­aligned buffer positions.
   This result is exactly the same result as the previous variant,
   but is restricted to platform endianess. */

int32_t
test_binB_write( char*    buffer,
                 uint32_t wlen,
                 uint32_t num )
{
    uint32_t i;
    char*    pos = buffer;

    uint64_t timestamp;
    uint32_t region_id;
    uint8_t  length;

    timestamp = TIMESTAMP_START;

    for ( i = 0; i < num; ++i )
    {
        assert( pos + 100 < buffer + wlen );

        timestamp += TIMESTAMP_INCR;
        region_id  = REGIONID( i );
        length     = 1 + 1 + 8 + 4;

        pos = binA_write_uint8( pos, 42 );
        pos = binA_write_uint8( pos, length );
        pos = binA_write_uint64( pos, timestamp );
        pos = binA_write_uint32( pos, region_id );
    }

    return pos - buffer;
}


/* write uint8_t */
inline char*
binB_write_uint8( char*   pos,
                  uint8_t value )
{
    memcpy( pos, &value, 1 );
    return pos + 1;
}


/* write uint32_t */
inline char*
binB_write_uint32( char*    pos,
                   uint32_t value )
{
    memcpy( pos, &value, 4 );
    return pos + 4;
}


/* write uint64_t */
inline char*
binB_write_uint64( char*    pos,
                   uint64_t value )
{
    memcpy( pos, &value, 8 );
    return pos + 8;
}


/* binB READING */
/* Binary encoding with memcpy writing to non­aligned buffer positions.
   This result is exactly the same result as the previous variant,
   but is restricted to platform endianess. */

int32_t
test_binB_read( char*    buffer,
                uint32_t num )
{
    uint32_t i;
    char*    pos = buffer;

    uint64_t timestamp;
    uint32_t region_id;
    uint8_t  length;
    uint8_t  token;

    timestamp_check = TIMESTAMP_START + TIMESTAMP_INCR;

    for ( i = 0; i < num; ++i )
    {
        pos = binA_read_uint8( pos, &token );
        pos = binA_read_uint8( pos, &length );
        pos = binA_read_uint64( pos, &timestamp );
        pos = binA_read_uint32( pos, &region_id );

        call_record( token, timestamp, region_id );
    }

    return pos - buffer;
}


/* read uint8_t */
inline char*
binB_read_uint8( char*    pos,
                 uint8_t* value )
{
    memcpy( &value, pos, 1 );
    return pos + 1;
}


/* read uint32_t */
inline char*
binB_read_uint32( char*     pos,
                  uint32_t* value )
{
    memcpy( &value, pos, 4 );
    return pos + 4;
}


/* read uint64_t */
inline char*
binB_read_uint64( char*     pos,
                  uint64_t* value )
{
    memcpy( &value, pos, 8 );
    return pos + 8;
}


/* binC WRITING */
/* Binary encoding without leading zero bytes.
   Instead, for every 32 bit or 64 bit integer first the number of used bytes
   is written as one byte, then the lower bytes are  written leaving all higher zero bytes.
   It uses the byte­wise writing and the same little  endian byte order as variant bin A. */

int32_t
test_binC_write( char*    buffer,
                 uint32_t wlen,
                 uint32_t num )
{
    uint32_t i;
    char*    pos = buffer;
    char*    pos_start;

    uint64_t timestamp;
    uint32_t region_id;
    uint8_t  length;

    timestamp = TIMESTAMP_START;

    for ( i = 0; i < num; ++i )
    {
        assert( pos + 100 < buffer + wlen );

        timestamp += TIMESTAMP_INCR;
        region_id  = REGIONID( i );
        pos_start  = pos;
        pos        = binC_write_uint8( pos, 42 );
        pos        = binC_write_uint64( pos, timestamp );
        pos        = binC_write_uint32( pos, region_id );
        length     = pos - pos_start;
        pos        = binC_write_uint8( pos, length + 1 );
    }

    return pos - buffer;
}

/* write uint8_t */
inline char*
binC_write_uint8( char*   pos,
                  uint8_t value )
{
    *pos = value;
    return pos + 1;
}


/* write uint32_t with size byte and without leading zeros */
inline char*
binC_write_uint32( char*    pos,
                   uint32_t value )
{
    if ( 0x100 > value )
    {
        /* 8 bit */

        pos[ 0 ] = 1;
        pos[ 1 ] = ( value >>  0 ) & 0xff;

        return pos + 2;
    }
    else if ( 0x10000 > value )
    {
        /* 16 bit */

        pos[ 0 ] = 2;
        pos[ 1 ] = ( value >>  0 ) & 0xff;
        pos[ 2 ] = ( value >>  8 ) & 0xff;

        return pos + 3;
    }
    else if ( 0x1000000 > value )
    {
        /* 24 bit */

        pos[ 0 ] = 3;
        pos[ 1 ] = ( value >>  0 ) & 0xff;
        pos[ 2 ] = ( value >>  8 ) & 0xff;
        pos[ 3 ] = ( value >>  16 ) & 0xff;

        return pos + 4;
    }
    else
    {
        /* 32 bit */

        pos[ 0 ] = 4;
        pos[ 1 ] = ( value >>  0 ) & 0xff;
        pos[ 2 ] = ( value >>  8 ) & 0xff;
        pos[ 3 ] = ( value >> 16 ) & 0xff;
        pos[ 4 ] = ( value >> 24 ) & 0xff;

        return pos + 5;
    }
}


/* write uint64_t with size byte and without leading zeros */
inline char*
binC_write_uint64( char*    pos,
                   uint64_t value )
{
    if ( 0x100 > value )
    {
        /* 8 bit */

        pos[ 0 ] = 1;
        pos[ 1 ] = ( value >>  0 ) & 0xff;

        return pos + 2;
    }
    else if ( 0x10000 > value )
    {
        /* 16 bit */

        pos[ 0 ] = 2;
        pos[ 1 ] = ( value >>  0 ) & 0xff;
        pos[ 2 ] = ( value >>  8 ) & 0xff;

        return pos + 3;
    }
    else if ( 0x1000000LL > value )
    {
        /* 24 bit */

        pos[ 0 ] = 3;
        pos[ 1 ] = ( value >>  0 ) & 0xff;
        pos[ 2 ] = ( value >>  8 ) & 0xff;
        pos[ 3 ] = ( value >> 16 ) & 0xff;

        return pos + 4;
    }
    else if ( 0x100000000LL > value )
    {
        /* 32 bit */

        pos[ 0 ] = 4;
        pos[ 1 ] = ( value >>  0 ) & 0xff;
        pos[ 2 ] = ( value >>  8 ) & 0xff;
        pos[ 3 ] = ( value >> 16 ) & 0xff;
        pos[ 4 ] = ( value >> 24 ) & 0xff;

        return pos + 5;
    }
    else if ( 0x10000000000LL > value )
    {
        /* 40 bit */

        pos[ 0 ] = 5;
        pos[ 1 ] = ( value >>  0 ) & 0xff;
        pos[ 2 ] = ( value >>  8 ) & 0xff;
        pos[ 3 ] = ( value >> 16 ) & 0xff;
        pos[ 4 ] = ( value >> 24 ) & 0xff;
        pos[ 5 ] = ( value >> 32 ) & 0xff;

        return pos + 6;
    }
    else if ( 0x1000000000000LL > value )
    {
        /* 48 bit */

        pos[ 0 ] = 6;
        pos[ 1 ] = ( value >>  0 ) & 0xff;
        pos[ 2 ] = ( value >>  8 ) & 0xff;
        pos[ 3 ] = ( value >> 16 ) & 0xff;
        pos[ 4 ] = ( value >> 24 ) & 0xff;
        pos[ 5 ] = ( value >> 32 ) & 0xff;
        pos[ 6 ] = ( value >> 40 ) & 0xff;

        return pos + 7;
    }
    else if ( 0x100000000000000LL > value )
    {
        /* 56 bit */

        pos[ 0 ] = 7;
        pos[ 1 ] = ( value >>  0 ) & 0xff;
        pos[ 2 ] = ( value >>  8 ) & 0xff;
        pos[ 3 ] = ( value >> 16 ) & 0xff;
        pos[ 4 ] = ( value >> 24 ) & 0xff;
        pos[ 5 ] = ( value >> 32 ) & 0xff;
        pos[ 6 ] = ( value >> 40 ) & 0xff;
        pos[ 7 ] = ( value >> 48 ) & 0xff;

        return pos + 8;
    }
    else
    {
        /* 64 bit */

        pos[ 0 ] = 8;
        pos[ 1 ] = ( value >>  0 ) & 0xff;
        pos[ 2 ] = ( value >>  8 ) & 0xff;
        pos[ 3 ] = ( value >> 16 ) & 0xff;
        pos[ 4 ] = ( value >> 24 ) & 0xff;
        pos[ 5 ] = ( value >> 32 ) & 0xff;
        pos[ 6 ] = ( value >> 40 ) & 0xff;
        pos[ 7 ] = ( value >> 48 ) & 0xff;
        pos[ 8 ] = ( value >> 56 ) & 0xff;

        return pos + 9;
    }
}


/* binC READING */
/* Binary encoding without leading zero bytes.
   Instead, for every 32 bit or 64 bit integer first the number of used bytes
   is written as one byte, then the lower bytes are  written leaving all higher zero bytes.
   It uses the byte­wise writing and the same little  endian byte order as variant bin A. */

int32_t
test_binC_read( char*    buffer,
                uint32_t num )
{
    uint32_t i;
    char*    pos = buffer;

    uint64_t timestamp;
    uint32_t region_id;
    uint8_t  length;
    uint8_t  token;

    timestamp_check = TIMESTAMP_START + TIMESTAMP_INCR;

    for ( i = 0; i < num; ++i )
    {
        pos = binC_read_uint8( pos, &token );
        pos = binC_read_uint64( pos, &timestamp );
        pos = binC_read_uint32( pos, &region_id );
        pos = binC_read_uint8( pos, &length );

        call_record( token, timestamp, region_id );
    }

    return pos - buffer;
}

/* read uint8_t */
inline char*
binC_read_uint8( char*    pos,
                 uint8_t* value )
{
    *value = *pos;
    return pos + 1;
}


/* read uint32_t using size byte */
inline char*
binC_read_uint32( char*     pos,
                  uint32_t* value )
{
    uint32_t v;
    uint8_t  i, len = *pos;
    pos++;

    if ( len == 1 )
    {
        v = ( uint8_t )pos[ 0 ]; /* type cast to unsigned necessary */
    }
    else if ( len == 2 )
    {
        v  = ( uint8_t )pos[ 1 ];
        v  = ( v << 8 );
        v |= ( uint8_t )pos[ 0 ];
    }
    else if ( len == 3 )
    {
        v  = ( uint8_t )pos[ 2 ];
        v  = ( v << 8 );
        v |= ( uint8_t )pos[ 1 ];
        v  = ( v << 8 );
        v |= ( uint8_t )pos[ 0 ];
    }
    else
    {
        v  = ( uint8_t )pos[ 3 ];
        v  = ( v << 8 );
        v |= ( uint8_t )pos[ 2 ];
        v  = ( v << 8 );
        v |= ( uint8_t )pos[ 1 ];
        v  = ( v << 8 );
        v |= ( uint8_t )pos[ 0 ];
    }

    *value = v;
    return pos + len;
}


/* read uint64_t using size byte */
inline char*
binC_read_uint64( char*     pos,
                  uint64_t* value )
{
    uint64_t v;
    uint8_t  i, len = *pos;
    pos++;

    if ( len == 1 )
    {
        v = ( uint8_t )pos[ 0 ]; /* type cast to unsigned necessary */
    }
    else if ( len == 2 )
    {
        v  = ( uint8_t )pos[ 1 ];
        v  = ( v << 8 );
        v |= ( uint8_t )pos[ 0 ];
    }
    else if ( len == 3 )
    {
        v  = ( uint8_t )pos[ 2 ];
        v  = ( v << 8 );
        v |= ( uint8_t )pos[ 1 ];
        v  = ( v << 8 );
        v |= ( uint8_t )pos[ 0 ];
    }
    else if ( len == 4 )
    {
        v  = ( uint8_t )pos[ 3 ];
        v  = ( v << 8 );
        v |= ( uint8_t )pos[ 2 ];
        v  = ( v << 8 );
        v |= ( uint8_t )pos[ 1 ];
        v  = ( v << 8 );
        v |= ( uint8_t )pos[ 0 ];
    }
    else if ( len == 5 )
    {
        v  = ( uint8_t )pos[ 4 ];
        v  = ( v << 8 );
        v |= ( uint8_t )pos[ 3 ];
        v  = ( v << 8 );
        v |= ( uint8_t )pos[ 2 ];
        v  = ( v << 8 );
        v |= ( uint8_t )pos[ 1 ];
        v  = ( v << 8 );
        v |= ( uint8_t )pos[ 0 ];
    }
    else if ( len == 6 )
    {
        v  = ( uint8_t )pos[ 5 ];
        v  = ( v << 8 );
        v |= ( uint8_t )pos[ 4 ];
        v  = ( v << 8 );
        v |= ( uint8_t )pos[ 3 ];
        v  = ( v << 8 );
        v |= ( uint8_t )pos[ 2 ];
        v  = ( v << 8 );
        v |= ( uint8_t )pos[ 1 ];
        v  = ( v << 8 );
        v |= ( uint8_t )pos[ 0 ];
    }
    else if ( len == 7 )
    {
        v  = ( uint8_t )pos[ 6 ];
        v  = ( v << 8 );
        v |= ( uint8_t )pos[ 5 ];
        v  = ( v << 8 );
        v |= ( uint8_t )pos[ 4 ];
        v  = ( v << 8 );
        v |= ( uint8_t )pos[ 3 ];
        v  = ( v << 8 );
        v |= ( uint8_t )pos[ 2 ];
        v  = ( v << 8 );
        v |= ( uint8_t )pos[ 1 ];
        v  = ( v << 8 );
        v |= ( uint8_t )pos[ 0 ];
    }
    else
    {
        v  = ( uint8_t )pos[ 7 ];
        v  = ( v << 8 );
        v |= ( uint8_t )pos[ 6 ];
        v  = ( v << 8 );
        v |= ( uint8_t )pos[ 5 ];
        v  = ( v << 8 );
        v |= ( uint8_t )pos[ 4 ];
        v  = ( v << 8 );
        v |= ( uint8_t )pos[ 3 ];
        v  = ( v << 8 );
        v |= ( uint8_t )pos[ 2 ];
        v  = ( v << 8 );
        v |= ( uint8_t )pos[ 1 ];
        v  = ( v << 8 );
        v |= ( uint8_t )pos[ 0 ];
    }

    *value = v;
    return pos + len;
}


/* binD WRITING */
/* use 7 bits per byte for the value. if the most significant bit is set,
   then another byte needs to be added, otherwise the end of the number is reached */

int32_t
test_binD_write( char*    buffer,
                 uint32_t len,
                 uint32_t num )
{
    uint32_t i;
    char*    pos = buffer;
    char*    pos_start;

    uint64_t timestamp;
    uint32_t region_id;
    uint8_t  length;

    timestamp = TIMESTAMP_START;

    for ( i = 0; i < num; ++i )
    {
        assert( pos + 100 < buffer + len );

        timestamp += TIMESTAMP_INCR;
        region_id  = REGIONID( i );
        pos_start  = pos;
        pos        = binD_write_uint8( pos, 42 );
        pos        = binD_write_uint64( pos, timestamp );
        pos        = binD_write_uint32( pos, region_id );
        length     = pos - pos_start;
        pos        = binC_write_uint8( pos, length + 1 );
    }

    return pos - buffer;
}


inline char*
binD_write_uint8( char*   pos,
                  uint8_t value )
{
    *pos = value;
    return pos + 1;
}


inline char*
binD_write_uint32( char*    pos,
                   uint32_t value )
{
    /* use 7 bits per byte for the value. if the most significant bit is set,
       then another byte needs to be added, otherwise the end of the number is reached */

    char*   p = pos;
    uint8_t v;
    uint8_t goon;

    do
    {
        v     = ( value & 0x7f );         /* take 7 lower bits out of 8 */
        value = ( value >> 7 );
        goon  = ( 0 < value ) ? 0x80 : 0; /* true is 10000000b */

        *p = ( goon | v );
        p++;
    }
    while ( 0 < goon );

    return p;
}


inline char*
binD_write_uint64( char*    pos,
                   uint64_t value )
{
    /* use 7 bits per byte for the value. if the most significant bit is set,
       then another byte needs to be added, otherwise the end of the number is reached */

    char*   p = pos;
    uint8_t v;
    uint8_t goon;

    do
    {
        v     = ( value & 0x7f );         /* take 7 lower bits out of 8 */
        value = ( value >> 7 );
        goon  = ( 0 < value ) ? 0x80 : 0; /* true is 10000000b */

        *p = ( goon | v );
        p++;
    }
    while ( 0 < goon );

    return p;
}


/* binD READING */
/* use 7 bits per byte for the value. if the most significant bit is set,
   then another byte needs to be added, otherwise the end of the number is reached */

int32_t
test_binD_read( char*    buffer,
                uint32_t num )
{
    uint32_t i;
    char*    pos = buffer;

    uint64_t timestamp;
    uint32_t region_id;
    uint8_t  length;
    uint8_t  token;

    timestamp_check = TIMESTAMP_START + TIMESTAMP_INCR;

    for ( i = 0; i < num; ++i )
    {
        pos = binD_read_uint8( pos, &token );
        pos = binD_read_uint64( pos, &timestamp );
        pos = binD_read_uint32( pos, &region_id );
        pos = binC_read_uint8( pos, &length );

        call_record( token, timestamp, region_id );
    }

    return pos - buffer;
}


/* read uint8_t like above versions */
inline char*
binD_read_uint8( char*    pos,
                 uint8_t* value )
{
    *value = *pos;
    return pos + 1;
}


/* rebuilt uint32_t from 7bit values using most significant bit */
inline char*
binD_read_uint32( char*     pos,
                  uint32_t* value )
{
    uint32_t v = 0, w = 0;
    uint8_t  goon, i = 0;

    do
    {
        w    = ( *pos & 0x7f );
        w    = ( w << 7 * i );
        v   |= w;
        goon = ( *pos & 0x80 );
        i++;
        pos++;
    }
    while ( 0 < goon );

    *value = v;
    return pos;
}


/* rebuilt uint64_t from 7bit values using most significant bit */
inline char*
binD_read_uint64( char*     pos,
                  uint64_t* value )
{
    uint64_t v = 0, w = 0;
    uint8_t  goon, i = 0;

    do
    {
        w    = ( *pos & 0x7f );
        w    = ( w << 7 * i );
        v   |= w;
        goon = ( *pos & 0x80 );
        i++;
        pos++;
    }
    while ( 0 < goon );

    *value = v;
    return pos;
}


/* record read call */
int
call_record( char     keyword,
             uint64_t timestamp,
             uint32_t region_id )
{
    if ( timestamp_check != timestamp )
    {
        fprintf( stderr, "parse error %llu != %llu \n",
                 timestamp_check,
                 timestamp );
        assert( timestamp_check == timestamp );
    }

    timestamp_check = timestamp + TIMESTAMP_INCR;
}
