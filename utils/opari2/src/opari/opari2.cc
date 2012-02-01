/*
 * This file is part of the Score-P software (http://www.score-p.org)
 *
 * Copyright (c) 2009-2011,
 *    RWTH Aachen University, Germany
 *    Gesellschaft fuer numerische Simulation mbH Braunschweig, Germany
 *    Technische Universitaet Dresden, Germany
 *    University of Oregon, Eugene, USA
 *    Forschungszentrum Juelich GmbH, Germany
 *    German Research School for Simulation Sciences GmbH, Juelich/Aachen, Germany
 *    Technische Universitaet Muenchen, Germany
 *
 * See the COPYING file in the package base directory for details.
 *
 */
/****************************************************************************
**  SCALASCA    http://www.scalasca.org/                                   **
**  KOJAK       http://www.fz-juelich.de/jsc/kojak/                        **
*****************************************************************************
**  Copyright (c) 1998-2009                                                **
**  Forschungszentrum Juelich, Juelich Supercomputing Centre               **
**                                                                         **
**  See the file COPYRIGHT in the package base directory for details       **
****************************************************************************/
/** @internal
 *
 *  @file       opari2.cc
 *  @status     beta
 *
 *  @maintainer Dirk Schmidl <schmidl@rz.rwth-aachen.de>
 *
 *  @authors    Bernd Mohr <b.mohr@fz-juelich.de>
 *              Dirk Schmidl <schmidl@rz-rwth-aachen.de>
 *              Peter Philippen <p.philippen@fz-juelich.de>
 *
 *  @brief     This File containes the opari main function. It is used
 *             to handle input arguments and open input and output
 *             files.  Afterwards the function process_c_or_cxx or
 *             process_fortran is called, depending on the file type
 *             or provided arguments. */

#include <config.h>
#include <fstream>
using std::ifstream;
using std::ofstream;
#include <iostream>
using std::cout;
using std::cerr;
#include <cstdio>
using std::sprintf;
using std::remove;
#include <cstring>
using std::strcmp;
using std::strrchr;
using std::strncpy;
using std::strcat;
using std::strlen;
#include <cstdlib>
using std::exit;
#include <string>
#include <assert.h>
#include <unistd.h>

#include "opari2.h"
#include "handler.h"

string pomp_tpd;
bool   copytpd        = false;
bool   task_abort     = false;
bool   task_warn      = false;
bool   task_remove    = false;
bool   untied_abort   = false;
bool   untied_keep    = false;
bool   untied_no_warn = false;

namespace
{
char* out_filename = 0;
}

void
print_usage_information( char* prog )
{
    std::string usage =
#include "opari2_usage.h"
    ;
    cerr << prog << "\n\n" << usage << std::endl;
}

void
cleanup_and_exit()
{
    if ( out_filename )
    {
        remove( out_filename );
    }
    exit( 1 );
}

#define SCOREP_STR_( str ) #str
#define SCOREP_STR( str ) SCOREP_STR_( str )
#define POMP_TPD_MANGLED FORTRAN_MANGLED( pomp_tpd )

/** @brief Main function to read and handle arguments, open files and call
 *         appropriate process function.*/
int
main( int   argc,
      char* argv[] )
{
    // -- parse options
    int         a             = 1;
    Language    lang          = L_NA;
    bool        keepSrcInfo   = true;
    bool        addSharedDecl = true;
    bool        errFlag       = false;
    char*       infile        = 0;
    const char* disabled      = 0;
    pomp_tpd = SCOREP_STR( POMP_TPD_MANGLED );
    int         retval = gettimeofday( &compiletime, NULL );
    assert( retval == 0 );


    while ( a < argc && argv[ a ][ 0 ] == '-' )
    {
        if ( strcmp( argv[ a ], "--f77" ) == 0 )
        {
            lang = L_F77;
        }
        else if ( strcmp( argv[ a ], "--f90" ) == 0 )
        {
            lang = L_F90;
        }
        else if ( strcmp( argv[ a ], "--c++" ) == 0 )
        {
            lang = L_CXX;
        }
        else if ( strcmp( argv[ a ], "--c" ) == 0 )
        {
            lang = L_C;
        }
        else if ( strcmp( argv[ a ], "--version" ) == 0 )
        {
            std::cout << "opari version " << PACKAGE_VERSION << std::endl;
            return 0;
        }
        else if ( strcmp( argv[ a ], "--help" ) == 0 )
        {
            print_usage_information( argv[ 0 ] );
            return 0;
        }
        else if ( strcmp( argv[ a ], "--nosrc" ) == 0 )
        {
            keepSrcInfo = false;
        }
        else if ( strcmp( argv[ a ], "--nodecl" ) == 0 )
        {
            addSharedDecl = false;
        }
        else if ( strcmp( argv[ a ], "--tpd" ) == 0 )
        {
            copytpd = true;
        }
        else if ( strncmp( argv[ a ], "--tpd-mangling=", 15 ) == 0 )
        {
            char* tpd_arg = strchr( argv[ a ], '=' );
            if ( tpd_arg != NULL )
            {
                tpd_arg++;
                if ( strcmp( tpd_arg, "gnu" )   == 0 || strcmp( tpd_arg, "sun" ) == 0 ||
                     strcmp( tpd_arg, "intel" ) == 0 || strcmp( tpd_arg, "pgi" ) == 0 ||
                     strcmp( tpd_arg, "cray" )  == 0 )
                {
                    pomp_tpd = "pomp_tpd_";
                }
                else if ( strcmp( tpd_arg, "ibm" ) == 0 )
                {
                    pomp_tpd = "pomp_tpd";
                }
                else
                {
                    cerr << "ERROR: unknown option for --tpd-mangling\n";
                    errFlag = true;
                }
            }
            else
            {
                cerr << "ERROR: missing value for option --tpd-mangling\n";
                errFlag = true;
            }
        }
        else if ( strncmp( argv[ a ], "--task=", 7 ) == 0 )
        {
            char* token = strtok( argv[ a ], "=" );
            token = strtok( NULL, "," );
            while ( token != NULL )
            {
                if ( strcmp( token, "abort" ) == 0 )
                {
                    task_abort = true;
                }
                else if ( strcmp( token, "warn" ) == 0 )
                {
                    task_warn = true;
                }
                else if ( strcmp( token, "remove" ) == 0 )
                {
                    task_remove = true;
                }
                else
                {
                    cerr << "ERROR: unknown option \"" << token << "\" for --task\n";
                    errFlag = true;
                }
                token = strtok( NULL, "," );
            }
        }
        else if ( strncmp( argv[ a ], "--untied=", 9 ) == 0 )
        {
            char* token = strtok( argv[ a ], "=" );
            token = strtok( NULL, "," );
            do
            {
                if ( strcmp( token, "abort" ) == 0 )
                {
                    untied_abort = true;
                }
                else if ( strcmp( token, "no-warn" ) == 0 )
                {
                    untied_no_warn = true;
                }
                else if ( strcmp( token, "keep" ) == 0 )
                {
                    untied_keep = true;
                }
                else
                {
                    cerr << "ERROR: unknown option \"" << token << "\" for --untied\n";
                    errFlag = true;
                }
                token = strtok( NULL, "," );
            }
            while ( token != NULL );
        }
        else if ( strncmp( argv[ a ], "--disable", 9 ) == 0 )
        {
            if ( strlen( argv[ a ] ) > 9 )
            {
                disabled = strchr( argv[ a ], '=' );
                if ( disabled != NULL )
                {
                    disabled++;
                    if ( set_disabled( disabled ) )
                    {
                        errFlag = true;
                    }
                }
                else
                {
                    cerr << "ERROR: missing value for option -disable\n";
                    errFlag = true;
                }
            }
            //*** Deprecated options that are still active due to compatibility reasons
            else
            {
                cerr << "WARNING: Option \"--disable <comma separated list>\" is deprecated please use --disable=<comma separated list> for future compatibilty.\n";
                if ( ( a + 1 ) < argc )
                {
                    disabled = argv[ ++a ];
                    if ( set_disabled( disabled ) )
                    {
                        errFlag = true;
                    }
                }
                else
                {
                    cerr << "ERROR: missing value for option -disable\n";
                    errFlag = true;
                }
            }
        }
        else if ( strcmp( argv[ a ], "-disable" ) == 0 )
        {
            cerr << "WARNING: Option -disable is deprecated please use --disable=<comma separated list> for future compatibilty.\n";
            if ( ( a + 1 ) < argc )
            {
                disabled = argv[ ++a ];
                if ( set_disabled( disabled ) )
                {
                    errFlag = true;
                }
            }
            else
            {
                cerr << "ERROR: missing value for option -disable\n";
                errFlag = true;
            }
        }
        else if ( strcmp( argv[ a ], "--tpd-mangling" ) == 0 )
        {
            if ( ( a + 1 ) < argc )
            {
                cerr << "WARNING: Option \"--tpd-mangling <comp>\" is deprecated please use \"--tpd-mangling=<comp>\" for future compatibilty.\n";
                a++;
                if ( strcmp( argv[ a ], "gnu" ) == 0 || strcmp( argv[ a ], "sun" ) == 0 || strcmp( argv[ a ], "intel" ) == 0 || strcmp( argv[ a ], "pgi" ) == 0 || strcmp( argv[ a ], "cray" ) == 0 )
                {
                    pomp_tpd = "pomp_tpd_";
                }
                else if ( strcmp( argv[ a ], "ibm" ) == 0 )
                {
                    pomp_tpd = "pomp_tpd";
                }
                else
                {
                    cerr << "ERROR: unknown option for --tpd-mangling\n";
                    errFlag = true;
                }
            }
            else
            {
                cerr << "ERROR: missing value for option --tpd-mangling\n";
                errFlag = true;
            }
        }
        else if ( strcmp( argv[ a ], "-nosrc" ) == 0 )
        {
            cerr << "WARNING: Option \"-nosrc\" is deprecated, please use \"--nosrc\" for future compatibilty.\n";
            keepSrcInfo = false;
        }
        else if ( strcmp( argv[ a ], "-nodecl" ) == 0 )
        {
            cerr << "WARNING: Option \"-nodecl\" is deprecated, please use \"--nodecl\" for future compatibilty.\n";
            addSharedDecl = false;
        }
        else if ( strcmp( argv[ a ], "-f77" ) == 0 )
        {
            cerr << "WARNING: Option \"-f77\" is deprecated, please use \"--f77\" for future compatibilty.\n";
            lang = L_F77;
        }
        else if ( strcmp( argv[ a ], "-f90" ) == 0 )
        {
            cerr << "WARNING: Option \"-f90\" is deprecated, please use \"--f90\" for future compatibilty.\n";
            lang = L_F90;
        }
        else if ( strcmp( argv[ a ], "-c++" ) == 0 )
        {
            cerr << "WARNING: Option \"-c++\" is deprecated, please use \"--c++\" for future compatibilty.\n";
            lang = L_CXX;
        }
        else if ( strcmp( argv[ a ], "-c" ) == 0 )
        {
            cerr << "WARNING: Option \"-c\" is deprecated, please use \"--c\" for future compatibilty.\n";
            lang = L_C;
        }
        else if ( strcmp( argv[ a ], "-rcfile" ) == 0 )
        {
            cerr << "WARNING: Option \"-rcfile\" is deprecated and ignored.\n";
        }
        else if ( strcmp( argv[ a ], "-table" ) == 0 )
        {
            cerr << "WARNING: Option \"-table\" is deprecated and ignored.\n";
        }
        //*** End of deprecated options

        else
        {
            cerr << "ERROR: unknown option " << argv[ a ] << "\n";
            errFlag = true;
        }
        ++a;
    }
    // -- parse file arguments
    ifstream is;
    ofstream os;

    switch ( argc - a )
    {
        case 2:
            if ( strcmp( argv[ a + 1 ], "-" ) == 0 )
            {
                os.std::ostream::rdbuf( cout.rdbuf() );
            }
            else
            {
                os.open( argv[ a + 1 ] );
                if ( !os )
                {
                    cerr << "ERROR: cannot open output file " << argv[ a + 1 ] << "\n";
                    errFlag = true;
                }
                os << "\n";
                out_filename = argv[ a + 1 ];
            }
        /*NOBREAK*/
        case 1:
            if ( *argv[ a ] != '/' )
            {
                int pathlength;
                pathlength = 10;
                infile     = new char[ pathlength ];
                while ( !getcwd( infile, pathlength ) )
                {
                    pathlength += 10;
                    delete[] infile;
                    infile = new char[ pathlength ];
                }
                pathlength += strlen( argv[ a ] ) + 1;
                delete[] infile;
                infile = new char[ pathlength ];
                if ( !getcwd( infile, pathlength ) )
                {
                    cerr << "ERROR: cannot determine path of input file " << infile << "\n";
                    exit( -1 );
                }
                infile = strcat( infile, "/" );
                infile = strcat( infile, argv[ a ] );
            }
            else
            {
                infile = new char[ strlen( argv[ a ] ) + 1 ];
                strcpy( infile, argv[ a ] );
            }
            is.open( infile );
            if ( !is )
            {
                cerr << "ERROR: cannot open input file " << infile << "\n";
                errFlag = true;
            }
            break;
        default:
            errFlag = true;
            break;
    }

    if ( !errFlag && infile && lang == L_NA )
    {
        const char* dot = strrchr( infile, '.' );
        if ( dot != 0  && dot[ 1 ] )
        {
            switch ( dot[ 1 ] )
            {
                case 'f':
                case 'F':
                    lang = dot[ 2 ] == '9' ? L_F90 : L_F77;
                    break;
                case 'c':
                case 'C':
                    lang = dot[ 2 ] ? L_CXX : L_C;
                    break;
            }
        }
    }
    if ( !errFlag && infile && lang == L_NA )
    {
        cerr << "ERROR: cannot determine input file language\n";
        errFlag = true;
    }

    // generate output file name if necessary
    if ( !errFlag && ( a + 1 ) == argc )
    {
        out_filename = new char[ strlen( infile ) + 5 ];
        char* dot = ( char* )strrchr( infile, '.' );
        if ( dot != 0 )
        {
            sprintf( out_filename, "%.*s.mod%s", ( int )( dot - infile ), infile, dot );

            if ( keepSrcInfo && ( lang & L_FORTRAN ) )
            {
                dot        = strrchr( out_filename, '.' );
                *( ++dot ) = 'F';
            }

            os.open( out_filename );
            if ( !os )
            {
                cerr << "ERROR: cannot open output file " << out_filename << "\n";
                errFlag = true;
            }
            os << "\n";
        }
        else
        {
            cerr << "ERROR: cannot generate output file name\n";
            errFlag = true;
        }
    }

    // print usage and die on error
    if ( errFlag )
    {
        print_usage_information( argv[ 0 ] );
        exit( 1 );
    }

    // generate opari include file name
    char* incfile       = 0;
    char* incfileNoPath = 0;

    if ( lang & L_FORTRAN )
    {
        // only need base filename without path for include statement
        // in Fortran files
        const char* dirsep = strrchr( infile, '/' );
        if ( dirsep )
        {
            incfileNoPath = new char[ strlen( dirsep ) + 12 ];
            sprintf( incfileNoPath, "%s.opari.inc", dirsep + 1 );
        }
        else
        {
            incfileNoPath = new char[ strlen( infile ) + 13 ];
            sprintf( incfileNoPath, "%s.opari.inc", infile );
        }
    }

    incfile = new char[ strlen( infile ) + 12 ];
    sprintf( incfile, "%s.opari.inc", infile );

    // transform
    do_transform = true;
    init_handler( infile, lang, keepSrcInfo );

    if ( lang & L_FORTRAN )
    {
        /*in Fortran no Underscore is needed*/
        pomp_tpd = "pomp_tpd";
        if ( keepSrcInfo )
        {
            os << "#line 1 \"" << infile << "\"" << "\n";
        }
        process_fortran( is, infile, os, addSharedDecl, incfileNoPath, lang );
    }
    else
    {
        // include file filenames are relative to base file -> need base filename
        const char* dirsep = strrchr( incfile, '/' );
        if ( dirsep )
        {
            os << "#include \"" << ( dirsep + 1 ) << "\"" << "\n";
        }
        else
        {
            os << "#include \"" << incfile << "\"" << "\n";
        }
        if ( keepSrcInfo )
        {
            os << "#line 1 \"" << infile << "\"" << "\n";
        }
        process_c_or_cxx( is, infile, os, addSharedDecl );
    }
    finalize_handler( incfile, incfileNoPath, os );
    delete[] infile;
    delete[] incfile;
    delete[] incfileNoPath;

    return 0;
}
