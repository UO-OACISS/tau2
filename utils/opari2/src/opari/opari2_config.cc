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
/** @internal
 *
 *  @file       opari2_config.cc
 *
 *  @brief      Implementation of the OPARI2 config tool.
 *
 */

#include <config.h>
#include <string.h>
#include <stdlib.h>
#include <fstream>

#include <opari2_config_tool_frontend.h>
#include "opari2_config.h"

#if HAVE( READLINK )
#include <unistd.h>
#endif

#define ACTION_NM      1
#define ACTION_AWK     2
#define ACTION_SCRIPT  3
#define ACTION_EGREP   4
#define ACTION_VERSION 5
#define ACTION_NM2AWK  6
#define ACTION_CFLAGS  7
#define ACTION_POMP2_API_VERSION 8
#define ACTION_CFLAGS_GNU  9
#define ACTION_CFLAGS_INTEL  10

void
opari2_print_help( char** argv )
{
    std::string usage =
#include "opari2-config_usage.h"
    ;
    std::cout << "\n\n" << usage << std::endl;
}

int
main( int    argc,
      char** argv )
{
    int          i;
    int          action      = 0;
    char*        config_file = NULL;
    int          n_obj_files = 0;
    char**       obj_files   = NULL;
    OPARI_Config app;
    bool         fortran = false;


    /* parsing the command line */
    for ( i = 1; i < argc; i++ )
    {
        if ( strcmp( argv[ i ], "--help" ) == 0 || strcmp( argv[ i ], "-h" ) == 0 )
        {
            opari2_print_help( argv );
            return EXIT_SUCCESS;
        }
        else if ( strcmp( argv[ i ], "--nm" ) == 0 )
        {
            action = ACTION_NM;
        }
        else if ( strcmp( argv[ i ], "--awk-cmd" ) == 0 )
        {
            action = ACTION_AWK;
        }
        else if ( strcmp( argv[ i ], "--awk-script" ) == 0 )
        {
            action = ACTION_SCRIPT;
        }
        else if ( strcmp( argv[ i ], "--region-initialization" ) == 0 )
        {
            action = ACTION_SCRIPT;
        }
        else if ( strcmp( argv[ i ], "--egrep" ) == 0 )
        {
            action = ACTION_EGREP;
        }
        else if ( strcmp( argv[ i ], "--cflags=gnu" ) == 0 )
        {
            action = ACTION_CFLAGS_GNU;
        }
        else if ( strcmp( argv[ i ], "--cflags=intel" ) == 0 )
        {
            action = ACTION_CFLAGS_INTEL;
        }
        else if ( ( strcmp( argv[ i ], "--cflags" ) == 0 )       ||
                  ( strcmp( argv[ i ], "--cflags=sun" ) == 0 )   ||
                  ( strcmp( argv[ i ], "--cflags=pgi" ) == 0 )   ||
                  ( strcmp( argv[ i ], "--cflags=ibm" ) == 0 )   ||
                  ( strcmp( argv[ i ], "--cflags=cray" ) == 0 )  ||
                  ( strcmp( argv[ i ], "--cflags=fujitsu" ) == 0 )  )
        {
            action = ACTION_CFLAGS;
        }

        else if ( strcmp( argv[ i ], "--create-pomp2-regions" ) == 0 )
        {
            int j = 0;
            n_obj_files = argc - i - 1;
            if ( n_obj_files > 0 )
            {
                obj_files = new char*[ n_obj_files ];
                while ( ++i < argc )
                {
                    obj_files[ j++ ] = argv[ i ];
                }
                action = ACTION_NM2AWK;
            }
            else
            {
                std::cerr << "\nERROR: Object files missing. Abort.\n" << std::endl;
                return EXIT_FAILURE;
            }
        }
        else if ( strcmp( argv[ i ], "--version" ) == 0 )
        {
            action = ACTION_VERSION;
        }
        else if ( strcmp( argv[ i ], "--interface-version" ) == 0 )
        {
            action = ACTION_POMP2_API_VERSION;
        }
        else if ( strcmp( argv[ i ], "--opari2-revision" ) == 0 )
        {
            std::cout << SCOREP_COMPONENT_REVISION << std::endl;
            exit( EXIT_SUCCESS );
        }
        else if ( strcmp( argv[ i ], "--common-revision" ) == 0 )
        {
            std::cout << SCOREP_COMMON_REVISION << std::endl;
            exit( EXIT_SUCCESS );
        }
        else if ( strncmp( argv[ i ], "--config", 8 ) == 0 )
        {
            // Expect the config file name after an equal sign
            if ( ( argv[ i ][ 8 ] == '=' ) && ( argv[ i ][ 9 ] != '\0' ) )
            {
                config_file = &argv[ i ][ 9 ];
            }
            else
            {
                std::cerr << "\nConfig file name missing. Abort.\n" << std::endl;
                return EXIT_FAILURE;
            }
        }
        else if ( strcmp( argv[ i ], "--build-check" ) == 0 )
        {
            app.setBuildCheck();
        }
        else if ( strcmp( argv[ i ], "--fortran" ) == 0 )
        {
            fortran = true;
        }
        else
        {
            std::cerr << "\nUnknown option " << argv[ i ] << ". Abort.\n" << std::endl;
            std::cerr << "Print " << argv[ 0 ]
                      << " --help to get a list of options" << std::endl;
            return EXIT_FAILURE;
        }
    }

    /* read data in case a config file was specified */
    if ( config_file != NULL )
    {
        app.readConfigFile( config_file );
    }

    switch ( action )
    {
        case ACTION_NM:
            std::cout << app.m_nm;
            std::cout.flush();
            break;

        case ACTION_AWK:
            std::cout << app.m_awk;
            std::cout.flush();
            break;

        case ACTION_SCRIPT:
            std::cout << app.m_script;
            std::cout.flush();
            break;

        case ACTION_EGREP:
            std::cout << app.m_egrep;
            std::cout.flush();
            break;

        case ACTION_CFLAGS:
            std::cout << app.m_cflags;
            std::cout.flush();
            break;
        case ACTION_CFLAGS_GNU:
            if ( fortran )
            {
                std::cout << app.m_cflags << " -Wno-unused";
            }
            else
            {
                std::cout << app.m_cflags;
            }
            std::cout.flush();
            break;
        case ACTION_CFLAGS_INTEL:
            if ( fortran )
            {
                std::cout << app.m_cflags << " -warn nounused";
            }
            else
            {
                std::cout << app.m_cflags;
            }
            std::cout.flush();
            break;

        case ACTION_NM2AWK:
            std::cout << app.m_nm << " ";
            for ( int i = 0; i < n_obj_files; i++ )
            {
                std::cout << obj_files[ i ] << " ";
            }
            std::cout << " | "  << app.m_script;
            break;

        case ACTION_VERSION:
            std::cout << app.m_version << "\n";
            std::cout.flush();
            break;

        case ACTION_POMP2_API_VERSION:
            std::cout << app.m_pomp2_api_version << "\n";
            std::cout.flush();
            break;

        default:
            opari2_print_help( argv );
    }

    return EXIT_SUCCESS;
}

OPARI_Config::OPARI_Config( void )
{
    m_nm                = NM;
    m_awk               = AWK;
    m_egrep             = EGREP;
    m_version           = VERSION;
    m_pomp2_api_version = POMP2_API_VERSION;
    m_script            = SCRIPT;
    m_cflags            = CFLAGS;
}

OPARI_Config::~OPARI_Config( void )
{
}

void
OPARI_Config::setBuildCheck( void )
{
    #if !HAVE( READLINK )
    std::cerr << "Option --build-check not supported without readlink support." << std::endl;
    exit( EXIT_FAILURE );
    #endif /* !HAVE(READLINK) */

    unsigned bufsize = 8192;
    char     buffer[ bufsize ];
    memset( buffer, 0, bufsize );
    ssize_t result = readlink( "/proc/self/exe", buffer, bufsize );
    if ( result >= bufsize || result == -1 )
    {
        std::cerr << "Could not determine executable path. Option --build-check not supported." << std::endl;
        exit( EXIT_FAILURE );
    }
    std::string opari2_config_exe        = std::string( buffer );
    std::size_t found                    = opari2_config_exe.rfind( "opari2-config" );
    std::string opari2_config_build_path = opari2_config_exe.substr( 0, found );

    m_script = opari2_config_build_path + "pomp2-parse-init-regions.awk";
    m_cflags = "-I" + opari2_config_build_path + "../include";
}

void
OPARI_Config::readConfigFile( std::string config_file )
{
    std::ifstream inFile;

    inFile.open( config_file.c_str(), std::ios_base::in );
    if ( !( inFile.good() ) )
    {
        std::cerr << "Cannot open config file: "
                  << config_file << std::endl;
        abort();
    }

    while ( inFile.good() )
    {
        char line[ 512 ] = { "" };
        inFile.getline( line, 512 );
        read_parameter( line );
    }
}

void
OPARI_Config::set_value( std::string key, std::string value )
{
    if ( key == "EGREP" )
    {
        m_egrep = value;
    }
    else if ( key == "VERSION" )
    {
        m_version = value;
    }
    else if ( key == "POMP2_API_VERSION" )
    {
        m_pomp2_api_version = value;
    }
    else if ( key == "NM" )
    {
        m_nm = value;
    }
    else if ( key == "AWK" )
    {
        m_awk = value;
    }
    else if ( key == "OPARI_SCRIPT" )
    {
        m_script = value;
    }
    else if ( key == "CFLAGS" )
    {
        m_cflags = value;
    }
    /* Ignore unknown entries */
}

void
OPARI_Config::read_parameter( std::string line )
{
    /* check for comments */
    size_t pos = line.find( "#" );
    if ( pos == 0 )
    {
        return; // Whole line commented out
    }
    if ( pos != std::string::npos )
    {
        // Truncate line at comment
        line = line.substr( pos, line.length() - pos - 1 );
    }

    // Ignore empty lines
    if ( line == "" )
    {
        return;
    }

    /* separate value and key */
    pos = line.find( "=" );
    if ( pos == std::string::npos )
    {
        std::cerr << "Error while parsing config file: Missing separator '='."
                  << std::endl;

        abort();
    }
    std::string key   = line.substr( 0, pos );
    std::string value = line.substr( pos + 2, line.length() - pos - 3 );

    /* process parameter */
    set_value( key, value );
}
