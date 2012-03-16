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
 *  @status     alpha
 *
 *  @maintainer Dirk Schmidl <schmidl@rz.rwth-aachen.de>
 *  @autors     Daniel Lorenz <d.lorenz@fz-juelich.de>
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

#define ACTION_NM      1
#define ACTION_AWK     2
#define ACTION_SCRIPT  3
#define ACTION_EGREP   4
#define ACTION_VERSION 5
#define ACTION_NM2AWK  6
#define ACTION_CFLAGS  7
#define ACTION_POMP2_API_VERSION 8

void
opari2_print_help( char** argv )
{
    std::cout << argv[ 0 ] << "\n\n"
              << "Usage: opari2-config [OPTION] ... <command>\n\n"
              << "with following commands:\n"
              << "   --nm                   Prints the nm command.\n"
              << "   --awk-cmd              Prints the awk command.\n"
              << "   --awk-script           Prints the awk script.\n"
              << "   --egrep                Prints the egrep command.\n"
              << "   --create-pomp2-regions Prints the whole command necessary\n"
              << "                          for creating the initialization file.\n"
              << "   --cflags               Prints compiler options to include installed headers.\n"
              << "   --version              Prints the opari2 version number.\n"
              << "   --pomp2-api-version    Prints the pomp2 API version that instrumented files\n"
              << "                          conform too.\n\n"
              << "   --opari2-revision      prints the revision number of the OPARI2 package\n"
              << "   --common-revision      prints the revision number of the common package\n\n"
              << "and following options:\n"
              << "   --help                  Prints this help text.\n"
              << "   --config=<config file>  Reads in a configuration from the given file.\n\n"
              << "Report bugs to <scorep-bugs@groups.tu-dresden.de>."
              << std::endl;
    return;
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
        else if ( strcmp( argv[ i ], "--egrep" ) == 0 )
        {
            action = ACTION_EGREP;
        }
        else if ( strcmp( argv[ i ], "--cflags" ) == 0 )
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
        else if ( strcmp( argv[ i ], "--pomp2-api-version" ) == 0 )
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
        app.ReadConfigFile( config_file );
    }

    switch ( action )
    {
        case ACTION_NM:
            std::cout << app.nm;
            std::cout.flush();
            break;

        case ACTION_AWK:
            std::cout << app.awk;
            std::cout.flush();
            break;

        case ACTION_SCRIPT:
            std::cout << app.script;
            std::cout.flush();
            break;

        case ACTION_EGREP:
            std::cout << app.egrep;
            std::cout.flush();
            break;

        case ACTION_CFLAGS:
            std::cout << app.cflags;
            std::cout.flush();
            break;

        case ACTION_NM2AWK:
            std::cout << app.nm << " ";
            for ( int i = 0; i < n_obj_files; i++ )
            {
                std::cout << obj_files[ i ] << " ";
            }
            std::cout << " | " << app.egrep << " -i POMP2_Init_regions | "
                      << app.egrep << " \" [TD] \" | "
                      << app.awk << " -f " << app.script;
            break;

        case ACTION_VERSION:
            std::cout << app.version << "\n";
            std::cout.flush();
            break;

        case ACTION_POMP2_API_VERSION:
            std::cout << app.pomp2_api_version << "\n";
            std::cout.flush();
            break;

        default:
            opari2_print_help( argv );
    }

    return EXIT_SUCCESS;
}

OPARI_Config::OPARI_Config()
{
    nm                = NM;
    awk               = AWK;
    egrep             = EGREP;
    version           = VERSION;
    pomp2_api_version = POMP2_API_VERSION;
    script            = SCRIPT;
    cflags            = CFLAGS;
}

OPARI_Config::~OPARI_Config()
{
}

void
OPARI_Config::ReadConfigFile( std::string config_file )
{
    std::ifstream inFile;

    inFile.open( config_file.c_str(), std::ios_base::in );
    if ( !( inFile.good() ) )
    {
        std::cerr << "Can not open config file: "
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
        egrep = value;
    }
    else if ( key == "VERSION" )
    {
        version = value;
    }
    else if ( key == "POMP2_API_VERSION" )
    {
        pomp2_api_version = value;
    }
    else if ( key == "NM" )
    {
        nm = value;
    }
    else if ( key == "AWK" )
    {
        awk = value;
    }
    else if ( key == "OPARI_SCRIPT" )
    {
        script = value;
    }
    else if ( key == "CFLAGS" )
    {
        cflags = value;
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
