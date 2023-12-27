/*
 * This file is part of the Score-P software (http://www.score-p.org)
 *
 * Copyright (c) 2009-2013,
 * RWTH Aachen University, Germany
 *
 * Copyright (c) 2009-2013,
 * Gesellschaft fuer numerische Simulation mbH Braunschweig, Germany
 *
 * Copyright (c) 2009-2013,
 * Technische Universitaet Dresden, Germany
 *
 * Copyright (c) 2009-2013,
 * University of Oregon, Eugene, USA
 *
 * Copyright (c) 2009-2013, 2015,
 * Forschungszentrum Juelich GmbH, Germany
 *
 * Copyright (c) 2009-2013,
 * German Research School for Simulation Sciences GmbH, Juelich/Aachen, Germany
 *
 * Copyright (c) 2009-2013,
 * Technische Universitaet Muenchen, Germany
 *
 * This software may be modified and distributed under the terms of
 * a BSD-style license. See the COPYING file in the package base
 * directory for details.
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
 *  @file		opari2.cc
 *
 *  @brief This File containes the opari main function. It is used to
 *              handle input arguments and open input and output
 *              files. Afterwards the C or Fortran parsers are used,
 *              depending on the file type or provided arguments. */

#include <config.h>
#include <fstream>
using std::ifstream;
using std::ofstream;
#include <sstream>
using std::stringstream;
#include <iostream>
using std::cout;
using std::cerr;
#include <cstdio>
using std::sprintf;
using std::remove;
#include <cstring>
using std::string;
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
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "opari2.h"
#include "opari2_directive_manager.h"
#include "openmp/opari2_directive_openmp.h"
#include "opari2_parser_c.h"
#include "opari2_parser_f.h"


#define DEPRECATED_ON


/* cmd line options */
OPARI2_Option_t opt;

void
print_usage_information( char* prog, std::ostream& output )
{
    std::string usage =
        #include "opari2_usage.h"
    ;
    output << prog << "\n\n" << usage << std::endl;
}

void
cleanup_and_exit( void )
{
    if ( !opt.outfile.empty() )
    {
        remove( opt.outfile.c_str() );
    }

    exit( 1 );
}

void
print_deprecated_msg( const string old_form, const string new_form )
{
#ifdef DEPRECATED_ON
    cerr << "Warning: Option \"" << old_form << "\" is deprecated.\n\
Please use \"" << new_form << "\" for future compatibility.\n";
#endif
}

/**
 * @brief Disable directive entry in the directive_table.
 *
 * Currently supported option:
 * --disable=xx,xx,...
 *  Here xx can possibly be an openmp/pomp directive name, openmp/pomp
 *  group name, or a paradigm type name.
 *
 * Not supported anymore:
 *  -disable xx,xx,...
 *
 * Current --help output (27.03.2014):
 * [--disable=paradigm[:directive|group[:inner],...][+paradigm...]
 *   [OPTIONAL] Disable the instrumentation of whole paradigms, or
 *   specific directives or groups of directives of a paradigm.
 *   Furthermore it gives the possibility to suppress the insertion of
 *   instrumentation functions inside code regions, i.e. only the
 *   surrounding instrumentation is inserted.  *
 */
bool
set_disabled( const string& constructs )
{
    typedef std::pair<char*, bool> dir_and_inner_t;

    char str[ constructs.length() + 1 ];
    std::strcpy( &str[ 0 ], constructs.c_str() );

    std::vector<char*> paradigms;
    char*              paradigm = strtok( &str[ 0 ], "+" );

    while ( paradigm != NULL )
    {
        paradigms.push_back( paradigm );
        paradigm = strtok( NULL, "+" );
    }

    for ( vector<char*>::iterator it = paradigms.begin(); it != paradigms.end(); ++it )
    {
        paradigm = strtok( *it, ",:" );

        dir_and_inner_t               directive;
        std::vector<dir_and_inner_t > directives;

        directive.first  = strtok( NULL, "," );
        directive.second = false;
        while ( directive.first != NULL )
        {
            directives.push_back( directive );
            directive.second = false;
            directive.first  = strtok( NULL, "," );
        }

        for ( vector<dir_and_inner_t>::iterator it = directives.begin(); it != directives.end(); ++it )
        {
            if ( strchr( it->first, ':' ) )
            {
                it->first = strtok( it->first, ":" );
                char* inner = strtok( NULL, ":" );
                if ( strcmp( inner, "inner" ) == 0 )
                {
                    it->second = true;
                }
                else
                {
                    cerr << "Error, unknown identifier " <<  inner << std::endl;
                    return false;
                }
            }
        }

        if ( directives.empty() )
        {
            if ( !DisableParadigmDirectiveOrGroup( paradigm, "", false ) )
            {
                return false;
            }
        }
        else
        {
            for ( vector<dir_and_inner_t>::iterator it = directives.begin(); it != directives.end(); ++it )
            {
                if ( !DisableParadigmDirectiveOrGroup( paradigm, it->first, it->second ) )
                {
                    return false;
                }
            }
        }
    }

    return true;
}

/**
 * @brief Parse and handle cmd line options.
 *
 * First hanlde global options,
 * then handle paradigm-specific options.
 */
void
process_cmd_line( int argc, char* argv[] )
{
    int              a        = 1;
    OPARI2_ErrorCode err_flag = OPARI2_NO_ERROR;
    const char*      ptr      = NULL;

    opt.lang          = L_NA;
    opt.form          = F_NA;
    opt.keep_src_info = true;

    opari2_omp_option* omp_opt = OPARI2_DirectiveOpenmp::GetOpenmpOpt();

    /* parse global options */
    while ( a < argc && argv[ a ][ 0 ] == '-' )
    {
        if ( strncmp( argv[ a ], "--omp", 5 ) == 0 )
        {
            err_flag =  OPARI2_DirectiveOpenmp::ProcessOption( argv[ a ] );
            if ( err_flag )
            {
                cerr << "ERROR: unknown option " << argv[ a ] << "\n";
                err_flag = OPARI2_ERROR_WITH_MESSAGE;
            }
        }
        else if ( strcmp( argv[ a ], "--f77" ) == 0 )
        {
            opt.lang = L_F77;
        }
        else if ( strcmp( argv[ a ], "--f90" ) == 0 )
        {
            opt.lang = L_F90;
        }
        else if ( strcmp( argv[ a ], "--c++" ) == 0 )
        {
            opt.lang = L_CXX;
        }
        else if ( strcmp( argv[ a ], "--c" ) == 0 )
        {
            opt.lang = L_C;
        }
        else if ( strcmp( argv[ a ], "--free-form" ) == 0 )
        {
            opt.form = F_FREE;
        }
        else if ( strcmp( argv[ a ], "--fix-form" ) == 0 )
        {
            opt.form = F_FIX;
        }
        else if ( strcmp( argv[ a ], "--version" ) == 0 )
        {
            std::cout << "opari2 version " << PACKAGE_VERSION << std::endl;
        }
        else if ( strcmp( argv[ a ], "--help" ) == 0 )
        {
            print_usage_information( argv[ 0 ], std::cout );
            exit( 0 );
        }
        else if ( strcmp( argv[ a ], "--nosrc" ) == 0 )
        {
            opt.keep_src_info = false;
        }
        else if ( strcmp( argv[ a ], "--preprocessed" ) == 0 )
        {
            opt.preprocessed_file = true;
        }
        /* handle "--disable=" */
        else if ( strncmp( argv[ a ], "--disable", 9 ) == 0 )
        {
            ptr = strchr( argv[ a ], '=' );
            if ( ptr )
            {
                ptr++;
                if ( !set_disabled( ptr ) )
                {
                    err_flag = OPARI2_ERROR_WITH_MESSAGE;
                }
            }
            else
            {
                err_flag = OPARI2_ERROR_WITH_MESSAGE;
                cerr << "ERROR: missing value for option --disable\n";
            }
        }
        /*  handle deprecated options */
        else if ( strcmp( argv[ a ], "--tpd" ) == 0 )
        {
            print_deprecated_msg( "--tpd", "--omp-tpd" );
            omp_opt->copytpd = true;
            #if HAVE( PLATFORM_K ) || HAVE( PLATFORM_FX10 ) || HAVE( PLATFORM_FX100 )
            cerr << "WARNING: option --tpd not supported on Fujitsu systems.\n";
            #endif
        }
        else if ( strncmp( argv[ a ], "--tpd-mangling=", 15 ) == 0 )
        {
            print_deprecated_msg( "--tpd-mangling=<comp>", "--omp-tpd-mangling=<comp>" );
            char* tpd_arg = strchr( argv[ a ], '=' );
            if ( tpd_arg != NULL )
            {
                tpd_arg++;
                if ( strcmp( tpd_arg, "gnu" )   == 0 || strcmp( tpd_arg, "sun" ) == 0 ||
                     strcmp( tpd_arg, "intel" ) == 0 || strcmp( tpd_arg, "pgi" ) == 0 ||
                     strcmp( tpd_arg, "cray" )  == 0 )
                {
                    omp_opt->pomp_tpd            = "pomp_tpd_";
                    omp_opt->tpd_in_extern_block = false;
                }
                else if ( strcmp( tpd_arg, "ibm" ) == 0 )
                {
                    omp_opt->pomp_tpd            = "pomp_tpd";
                    omp_opt->tpd_in_extern_block = true;
                }
                else
                {
                    cerr << "ERROR: unknown option for --tpd-mangling\n";
                    err_flag = OPARI2_ERROR_WITH_MESSAGE;
                }
            }
            else
            {
                cerr << "ERROR: missing value for option --tpd-mangling\n";
                err_flag = OPARI2_ERROR_WITH_MESSAGE;
            }
        }
        else if ( strncmp( argv[ a ], "--task=", 7 ) == 0 )
        {
            print_deprecated_msg( "--task=<comp>", "--omp-task=<comp>" );
            char* token = strtok( argv[ a ], "=" );
            token = strtok( NULL, "," );
            while ( token != NULL )
            {
                if ( strcmp( token, "abort" ) == 0 )
                {
                    omp_opt->task_abort = true;
                }
                else if ( strcmp( token, "warn" ) == 0 )
                {
                    omp_opt->task_warn = true;
                }
                else if ( strcmp( token, "remove" ) == 0 )
                {
                    omp_opt->task_remove = true;
                }
                else
                {
                    cerr << "ERROR: unknown option \"" << token << "\" for --task\n";
                    err_flag = OPARI2_ERROR_WITH_MESSAGE;
                }
                token = strtok( NULL, "," );
            }
        }
        else if ( strncmp( argv[ a ], "--untied=", 9 ) == 0 )
        {
            print_deprecated_msg( "--untied=<comp>", "--omp-task-untied=<comp>" );
            char* token = strtok( argv[ a ], "=" );
            token = strtok( NULL, "," );
            do
            {
                if ( strcmp( token, "abort" ) == 0 )
                {
                    omp_opt->untied_abort = true;
                }
                else if ( strcmp( token, "no-warn" ) == 0 )
                {
                    omp_opt->untied_nowarn = true;
                }
                else if ( strcmp( token, "keep" ) == 0 )
                {
                    omp_opt->untied_keep = true;
                }
                else
                {
                    cerr << "ERROR: unknown option \"" << token << "\" for --untied\n";
                    err_flag = OPARI2_ERROR_WITH_MESSAGE;
                }
                token = strtok( NULL, "," );
            }
            while ( token != NULL );
        }
        else if ( strcmp( argv[ a ], "-disable" ) == 0 )
        {
            cerr << "ERROR: -disable not supported by this version of OPARI2. "
                 << "Please use --disable=paradigm[:directive|group[:inner],...][+paradigm...]. "
                 << "Use opari2 --help or refer to the documentation for more details." << std::endl;
            err_flag = OPARI2_ERROR_WITH_MESSAGE;
        }
        else if ( strcmp( argv[ a ], "-f77" ) == 0 )
        {
            print_deprecated_msg( "-f77", "--f77" );
            opt.lang = L_F77;
        }
        else if ( strcmp( argv[ a ], "-f90" ) == 0 )
        {
            print_deprecated_msg( "-f90", "--f90" );
            opt.lang = L_F90;
        }
        else if ( strcmp( argv[ a ], "-c++" ) == 0 )
        {
            print_deprecated_msg( "-c++", "--c++" );
            opt.lang = L_CXX;
        }
        else if ( strcmp( argv[ a ], "-c" ) == 0 )
        {
            print_deprecated_msg( "-c", "--c" );
            opt.lang = L_C;
        }
        else if ( strcmp( argv[ a ], "-nosrc" ) == 0 )
        {
            print_deprecated_msg( "-nosrc", "--nosrc" );
            opt.keep_src_info = false;
        }
        else if ( strcmp( argv[ a ], "-rcfile" ) == 0 )
        {
            cerr << "WARNING: Option \"-rcfile\" is deprecated and ignored.\n";
        }
        else if ( strcmp( argv[ a ], "-table" ) == 0 )
        {
            cerr << "WARNING: Option \"-table\" is deprecated and ignored.\n";
        }
        /* End of deprecated options */
        ++a;
    }

    /* parse file arguments, prepare input/output stream if specified */
    switch ( argc - a )
    {
        case 2:
            if ( strcmp( argv[ a + 1 ], "-" ) == 0 )
            {
                opt.os.std::ostream::rdbuf( cout.rdbuf() );
            }
            else
            {
                opt.os.open( argv[ a + 1 ] );
                if ( !opt.os )
                {
                    cerr << "ERROR: cannot open output file " << argv[ a + 1 ] << "\n";
                    err_flag = OPARI2_ERROR_WITH_MESSAGE;
                }

                opt.outfile = string( argv[ a + 1 ] );
            }
        /*NOBREAK*/
        case 1:
            if ( *argv[ a ] != '/' )
            {
                int   pathlength = 10;
                char* tmp_inf    = new char[ pathlength ];
                while ( !getcwd( tmp_inf, pathlength ) )
                {
                    pathlength += 10;
                    delete[] tmp_inf;
                    tmp_inf = new char[ pathlength ];
                }
                pathlength += strlen( argv[ a ] ) + 1;
                delete[] tmp_inf;
                tmp_inf = new char[ pathlength ];
                if ( !getcwd( tmp_inf, pathlength ) )
                {
                    cerr << "ERROR: cannot determine path of input file " << tmp_inf << "\n";
                    exit( -1 );
                }
                tmp_inf    = strcat( tmp_inf, "/" );
                tmp_inf    = strcat( tmp_inf, argv[ a ] );
                opt.infile = string( tmp_inf );
                delete[] tmp_inf;
            }
            else
            {
                opt.infile = string( argv[ a ] );
            }
            opt.is.open( opt.infile.c_str() );
            if ( !opt.is )
            {
                cerr << "ERROR: cannot open input file " << opt.infile << "\n";
                err_flag = OPARI2_ERROR_WITH_MESSAGE;
            }
            break;
        default:
            err_flag = OPARI2_ERROR_NO_MESSAGE;
            break;
    }

    /* determine language and format by filename if not specified */
    if ( !err_flag && !opt.infile.empty() && opt.lang == L_NA )
    {
        size_t pos = opt.infile.find_last_of( '.' );

        if ( pos < opt.infile.length() + 1  && opt.infile[ pos + 1 ] )
        {
            switch ( opt.infile[ pos + 1 ] )
            {
                case 'f':
                case 'F':
                    opt.lang = opt.infile[ pos + 2 ] == '9' ? L_F90 : L_F77;
                    break;
                case 'c':
                    /*Files *.CUF and *.cuf are CUDA Fortran files*/
                    if ( opt.infile[ pos + 2 ] == 'u' && opt.infile[ pos + 3 ] == 'f' )
                    {
                        opt.lang = L_F90;
                        break;
                    }
                case 'C':
                    if ( opt.infile[ pos + 2 ] == 'U' && opt.infile[ pos + 3 ] == 'F' )
                    {
                        opt.lang = L_F90;
                        break;
                    }
                    opt.lang = opt.infile[ pos + 2 ] ? L_CXX : L_C;
                    break;
            }
        }
    }
    if ( !err_flag && opt.infile.empty() && opt.lang == L_NA )
    {
        cerr << "ERROR: cannot determine input file language\n";
        err_flag = OPARI2_ERROR_WITH_MESSAGE;
    }
    /* if no format is specified, default is free format for f90 and fix form for f77 */
    if ( ( opt.form == F_NA ) && ( opt.lang & L_FORTRAN ) )
    {
        if ( opt.lang & L_F77 )
        {
            opt.form = F_FIX;
        }
        else
        {
            opt.form = F_FREE;
        }
    }

    /* generate output file name if necessary */
    if ( !err_flag && opt.outfile.empty() )
    {
        size_t pos = opt.infile.find_last_of( '.' );
        if ( pos != string::npos )
        {
            opt.outfile = opt.infile;
            opt.outfile.replace( pos, 1, ".mod." );

            if ( opt.keep_src_info && ( opt.lang & L_FORTRAN ) )
            {
                if ( opt.outfile.find( "cuf", pos ) == pos + 5 ||
                     opt.outfile.find( "CUF", pos ) == pos + 5 )
                {
                    opt.outfile[ pos + 5 ] = 'C';
                    opt.outfile[ pos + 6 ] = 'U';
                    opt.outfile[ pos + 7 ] = 'F';
                }
                else
                {
                    opt.outfile[ pos + 5 ] = 'F';
                }
            }

            opt.os.open( opt.outfile.c_str() );
            if ( !opt.os )
            {
                cerr << "ERROR: cannot open output file " << opt.outfile << "\n";
                err_flag = OPARI2_ERROR_WITH_MESSAGE;
            }
            //            opt.os << "\n";
        }
        else
        {
            cerr << "ERROR: cannot generate output file name\n";
            err_flag = OPARI2_ERROR_WITH_MESSAGE;
        }
    }

    /* print usage and die on error */
    if ( err_flag )
    {
        if ( err_flag == OPARI2_ERROR_NO_MESSAGE )
        {
            print_usage_information( argv[ 0 ], std::cerr );
        }

        exit( 1 );
    }

    return;
}

void
misc_init()
{
    struct stat status;
    timeval     compiletime;
    //long long int     id[ 3 ];
    uint64_t     id[ 3 ];
    stringstream id_str;
    int          rest = 0;
    /* query inode number of the infile and timestamp as unique attribute */
    int retval = stat( opt.infile.c_str(), &status );

    // initialize inod_compiletime_id
    assert( retval == 0 );
    gettimeofday( &compiletime, NULL );

    //id[ 0 ] = ( long long int )status.st_ino;
    id[ 0 ] = static_cast< uint64_t > ( status.st_ino );
    id[ 1 ] = static_cast< uint64_t > ( compiletime.tv_sec );
    id[ 2 ] = static_cast< uint64_t > ( compiletime.tv_usec );

    for ( int i = 0; i < 3; i++ )
    {
        while ( id[ i ] > 36 || ( i > 1 && id[ i ] > 0 ) )
        {
            rest     = id[ i ] % 36;
            id[ i ] -= rest;
            id[ i ] /= 36;
            if ( rest < 10 )
            {
                id_str << ( char )( rest + 48 );
            }
            else
            {
                id_str << ( char )( rest + 87 );
            }
        }
        if ( i < 2 )
        {
            id[ i + 1 ] += id[ i ];
        }
    }

    // generate opari2 include file name

    // only need base filename without path for include statement
    // in Fortran files and if an output file without dir is used

    size_t sep_in = opt.infile.find_last_of( '/' );
    opt.incfile_nopath = string( opt.infile.substr( sep_in + 1 ) +
                                 ".opari.inc" );

    size_t sep_out = opt.outfile.find_last_of( '/' );
    if ( sep_out == string::npos )
    {
        opt.incfile = "";
    }
    else
    {
        opt.incfile = opt.outfile.substr( 0, sep_out + 1 );
    }
    opt.incfile += opt.incfile_nopath;

    OPARI2_Directive::SetOptions( opt.lang, opt.form, opt.keep_src_info,
                                  opt.preprocessed_file, id_str.str() );

    return;
}

/**
 * @brief Main function.
 *
 * Initialize directive and API table, handle command line options,
 * open files and call appropriate process function.
 */
int
main( int   argc,
      char* argv[] )
{
    process_cmd_line( argc, argv );

    misc_init();

    /* instrument source file */
    if ( opt.lang & L_FORTRAN )
    {
        /* in Fortran no Underscore is needed */
        OPARI2_DirectiveOpenmp::SetOptPomptpd( "pomp_tpd" );

        OPARI2_FortranParser parser( opt );
        parser.process();
    }
    else
    {
        if ( !opt.preprocessed_file )
        {
            // const char* basename = strrchr( opt.incfile, '/' );
            // basename =  basename ? ( basename + 1 ) : opt.incfile;
            opt.os << "#include \"" << opt.incfile_nopath << "\"" << "\n";

            if ( opt.keep_src_info )
            {
                opt.os << "#line 1 \"" << opt.infile << "\"" << "\n";
            }
        }

        OPARI2_CParser parser( opt );
        parser.process();
    }

    /* generate *.opari.inc ( by directive_manager ) */
    Finalize( opt );

    return 0;
}
