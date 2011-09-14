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
 *  @brief     This File containes the opari main function. It is used to
 *             handle input arguments and open input and output files. 
 *             Afterwards the function process_c_or_cxx or process_fortran
 *             is called, depending on the file type or provided arguments. */

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
bool copytpd=false;

namespace
{
void
define_POMP2( ostream& os )
{
    /// @todo define _POMP2 via svnversion during configure/make dist.
    os << "#ifdef _POMP2\n"
       << "#  undef _POMP2\n"
       << "#endif\n"
       << "#define _POMP2 200110\n\n";
}

char* out_filename = 0;
}

void
print_usage_information(char* prog)
{
        std::string usage =
#include "opari_user_usage.h"
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
 * 	   appropriate process function.*/
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
    char* infile        = 0;
    const char* disabled      = 0;
    pomp_tpd = SCOREP_STR(POMP_TPD_MANGLED);
    int     retval = gettimeofday( &compiletime, NULL );
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
		print_usage_information(argv[0]);
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
        else if ( strcmp( argv[ a ], "--disable" ) == 0 )
        {
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
                cerr << "ERROR: missing value for option --disable\n";
                errFlag = true;
            }
        }
	else if ( strcmp( argv[ a ], "--tpd-mangling" ) == 0 )
        {
            if ( ( a + 1 ) < argc )
            {
                a++;
		if ( strcmp( argv[ a ], "gnu" ) == 0 || strcmp( argv[ a ], "sun" ) == 0 || strcmp( argv[ a ], "intel" ) == 0 || strcmp( argv[ a ], "pgi" ) == 0 )
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
                out_filename = argv[ a + 1 ];
            }
        /*NOBREAK*/
        case 1:
	    int pathlength;
	    pathlength=10;
	    infile=new char [ pathlength ];
	    while (!getcwd(infile,pathlength)){
	    	pathlength+=10;
		delete[] infile;
	    	infile = new char[pathlength];
	    }
	    pathlength+=strlen(argv [ a ]) + 1 ;
	    delete[] infile;
	    infile = new char[pathlength];
	    if (!getcwd(infile,pathlength))
	    {
		cerr << "ERROR: cannot determine path of input file " << infile << "\n";
		exit(-1);
	    }
            infile = strcat(infile, "/");
            infile = strcat(infile, argv[ a ]);
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
    if ( infile && lang == L_NA )
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
        print_usage_information(argv[0]); 
        exit( 1 );
    }

    // generate opari include file name
    // C: in directory of C/C++ base file
    // F: in rcfile directory
    char* incfile = 0;
    if ( lang & L_FORTRAN )
    {
        // only need base filename without path
        const char* dirsep = strrchr( infile, '/' );
        if ( dirsep )
        {
            incfile = new char[ strlen( dirsep ) + 12 ];
            sprintf( incfile, "%s.opari.inc", dirsep + 1 );
        }
        else
        {
            incfile = new char[ strlen( infile ) + 13 ];
            sprintf( incfile, "%s.opari.inc", infile );
        }
    }
    else
    {
        incfile = new char[ strlen( infile ) + 12 ];
        sprintf( incfile, "%s.opari.inc", infile );
    }

    // transform
    do_transform = true;
    init_handler( infile, lang, keepSrcInfo );

    if ( lang & L_FORTRAN )
    {
	/*in Fortran no Underscore is needed*/
	pomp_tpd="pomp_tpd";
        if ( keepSrcInfo )
        {
            define_POMP2( os );
            os << "#line 1 \"" << infile << "\"" << "\n";
        }
        process_fortran( is, infile, os, addSharedDecl , incfile, lang);
    }
    else
    {
        define_POMP2( os );
        // include file filenames are relative to base file -> need base filename
        const char* dirsep = strrchr(incfile, '/');
        if ( dirsep ) { 
            os << "#include \"" << (dirsep+1) << "\"" << "\n";
        } else {
            os << "#include \"" << incfile << "\"" << "\n";
        }
        if ( keepSrcInfo )
        {
            os << "#line 1 \"" << infile << "\"" << "\n";
        }
        process_c_or_cxx( is, infile, os, addSharedDecl );
    }
    finalize_handler( incfile , os );
    delete[] infile;
    delete[] incfile;

    return 0;
}

