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

/**
 *
 * @maintainer Dirk Schmidl <schmidl@rz.rwth-aachen.de>
 * @authors    Daniel Lorenz <d.lorenz@fz.juelich.de>
 *
 */

#include <iostream>
#include <string>


class OPARI_Config
{
    /* ****************************************** Implemented public methods */
public:
    /**
       Constructor
     */
    OPARI_Config();

    /**
       Destructor
     */
    virtual
    ~
    OPARI_Config();

    /**
       Reads the configuration data from a file. To handle the read data
       you need to implement the AddLibDir() AddIncDir() AddLib() and
       SetCompilerFlags() methods.
       @param arg0  The first argument to the toll call. Should contain the
                    tool name. Needed to find the executable path.
       @returns SCOREP_SUCCESS if the file was successfully parsed.
     */
    void
    ReadConfigFile( std::string arg0 );

    /* **************************************** Protected implmented methods */
private:
    /**
       This function gives a (key, value) pair found in a configuration file and not
       processed by one of the former functions.
       @param key   The key
       @param value The value
     */
    virtual void
    set_value( std::string key,
               std::string value );

    /**
       Extracts parameter from configuration file
       It expects lines of the format key=value. Furthermore it truncates line
       at the scrpit comment character '#'.
       @param line    input line from the config file
       @returns SCOREP_SUCCESS if the line was successfully parsed. Else it
                returns an error code.
     */
    void
    read_parameter( std::string line );

    /* *************************************************** Public members */
public:
    /**nm command*/
    std::string nm;
    /** awk command*/
    std::string awk;
    /** egrep command*/
    std::string egrep;
    /** version information*/
    std::string version;
    /** pomp2 api version information*/
    std::string pomp2_api_version;
    /** awk script to use */
    std::string script;
    /** include path for installed headers */
    std::string cflags;
};
