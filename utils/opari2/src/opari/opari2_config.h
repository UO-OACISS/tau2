/*
 * This file is part of the Score-P software (http://www.score-p.org)
 *
 * Copyright (c) 2009-2011,
 * RWTH Aachen University, Germany
 *
 * Copyright (c) 2009-2011,
 * Gesellschaft fuer numerische Simulation mbH Braunschweig, Germany
 *
 * Copyright (c) 2009-2011,
 * Technische Universitaet Dresden, Germany
 *
 * Copyright (c) 2009-2011,
 * University of Oregon, Eugene, USA
 *
 * Copyright (c) 2009-2011,
 * Forschungszentrum Juelich GmbH, Germany
 *
 * Copyright (c) 2009-2011,
 * German Research School for Simulation Sciences GmbH, Juelich/Aachen, Germany
 *
 * Copyright (c) 2009-2011,
 * Technische Universitaet Muenchen, Germany
 *
 * This software may be modified and distributed under the terms of
 * a BSD-style license. See the COPYING file in the package base
 * directory for details.
 *
 */

/** @internal
 *
 *  @file      opari2_config.h
 *
 *  @brief

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
    OPARI_Config( void );

    /**
       Destructor
     */
    virtual
    ~
    OPARI_Config( void );

    /**
       Reads the configuration data from a file. To handle the read data
       you need to implement the AddLibDir() AddIncDir() AddLib() and
       SetCompilerFlags() methods.
       @param arg0  The first argument to the toll call. Should contain the
                    tool name. Needed to find the executable path.
       @returns SCOREP_SUCCESS if the file was successfully parsed.
     */
    void
    readConfigFile( std::string arg0 );

    /**
       Changes from the install path to the build path.
     */
    void
    setBuildCheck( void );

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
    std::string m_nm;
    /** awk command*/
    std::string m_awk;
    /** egrep command*/
    std::string m_egrep;
    /** version information*/
    std::string m_version;
    /** pomp2 api version information*/
    std::string m_pomp2_api_version;
    /** awk script to use */
    std::string m_script;
    /** include path for installed headers */
    std::string m_cflags;
};
