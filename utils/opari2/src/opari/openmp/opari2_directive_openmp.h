/*
 * This file is part of the Score-P software (http://www.score-p.org)
 *
 * Copyright (c) 2013,
 * Forschungszentrum Juelich GmbH, Germany
 *
 * This software may be modified and distributed under the terms of
 * a BSD-style license. See the COPYING file in the package base
 * directory for details.
 *
 */

/** @internal
 *
 *  @file      opari2_directive_openmp.h
 *
 *  @brief     Class definitions for Openmp directives in C/C++ and Fortran.
 */

#ifndef OPARI2_DIRECTIVE_OPENMP_H
#define OPARI2_DIRECTIVE_OPENMP_H

#include <string>
using std::string;
#include <vector>
using std::vector;
#include <iostream>
using std::ostream;

#include "opari2.h"
#include "opari2_directive.h"

/** @brief Anonymous namespace including structure definition for OpenMP specific cmd line options. */

/** @brief Structure for OpenMP specific cmd line options. */
typedef struct
{
    bool   add_shared_decl;
    bool   copytpd;
    bool   task_abort;
    bool   task_warn;
    bool   task_remove;
    bool   untied_abort;
    bool   untied_keep;
    bool   untied_nowarn;
    bool   tpd_in_extern_block;
    string pomp_tpd;
} opari2_omp_option;


/** @brief Base Class to store and manipulate OpenMP directive related data. */

class OPARI2_DirectiveOpenmp : public OPARI2_Directive
{
public:
    /**@brief Constructor. */
    OPARI2_DirectiveOpenmp( const string&   fname,
                            const int       ln,
                            vector<string>& lines,
                            vector<string>& directive_prefix )
        : OPARI2_Directive( fname, ln, lines, directive_prefix )
    {
        m_type = OPARI2_PT_OMP;

        m_nowait_added = false;
        m_has_untied   = false;
        m_has_ordered  = false;
        m_num_sections = 0;

        if ( lines.empty() )
        {
            m_indent = 0;
        }
        else
        {
            m_indent = lines[ 0 ].find_first_not_of( " \t" );
            m_indent = m_indent == string::npos ? 0 : m_indent;
        }
    }

    /** @brief Destructor. */
    virtual
    ~
    OPARI2_DirectiveOpenmp( void )
    {
    }

    /** Increment region counter */
    virtual void
    IncrementRegionCounter( void );

    /**
     * @brief Parse OpenMP-specific command-line option and store it
     *        in 's_omp_opt'.
     */
    static OPARI2_ErrorCode
    ProcessOption( string option );

    /** @brief Set value for 'opt.omp_pomp_tpd'. */
    static void
    SetOptPomptpd( string str );

    /** return 's_openmp_opt' */
    static opari2_omp_option*
    GetOpenmpOpt( void );

    /** Parse string and return OpenMP group id */
    static uint32_t
    String2Group( const string name );

    virtual void
    FindName( void );

    /** is the default data sharing changed,
     *  i.e. is default(none) or default(private) present?
     */
    bool
    ChangedDefault( void );

    /** returns value in brackets, if present, "" otherwise*/
    void
    FindUserName( void );

    string&
    GetUserName( void );

    void
    SetNumSections( int n );

    int
    GetNumSections( void );

    string&
    GetReduction( void );

    /** split compined constructs in two seperate statement */
    virtual OPARI2_DirectiveOpenmp*
    SplitCombined( void );

    /** add OpenMP descriptors at the right place */
    virtual void
    AddDescr( int n );

    virtual void
    GenerateDescr( ostream& os );

    static void
    FinalizeDescrs( ostream& os );

    /** Generate header of the include file */
    static void
    GenerateHeader( ostream& os );

    /** generate call POMP2_Init_reg_XXX function to initialize
     *  all handles, in the Fortran case
     */
    static void
    GenerateInitHandleCalls( ostream&     os,
                             const string incfile = "" );

    /** add a nowait to a directive*/
    virtual void
    AddNowait( void );

    bool
    IsNowaitAdded( void );

    /** Returns a directive for ending the loop region */
    virtual OPARI2_Directive*
    EndLoopDirective( const int lineno );

    /** Returns true for enddo and endparalleldo directives */
    virtual bool
    EndsLoopDirective( void );

private:
    /**
     * @brief Identifies clauses and their arguments
     */
    virtual void
    identify_clauses( void );

private:
    string m_arg_num_threads; /**< argument of the 'num_threads' clause */
    string m_arg_if;          /**< argument of the 'if' clause */
    string m_arg_reduction;   /**< argument of the 'reduction' clause */
    string m_arg_schedule;    /**< argument of the 'schedule' clause */
    string m_arg_collapse;    /**< argument of the 'collapse' clause */
    string m_user_name;       /**< name of named critical sections */
    bool   m_has_untied;      /**< true if has 'untied' clause */
    bool   m_has_ordered;     /**< true if hase 'ordered' clause */
    int    m_num_sections;    /**< number of sections */

    string::size_type m_indent;
    bool              m_nowait_added;

    virtual string
    generate_ctc_string( OPARI2_Format_t form );

    /** Static members */

    static opari2_omp_option s_omp_opt;          /**< OMP-specific cmd-line options */
    static stringstream      s_init_handle_calls;
    static int               s_num_regions;
    static const string      s_paradigm_prefix;

    static OPARI2_StrVStr_map_t   s_directive_clauses;
    static OPARI2_StrStr_pairs_t  s_outer_inner;
    static OPARI2_StrBool_pairs_t s_inner_clauses;
};
#endif
