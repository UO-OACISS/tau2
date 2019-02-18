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
 *  @file      opari2_directive.h
 *
 *  @brief     Abstract base class of all directives.
 */

#ifndef OPARI2_DIRECTIVE_H
#define OPARI2_DIRECTIVE_H

#include <string>
using std::string;
#include <vector>
using std::vector;
#include <set>
using std::set;
#include <sstream>
using std::stringstream;
#include <iostream>
using std::ostream;
#include <utility>
using std::pair;
using std::make_pair;
#include "opari2.h"

/**
 *  @brief Abstract base class to store and manipulate directive
 *         related data
 */
class OPARI2_Directive
{
public:
    OPARI2_Directive( const string&   fname,
                      const int       ln,
                      vector<string>& lines,
                      vector<string>& directive_prefix );

    virtual
    ~
    OPARI2_Directive( void )
    {
    }

public:
    /**********************************************************
    *                                                         *
    *                INTERFACE FOR DERIVED CLASSES            *
    *                                                         *
    **********************************************************/

    /**********************************************************
    * The following functions MUST be implemented by derived  *
    * classes                                                 *
    **********************************************************/

    /**
     * @brief Each paradigm must provide this method
     *
     * Each paradigm must keep a separate count of regions encountered
     * of this type. See the example for derived classes for a
     * standard implementation.
     */
    virtual void
    IncrementRegionCounter( void ) = 0;

    /**
     * @brief Generate region descriptors
     *
     * The function generate_descr_common() is provided to take care
     * of the generic descriptor information. See
     * OPARI2_DirectiveTemplate::GenerateDescr() for a standard
     * implementation.
     */
    virtual void
    GenerateDescr( ostream& os ) = 0;

    /**********************************************************
    * These functions may be implemented.                    *
    * Default implementations are provided below.            *
    **********************************************************/

public:
    /**
     * @brief Find the name of the directive
     */
    virtual void
    FindName( void )
    {
        find_name_common();
    };

private:
    /**
     * @brief Identifies clauses and their arguments
     */
    virtual void
    identify_clauses( void )
    {
    };

    /**********************************************************
    *                                                         *
    *             BASE CLASS MEMBERS AND METHODS              *
    *                                                         *
    **********************************************************/
private:
    /**
     * @brief Generates the part of the ctc-string which contains the
     *        paradigm specific information.
     *
     * The function generate_ctc_string_common() is provided to take
     * care of the generic source code region information.
     */
    virtual string
    generate_ctc_string( OPARI2_Format_t form )
    {
        return generate_ctc_string_common( form );
    };

public:
    /**
     * @brief Writes Directive for ending a loop
     */
    virtual OPARI2_Directive*
    EndLoopDirective( const int lineno )
    {
        return NULL;
    };

    /**
     * @brief Returns true for directives ending a loop region
     */
    virtual bool
    EndsLoopDirective( void )
    {
        return false;
    };

    /*******************************************************************
    * Furthermore there are some static functions implemented in the  *
    * derived classes, which are necessary or might be useful (and    *
    * should be named the same), when deriving a class for a new      *
    * paradigm.                                                       *
    *******************************************************************/

    /**
     * static OPARI2_ErrorCode
     * ProcessOption( string option );
     *
     * static uint32_t
     * String2Group( const string name );
     *
     * static void
     * GenerateHeader( ostream& os );
     *
     * static void
     * FinalizeDescrs( ostream& os );
     *
     * static void
     * GenerateInitHandleCalls( ostream&     os,
     *                          const string incfile = "" );
     */

protected:
    /** Name of currently processed file. This is not constant for one
        source file, as other files may have been included and the
        source code information is written correctly into the line
        directives */
    string m_filename;

    OPARI2_ParadigmType_t m_type;

    /** directive name */
    string m_name;

    /** A vector of pairs, consisting of clauses and their
        arguments */
    OPARI2_StrStr_map_t m_clauses;

    /* Global ID of instrumented region */
    int m_id;

    /** line numbers of the first and last line of the beginning and
        the end of the directive region in the sourcce file */
    int m_begin_first_line;
    int m_begin_last_line;
    int m_end_first_line;
    int m_end_last_line;

    /* Specifies whether there is an outer region */
    bool m_outer_reg;

    /* The Compile Time Context (CTC) string, its length and the
       variable it is stored in, in the instrumented file */
    string m_ctc_string;
    int    m_ctc_string_len;
    string m_ctc_string_variable;

    /** set of relevant region descriptors */
    set<int> m_descrs;

    /** the index of the currently parsed line within the directive statements */
    unsigned int m_pline;

    /** position index within the currently parsed line */
    string::size_type m_ppos;

    /** all lines */
    vector<string> m_lines;

    /** all original lines */
    vector<string> m_orig_lines;

    /** A series of tokens that identify a pragma/directive */
    vector<string> m_directive_prefix;

    /** In Fortran a common block is inserted in the output file with
     *  all region ids, the values are collected here and printed at
     *  the end*/
    static vector<int> s_common_block;

    /** In Fortran directives are usually ended with an "!$par end
     *  loop" directive.  In case of directives that refer to a
     *  directly following loop, the end of the loop also ends the
     *  code region for the directive. The end directive is inserted
     *  by OPARI2, if m_needs_end_loop_directive is true.*/
    bool m_needs_end_loop_directive;

    /*  The necessary directive for explicitly ending a loop
     *  region. */
    string m_end_loop_directive;

    /** directive of enclosing directive */
    OPARI2_Directive* m_enclosing;

    /** Static variables. This is information which is constant for
        the processed file and frequently accessed in the
        directives */

    /** pointer to outer directive */
    static OPARI2_Directive* s_outer;

    /** Region counter for all region types */
    static int s_num_all_regions;

    /** Language the input file is written in */
    static OPARI2_Language_t s_lang;

    /** Type of format (currently only needed for Fortran,
        distinguishes between fixed-form and free-form Fortran */
    static OPARI2_Format_t s_format;

    /** Specifies whether the source code information via #line
        directives is to be inserted */
    static bool s_keep_src_info;

    /** Specifies whether the file was already preprocessed */
    static bool s_preprocessed_file;

    /** A unique id that is needed to distinguish the initialization
        routines of the different compile units of the target
        application. It should not be placed here as it is strictly
        speaking not part of the necessary information of a directive,
        but it is used and needed here in the
        GenerateInitHandleCalls... functions. */
    static string s_inode_compiletime_id;

public:
    /** @brief Return 'type', for dynamic_cast */
    OPARI2_ParadigmType_t
    GetParadigmType( void );

    /** @brief Return the directive name */
    string&
    GetName( void );

    /** @brief Returns whether a clause is present */
    bool
    HasClause( const string& clause );

    /** @brief Returns the argument of a clause */
    string
    GetClauseArg( const string& clause );

    /** @brief Deletes comments in directive lines */
    void
    DelInlineComments( void );

    /** @brief Returns the filename of the file the directive belongs to */
    string&
    GetFilename( void );

    /* Return beginning line number of directive */
    int&
    GetLineno( void );

    /** @brief Update line information after the parser reaches the end of
        this directive region */
    void
    SetEndLineno( const int endline_begin,
                  const int endline_end );

    /** @brief Prints the #line directive to update the source code
        information */
    void
    ResetSourceInfo( ostream& os );

    /** @brief Enter region */
    void
    EnterRegion( bool new_outer = false,
                 bool save_on_vec = true );

    /** @brief Initialize region information. */
    void
    InitRegion( bool outer = false );

    /** @brief Initialize region information. */
    void
    InitRegion( OPARI2_Directive* parent,
                bool              outer = false );

    /** @brief Exit region */
    int
    ExitRegion( bool end_outer = false );

    /** @brief Inserts the id of a region nested inside another region */
    void
    InsertDescr( int descr );

    /** @brief Check if the directive has accociated descriptors */
    bool
    DescrsEmpty();

    /** @brief Returns the id of the directive region */
    int
    GetID( void )
    {
        return m_id;
    };

    /** @brief Returns the ctc-string */
    string
    GetCTCStringVariable( void )
    {
        return m_ctc_string_variable;
    };

    /** @brief Sets all necessary static information */
    static void
    SetOptions( OPARI2_Language_t lang,
                OPARI2_Format_t   form,
                bool              keep_src,
                bool              preprocessed,
                const string      id );

    /** @brief Returns the file specific identifier to distinguish different
        compilation units */
    static string
    GetInodeCompiletimeID( void )
    {
        return s_inode_compiletime_id;
    }

    /**
     * @brief Generate a function to initialize all region handles.
     */
    static void
    GenerateInitHandleCalls( ostream&            os,
                             const string        incfile,
                             const string        paradigm_prefix,
                             const stringstream& init_handle_calls,
                             const int           num_regions );

    /** @brief Fortran specific finalization of descriptors */
    static void
    FinalizeFortranDescrs( ostream& os );

    void
    FinishRegion( void );

    /** @brief Returns value of m_needs_end_loop_directive.*/
    bool
    NeedsEndLoopDirective( void );

    /** @brief Sets value of m_needs_end_loop_directive.*/
    void
    NeedsEndLoopDirective( bool val );

    /** @brief Prints directive to stream and updates source information */
    void
    PrintDirective( ostream&      os,
                    const string& adds = "" );

    /** @brief Prints directive without updating source information */
    void
    PrintPlainDirective( ostream&      os,
                         const string& adds = "" );

protected:
    /**
     * @brief Generates the part of the ctc-string which contains the
     *        generic region information
     */
    string
    generate_ctc_string_common( OPARI2_Format_t form,
                                string          specific_part = "" );

    /**
     * @brief Generate the generic part of the region descriptors
     */
    void
    generate_descr_common( ostream& os );


    /**
     * @brief Generic part of identifying a directive name
     */
    void
    find_name_common( void );


    /**
     * @brief Removes all unnecessary commas
     */
    void
    remove_commas( void );

    /**
     * @brief Returns the arguments of a clause.
     */
    string
    find_arguments( unsigned&          line,
                    string::size_type& pos,
                    bool               remove,
                    string             clause );

    /**
     * @brief Removes empty lines */
    bool
    remove_empty_line( unsigned& line );

    /**
     * @brief Check if 'word' can be found in directive 'm_lines'.
     *
     * If true, save the line number within 'm_lines' in 'line' and
     * offset within the line in 'pos'.
     *
     * @return  true  if 'word' is found in m_lines.
     *		false otherwise.
     */
    bool
    find_word( const string       word,
               unsigned&          line,
               string::size_type& pos );

    /**
     * @brief Find the next word in directive lines, starting from
     *       'm_pline' and 'm_ppos'.
     */
    string
    find_next_word( void );

    /**
     * @brief Takes care of moving arguments of clauses for split
     *        directives to the newly created inner directive.
     *
     * When combined directives are split the clauses are either kept
     * at the outer directive or moved to the inner directive. This
     * function takes care of moving all arguments of clauses
     * belonging to the inner directive.
     */
    void
    fix_clause_arg( vector<string>&    outer,
                    vector<string>&    inner,
                    unsigned&          line,
                    string::size_type& pos );


    /**
     * @brief Remove empty lines in directive 'lines'.
     */
    void
    remove_empties( void );

public:
/**
 * @brief Split combined parallel and worksharing constructs.
 *
 * They are split into two separate pragmas to allow the insertion of
 * POMP function calles between the parallel and the worksharing
 * construct.  The clauses need to be matched to the directive they
 * belong to. this is a template function as it needs to return the
 * right derived type of OPARI2_Directive*.
 */
    template<typename T>
    T*
    SplitCombinedT( OPARI2_StrStr_pairs_t&  outer_inner,
                    OPARI2_StrBool_pairs_t& inner_clauses )
    {
        remove_commas();

        vector<string>    inner_lines;
        string            sentinel = m_directive_prefix[ 0 ];
        string::size_type slen     = sentinel.length();


        // make empty copy with continuation characters
        if ( s_lang & L_C_OR_CXX )
        {
            for ( unsigned i = 0; i < m_lines.size(); ++i )
            {
                inner_lines.push_back( string( m_lines[ i ].size(), ' ' ) );
                if ( i != m_lines.size() )
                {
                    inner_lines[ i ][ inner_lines[ i ].size() - 1 ] = '\\';
                }
            }
        }
        else if ( s_lang & L_FORTRAN )
        {
            for ( unsigned i = 0; i < m_lines.size(); ++i )
            {
                inner_lines.push_back( string( m_lines[ i ].size(), ' ' ) );

                string::size_type s = m_lines[ i ].find( sentinel );

                // & continuation characters
                string::size_type com = m_lines[ i ].find( "!", s + slen );
                if ( com != string::npos )
                {
                    --com;
                }
                string::size_type amp2 = m_lines[ i ].find_last_not_of( " \t", com );
                if ( m_lines[ i ][ amp2 ] == '&' )
                {
                    inner_lines[ i ][ amp2 ] = '&';
                }
                string::size_type amp1 = m_lines[ i ].find_first_not_of( " \t", s + slen );
                if ( m_lines[ i ][ amp1 ] == '&' || m_lines[ i ][ amp1 ] == '+' )
                {
                    inner_lines[ i ][ amp1 ] = m_lines[ i ][ amp1 ];
                }
            }
        }

        // copy sentinel/directive_prefix
        unsigned          line = 0;
        string::size_type pos  = 0;

        for ( vector<string>::iterator it = m_directive_prefix.begin(); it != m_directive_prefix.end(); ++it )
        {
            for ( size_t i = 0; i < m_lines.size(); ++i )
            {
                pos = m_lines[ i ].find( *it );
                if ( pos != string::npos )
                {
                    inner_lines[ i ].replace( pos, ( *it ).length(), *it );
                }
            }
        }

        // fix pragma name
        for ( OPARI2_StrStr_pairs_t::iterator it = outer_inner.begin(); it != outer_inner.end(); ++it )
        {
            line = pos = 0;
            if ( find_word( ( *it ).first, line, pos ) )
            {
                string name = ( *it ).second;
                string blanks( name.length(), ' ' );

                if ( find_word( name, line, pos ) )
                {
                    m_lines[ line ].replace( pos, name.length(), blanks );
                    inner_lines[ line ].replace( pos, name.length(), name );
                }
            }
        }

        // fix pragma clauses
        for ( OPARI2_StrBool_pairs_t::iterator it = inner_clauses.begin(); it != inner_clauses.end(); ++it )
        {
            string name     = ( *it ).first;
            bool   has_args = ( *it ).second;
            string blanks( name.length(), ' ' );

            line = pos = 0;

            while ( find_word( name, line, pos ) )
            {
                m_lines[ line ].replace( pos, name.length(), blanks );
                inner_lines[ line ].replace( pos, name.length(), name );
                pos += name.length();
                if ( has_args )
                {
                    fix_clause_arg( m_lines, inner_lines, line, pos );
                }
            }
        }

        T* inner = new T( m_filename, m_begin_first_line,
                          inner_lines, m_directive_prefix );

        // final cleanup
        remove_empties();
        inner->remove_empties();

        return inner;
    }
    bool active;
};

#endif  //OPARI2_DIRECTIVE_H
