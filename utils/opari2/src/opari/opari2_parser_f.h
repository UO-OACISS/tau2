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
 *  @file      opari2_parser_f.h
 *
 *  @brief
 */

#ifndef OPARI2_PROCESS_F_H
#define OPARI2_PROCESS_F_H

#include <iostream>
using std::cerr;
using std::ostream;
using std::istream;
#include <vector>
using std::vector;
#include <stack>
using std::stack;

#include "opari2.h"
#include "opari2_directive_manager.h"
#include "opari2_directive.h"

/** @brief information about loops*/
typedef struct
{
    /** loop belongs to preceeding directive */
    bool   is_directive_loop;
    /** loop label */
    string label;
} LoopDescriptionT;


class OPARI2_FortranParser
{
public:

    OPARI2_FortranParser( OPARI2_Option_t& options );

    ~OPARI2_FortranParser()
    {
    };

/** @brief This function processes fortran files and searches for a
 *         place to insert variables, OpenMP pragmas and the begin and
 *         end of do loops which are needed to ensure correct
 *         instrumentation of parallel do constructs.*/
    void
    process( void );

private:
    string                        m_line;
    string                        m_lowline;
    bool                          m_unprocessed_line;
    int                           m_lineno;
    string                        m_curr_file;
    std::stack<OPARI2_Directive*> m_loop_directives;
    bool                          m_need_pragma;
    char                          m_in_string;
    bool                          m_normal_line;
    bool                          m_in_header;
    bool                          m_continuation;
    bool                          m_next_is_continuation;
    string                        m_sentinel;

    bool   m_offload_pragma;
    string m_offload_attribute;
    string m_current_offload_function;
    bool   m_offload_subroutine;
    bool   m_offload_function;

    // for endloop instrumentation
    stack<LoopDescriptionT> m_loop_stack;

    //    Line_type m_type_of_last_line;
    bool m_waitfor_loopstart;
    bool m_waitfor_loopend;
    int  m_lineno_loopend;

    OPARI2_Option_t& m_options;
    ofstream&        m_os;
    ifstream&        m_is;

    /**@brief Check if the line belongs to the header of a subroutine or function.
     *        After lines in the header, we can insert our variable definitions.*/
    bool
    is_sub_unit_header( void );

/**@brief check if this line is empty*/
    bool
    is_empty_line( void );

/**@brief check if this line is a comment line*/
    bool
    is_comment_line( void );

/**@brief check if this line is a continuation line*/
    bool
    is_continuation_line( void );

/**@brief check if this line starts a do loop*/
    bool
    is_loop_start( string& label );

/**@brief check if this line is the end of a do loop*/
    bool
    is_loop_end( void );

    void
    test_and_insert_enddo( void );

/** @brief Delete comments and strings before the lines are parsed
 *         to avoid finding keywords in comments or strings.*/
    void
    del_strings_and_comments( void );

    bool
    is_directive( void );

    bool
    get_next_line( void );

    void
    handle_line_directive( const string::size_type lstart );

    void
    handle_directive( void );

    bool
    is_free_offload_directive( void );

    void
    handle_free_offload_directive();

    void
    handle_offloaded_functions( void );

    void
    handle_normal_line();
};
#endif // __PROCESS_F_H
