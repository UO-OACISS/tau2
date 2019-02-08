/*
 * This file is part of the Score-P software (http://www.score-p.org)
 *
 * Copyright (c) 2013, 2014
 * Forschungszentrum Juelich GmbH, Germany
 *
 * This software may be modified and distributed under the terms of
 * a BSD-style license. See the COPYING file in the package base
 * directory for details.
 *
 */

/** @internal
 *
 *  @file      opari2_parser_c.h
 *
 *  @brief
 */

#ifndef OPARI2_PARSER_C_H
#define OPARI2_PARSER_C_H

class OPARI2_CParser
{
public:
    OPARI2_CParser( OPARI2_Option_t& options );

    ~OPARI2_CParser()
    {
    };

private:
    string            m_line;
    string::size_type m_pos;
    bool              m_in_comment;
    bool              m_in_string;
    bool              m_pre_cont_line;
    bool              m_require_end;
    bool              m_is_for;
    bool              m_block_closed;
    int               m_lineno;
    int               m_level;
    int               m_num_semi;
    string::size_type m_lstart;
    vector<string>    m_pre_stmt;
    vector<string>    m_end_stmt;
    stack<int>        m_next_end;
    string            m_current_file;
    string            m_infile;

    OPARI2_Option_t& m_options;
    ofstream&        m_os;
    ifstream&        m_is;


    string
    find_next_word( unsigned&          pline,
                    string::size_type& ppos );


/** @brief Check whether the current line is an extern function
    declaration */
    bool
    is_extern_decl( void );


/**
 * @brief  Instrument pragma directives.
 *
 * Preprocessor lines are passed to this function and checked, whether they are
 * pragmas.
 */
    bool
    process_prestmt( int               ln,
                     string::size_type ppos );


    bool
    handle_closed_block();

    void
    handle_preprocessor_directive( void );

    void
    handle_preprocessor_continuation_line( void );

    void
    handle_regular_line();

    bool
    get_next_line( void );


public:
/**  @brief Instrument directives / runtime APIs in C/C++ source file.
 */
    void
    process( void );
};
#endif //__PROCESS_C_H
