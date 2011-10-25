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
#ifndef OMPRAGMA_H
#define OMPRAGMA_H

#include <string>
using std::string;
#include <vector>
using std::vector;

/** @brief Class to store and manipulate openmp pragma related data*/

class OMPragma
{
public:
    /** initialize pragma */
    OMPragma( const string& f,
              int           l,
              int           pl,
              int           pp,
              bool          a )
        : filename( f ), lineno( l ), pline( pl ), ppos( pp ), asd( a )
    {
    }
    /** find and store the pragma name */
    void
    find_name();

    /** find and store the value of the num_threads clause, if present*/
    void
    find_numthreads();

    /** is a nowait clause present?*/
    bool
    is_nowait();

    /** is a copyprivate clause present*/
    bool
    has_copypriv();

    /** returns value in brackets, if present, "" otherwise*/
    string
    find_sub_name();

    /** add a nowait to a directive*/
    virtual void
    add_nowait() = 0;

    /** add descriptors at the right place*/
    virtual void
    add_descr( int n ) = 0;

    /** split compined parallel worksharing constructs in two seperate statement split compined parallel worksharing constructs in two seperate statements*/
    virtual OMPragma*
    split_combined() = 0;

    virtual
    ~
    OMPragma()
    {
    }
    /** filename */
    string            filename;
    /** line number */
    int               lineno;
    /** current parsing line */
    unsigned          pline;
    /** current parsing position*/
    string::size_type ppos;
    /** compiler allowes defines on pragma lines?*/
    bool              asd;
    /** pragma name*/
    string            name;
    /** expression of the num_threads clause*/
    string            num_threads;
    /** all lines*/
    vector<string>    lines;

private:
    virtual string
    find_next_word() = 0;
    virtual bool
    find_word( const char*        word,
               unsigned&          line,
               string::size_type& pos ) = 0;
};

/** @brief data and functions for the fortran specific issues of pragma handling*/
class OMPragmaF : public OMPragma
{
public:
    /** creates an OMPragmaF object*/
    OMPragmaF( const string& f,
               int           l,
               int           p,
               const string& line,
               int           pomp,
               bool          a )
        : OMPragma( f, l, 0, p, a ), slen( 5 + pomp )
    {
        lines.push_back( line );
        sentinel = pomp ? "$pomp" : "$omp";
    }
    virtual void
    add_nowait();
    virtual void
    add_descr( int n );
    virtual OMPragma*
    split_combined();

private:
    virtual string
    find_next_word();
    virtual bool
    find_word( const char*        word,
               unsigned&          line,
               string::size_type& pos );
    void
    remove_empties();

    string sentinel;
    int    slen;
};

/** @brief data and functions for the C specific issues of pragma handling*/
class OMPragmaC : public OMPragma
{
public:
/** creates an OMPragmaC object*/
    OMPragmaC( const string&   f,
               int             l,
               int             pl,
               int             pp,
               vector<string>& stmts,
               bool            a ) : OMPragma( f, l, pl, pp, a )
    {
        lines.swap( stmts );
    }
    virtual void
    add_nowait();
    virtual void
    add_descr( int n );
    virtual OMPragma*
    split_combined();

private:
    virtual string
    find_next_word();
    virtual bool
    find_word( const char*        word,
               unsigned&          line,
               string::size_type& pos );
    void
    remove_empties();
};

#endif
