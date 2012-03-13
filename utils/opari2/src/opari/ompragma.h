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
 *  @file       ompragma.h
 *  @status     alpha
 *
 *  @maintainer Dirk Schmidl <schmidl@rz.rwth-aachen.de>
 *
 *  @authors    Bernd Mohr <b.mohr@fz-juelich.de>
 *              Dirk Schmidl <schmidl@rz-rwth-aachen.de>
 *              Peter Philippen <p.philippen@fz-juelich.de>
 *
 *  @brief      This file contains the declaration of the classes used
 *              to store pragmas. The base class is \e OMPragma from
 *              which the classes \e OMPragma_C and \e OMPragma_F are
 *              derived to treat language specific issues.
 */

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

    /** @brief Removes all unnecessary commas. */
    void
    remove_commas();

    /** @brief Returns the arguments of a clause. */
    virtual string
    find_arguments( unsigned&          line,
                    string::size_type& pos,
                    bool               remove,
                    string             clause ) = 0;

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

    /** evaluate if there is a num_threads clause and store the arguments*/
    bool
    find_numthreads();

    /** evaluate if there is an if clause and store the arguments*/
    bool
    find_if();

    /** evaluate if there is an ordered clause and store the arguments*/
    bool
    find_ordered();

    /** evaluate if there is a schedule clause */
    bool
    find_schedule( string* reg_arg_schedule );

    /** evaluate if there is a reduction clause */
    bool
    find_reduction();

    /** evaluate if there is a collapse clause */
    bool
    find_collapse();

    /** evalueate if there is an untied clause present?*/
    bool
    find_untied( bool disableUntied );

    /** is the default data sharing changed, i.e. is default(none) or default(private) present*/
    bool
    changed_default();

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
    /** argument of the num_threads clause*/
    string            arg_num_threads;
    /** argument of the if clause*/
    string            arg_if;
    /** argument of the reduction clause*/
    string            arg_reduction;
    /** argument of the  schedule clause*/
    string            arg_schedule;
    /** argument of the collapse clause*/
    string            arg_collapse;

    /** all lines*/
    vector<string> lines;

private:
    virtual string
    find_next_word() = 0;
    virtual bool
    find_word( const string       word,
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
    virtual string
    find_arguments( unsigned&          line,
                    string::size_type& pos,
                    bool               remove,
                    string             clause );
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
    find_word( const string       word,
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
    virtual string
    find_arguments( unsigned&          line,
                    string::size_type& pos,
                    bool               remove,
                    string             clause );
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
    find_word( const string       word,
               unsigned&          line,
               string::size_type& pos );
    void
    remove_empties();
};

#endif
