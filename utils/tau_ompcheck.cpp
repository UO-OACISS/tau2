#include "pdbAll.h"
/****************************************************************************
**			TAU Portable Profiling Package			                               **
**			http://www.cs.uoregon.edu/research/paracomp/tau                    **
*****************************************************************************
**    Copyright 2006                                    						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
**    Research Center Juelich, Germany                                     **
****************************************************************************/
/***************************************************************************
**	File 		: tau_ompcheck.cpp			                                       **
**	Description 	: OMP Directive Checker                                  **
**	Author		: Scott Biersdorff			                                     **
**	Contact		: scottb@cs.uoregon.edu 	                                   **
***************************************************************************/
#include <stdio.h>
#include <algorithm>
#include <cctype>
#include <string>
#include <list>
#include <fstream>
using namespace std;

//---------------------------------------------------------------------------

char* file;
char* source;

int pragma_id;
int lang;
int verbosity;
ostream* output;


enum {Fortran, C};
enum {Standard, Verbose, Debug};

/***************************************************************
 *
 * Class Directive: Holds information about each individual
 * directive
 * 
 *************************************************************/         
class Directive
{
  const int type;
  int line;
  int col;
  int depth;
  const pdbStmt* block;
  public:

  /*
   * Constructor IN: 
   * int t         : the type of this directive (ie. do/for/paralell
   *                 opening/closing).
   * int l         : the line number in the source file where the
   *                 direcitve is declared.
   */
  /*Directive(Directive& d) : type(d.getType())
	{
    line = d.getLine();
		col =  d.getCol();
		depth = d.getDepth();
		block = d.getBlock();
	}*/
  Directive(const int t, const int l) : type(t)
  {
    //type = t;
		line = l;
	  //printf("CONSTRCTOR (1), type: %d\n", type);
	  if (type > 4 || type < -4)
		{
		  //printf("ERROR reading type.\n");
			exit(1);
		}
  }
  /*
   * Constructor IN: 
   * Directive d      : An Opening Directive, we will create a
   *                    matching closing Directive
   * int l            : the line number in the source file where the
   *                    direcitve is declared.
   * const pdbStmt* s : The pdbStmt which follows the opening
   *                    directive - a block statement.
   */
  Directive(Directive d, const int l, const pdbStmt* s) : type(d.getType())
  {
    //type = d.getType();
    line = s->stmtEnd().line();
    col = s->stmtEnd().col();
    if (type > 1 || type < -1 )
      col++;
    depth = l;
    block = s;
	  //printf("CONSTRCTOR (2), type: %d\n", type);
		if (type > 4 || type < -4)
		{
		  //printf("ERROR reading type.\n");
			exit(1);
		}
  }
  /*
   * Constructor IN: 
   * Directive d      : An Opening Directive, we will create a
   *                    matching closing Directive
   * int t            : the type of the Opening Directive.
   * int l            : the line number in the source file where the
   *                    direcitve is declared.
   * const pdbStmt* s : The pdbStmt which follows the opening
   *                    directive - a block statement.
   */
  Directive(Directive d, const int t, const int l, const pdbStmt* s) : type(-t)
  {
    //type = -t;
		//printf("hello: %d\n", s->id());
		int k = s->kind();
		//printf("hello: %d\n", s->id());
    line = s->stmtEnd().line();
    col = s->stmtEnd().col();
    if ((type > 1 || type < -1))
      col++;
    depth = l;
    block = s;
	  //printf("CONSTRCTOR (3), type: %d\n", type);
		if (type > 4 || type < -4)
		{
		  //printf("ERROR reading type.\n");
			exit(1);
		}
  }  
  /*
   * Constructor IN: 
   * Directive d      : A Directive on which base this new one
   * int l            : the line number in the source file where the
   *                    direcitve is declared.
   */
  Directive(Directive d, const int l) : type(d.getType())
  {
    //type = d.getType();
    line = d.getLine();
    depth = l;
	  //printf("CONSTRCTOR (4), type: %d\n", type);
		if (type > 4 || type < -4)
		{
		  //printf("ERROR reading type.\n");
			exit(1);
		}
  }
  int getType() const 
  { return type; }
  int getCol() const
  { return col; }
  int getLine() const
  { return line; }
  const pdbStmt* getBlock()
  { return block; }
  int getDepth() const 
  { return depth; }
  /*void setType(int t)
  {  type = t; }*/
  void setCol(int l)
  { col = l; }
  void setLine(int l)
  { line = l; }
  void setBlock(const pdbStmt* s)
  { block = s; }
  void setDepth(int d)
  { depth = d; }
};

/*
 * Operator<: 
 * Compares two directives ordered by the place inside the source files
 * IE. a < b iff a is declared before b in the source file.
 *
 * IN:
 * Directive& a
 * Directive& b
 */
bool operator< (const Directive& a, const Directive& b)
{
  if (a.getLine() == b.getLine())
    return a.getCol() < b.getCol();
  else
    return a.getLine() < b.getLine();
}
/***************************************************************
  *
  * Class CompleteDirective: A collection of functions that finds 
  * and completes unclosed directives within a routine.
  * 
  *************************************************************/
class CompleteDirectives
{
  /* A complete list of Directives in the routine. */
  list<Directive> directives;
  /* A list of currently open Directives at this stage in the routine. */
  list<Directive> openDirectives;
  /* A list of Directives that should be inserted in this routine. */
  list<Directive> addDirectives;
  int language;
  /* the line number at the begining this routine. */
  int beginRoutine;
  /* the line number at the end this routine. */
  int endRoutine;

  static const int STATE_CLOSED = 0;
  static const int STATE_OPEN = 1;
  static const int STATE_EXPECTING = 2;
  static const int STATE_ADD = 3;

  public:
    
 /*
  * Constructor
  * IN:
  * char* a         : the name of the file this routine is defined in.
  * pdbRoutine* ro  : the routine we are to consider.
  * int lan         : the language this program is written is.
  *
  */
  CompleteDirectives(char* a, pdbRoutine* ro, int lan)
  {
    language = lan;
    findOMPDirectives(a,ro);
  }
 /*
  * findFRoutineEnd : find the last line of a fortran routine.
  * IN:
  * pdbRoutine* ro  : the routine we are to consider.
  *
  * OUTPUT:
  * pdbLoc*         : the location of the end of this routine. This
  *                   is found by finding that last return or stop location in
  *                   the routine.
  */
  pdbLoc* findFRoutineEnd(pdbRoutine* ro)
  {
    if (verbosity == Debug)
      cerr << "finding end of the routine." << endl;
    pdbRoutine::locvec l = ro->returnLocations();
    if (verbosity == Debug)
      cerr << "got the return location." << endl;
    
    //search stop routine if the langage is fortran
    if (language == Fortran)
    {
      pdbFRoutine* fro = ((pdbFRoutine*) ro);
      pdbRoutine::locvec s = fro->stopLocations();
      if (verbosity == Debug)
        cerr << "got the stop location." << endl;
      l.insert(l.end(), s.begin(), s.end());
    }
    pdbLoc* lastReturn = *(l.begin());
    if (verbosity == Debug)
      cerr << "initialized size: " << l.size() << " first: " << (*l.begin())->line() << endl;
    
    //find the last return/stop location
    for (pdbRoutine::locvec::iterator loc = l.begin(); loc != l.end() ; loc++)
    {
      if (verbosity == Debug)
        cerr << "one return location: " << (*loc)->line() << endl;
      if (lastReturn->line() < (*loc)->line() || (lastReturn->line() == (*loc)->line() && lastReturn->col() < (*loc)->col()))
      {
        lastReturn = *loc;
      }
    }
    return lastReturn;
  }
 /*
  * findOMPDirectives : populates the directives list with directives decleared
  *                     in this routine.
  * IN:
  * char* a         : the name of the file where the routine resides.
  * pdbRoutine* ro  : the routine we are to consider.
  *
  */
  void findOMPDirectives(char* a, pdbRoutine* ro)
  {
    int i;
    PDB pdbHead(a);
    if (language == Fortran)
    {
      endRoutine = findFRoutineEnd(ro)->line();
      beginRoutine = ro->location().line();
    }
    else
    {  
      endRoutine = ro->body()->stmtEnd().line();
      beginRoutine = ro->location().line();
    }
    if (verbosity == Debug)
      cerr <<  "Routine lines: " << beginRoutine << "-" << endRoutine << endl;
    
    // iterate through the pragma vector added to the list each one that appears
    // with this routine.
		//cerr << "looking for prgmas/comments" << endl;
    for (PDB::pragmavec::iterator r = pdbHead.getPragmaVec().begin();
          r!=pdbHead.getPragmaVec().end(); r++)
    { 
					//cerr << "found pragmas" << endl;
		  if (getDirectiveType(**r) != 1 && getDirectiveType(**r) != -1)
			{
				if (verbosity == Debug)
					cerr << "<" << (**r).prBegin().line() << "  "  << getDirectiveType(**r) << ">" << endl;
				if (getDirectiveType(**r) != 0 && (**r).prBegin().line() > beginRoutine && (**r).prBegin().line() < endRoutine)
				{
					directives.push_back(Directive(getDirectiveType(**r),(**r).prBegin().line()));  
					if (verbosity == Debug)
						cerr << directives.back().getLine() << "  "  << directives.back().getType() << endl;
				}
			}
    }
    if (verbosity == Debug)
      cerr << "----------------" << endl;
  }

 /*
  * getDirectiveType: return the type of this directive.
  * IN:
  * pdbPragma p     : the pragma/directive.
  *
  * OUTPUT:
  * int             : The type of the routine:
  *                   1 - open parallel 
  *                   2 - open parallel do
  *                   3 - open do
  *                   4 - open for
  *                   closing directive type = -(opening directive type)
  *
  */
  int getDirectiveType(pdbPragma p)
  {
    //Transform pragma text to lower case
		//cerr << "in getDirectiveType." << endl;
    string text = p.text();
    std::transform (text.begin(), text.end(), text.begin(), (int(*)(int)) tolower);
    
    if (language == Fortran)
    {
		  int startChar = 0;
		  startChar = text.find("$omp");
			if (startChar != string::npos)
			{ 
			  /*cerr << "evaulating text=" << text << endl;
				cerr << "starting char search at: " << startChar << endl;
				cerr << "evaluating substr='" << text.substr(startChar,16) << "'" <<
				endl;*/
				//Match OMP PARALLEL directive
				if (text.substr(startChar,16) == "$omp parallel do")
					return 2;
				else if (text.substr(startChar,13) == "$omp parallel")
					return 1;
				//Match OMP DO directive
				else if (text.substr(startChar,7) == "$omp do")
					return 3;
				else if (text.substr(startChar,8) == "$omp for")
					return 4;
				//Match OMP END PARALLEL directive
				else if (text.substr(startChar,20) == "$omp end parallel do")
					return -2;
				else if (text.substr(startChar,17) == "$omp end parallel")
					return -1;
				//Match OMP DO directive
				else if (text.substr(startChar,11) == "$omp end do" ||
				text.substr(startChar,10) == "$omp enddo")
					return -3;
				else if (text.substr(startChar,12) == "$omp end for")
					return -4;
				else
					return 0;
			}
    }
    else
    {
      if (text.substr(0,23) == "#pragma omp parallel do")
        return 2;
      if (text.substr(0,20) == "#pragma omp parallel")
        return 1;
      else if (text.substr(0,14) == "#pragma omp do")
        return 3;
      else if (text.substr(0,15) == "#pragma omp for")
        return 4;
      else if (text.substr(0,27) == "#pragma omp end parallel do")
        return -2;
      else if (text.substr(0,24) == "#pragma omp end parallel")
        return -1;
      else if (text.substr(0,18) == "#pragma omp end do")
        return -3;
      else if (text.substr(0,19) == "#pragma omp end for")
        return -4;
      else
        return 0;
    }

  }
 /*
  * findOMPStmt     : initalizes the recursive processing of each statemet in
  *                   the routine.
  * IN:
  * pdbStmt *s      : The statement to begin processing.
  * PDB& pdb        : The pdb root object.
  *
  * OUTPUT:
  * list<Directive>&: A list of all the directives that need to be added to this
  *                   routine to complete each open directive.
  */
  list<Directive>& findOMPStmt(const pdbStmt *s, PDB& pdb)
  {
    /* We need to make sure the entire routine is processed to insure this
     * we will create two statements. The first as the head of each routine
     * enclosing all the statement in the routine. The second is placed at the
     * close of the routine.
     *
     *      Head statement (created) -> s1 -> ... -> s2
    */
		//cerr << "in findOMPStmt" << endl;
    pdbStmt head(-1);
    //pdbStmt tail(-1);
    //pdbStmt state(*s);
    pdbFile file(-1);
    head.stmtBegin(pdbLoc(&file,beginRoutine,0));
    head.stmtEnd(pdbLoc(&file,endRoutine,0));
    head.nextStmt(s);
    head.downStmt(NULL);
    head.extraStmt(NULL);
    /*tail.stmtBegin(pdbLoc(&file,endRoutine,0));
    tail.stmtEnd(pdbLoc(&file,endRoutine,0));
    tail.nextStmt(NULL);
    tail.downStmt(NULL);
    tail.extraStmt(NULL);
    state.nextStmt(&tail);
    state.downStmt(s);*/
    static list<Directive> emptyDirectives;
    if (directives.size() == 0 and openDirectives.size() == 0) {
		  cerr << "no directives to find.\n" << endl;
      return emptyDirectives;
    }
		else {
		  //cerr << "finding OMP stmts...\n" << endl;
		  return findOMPStmt(STATE_CLOSED, &head,s,0, pdb);
		}
  }
 /*
  * findOMPStmt     : recursivesly processes each statemet in
  *                   the routine opening and closing directives and
  *                   inserting directive when need.
  * IN:
  * int state       : The Logical state we are in.
  * pdbStmt *s      : The statement of which to begin processing.
  * pdbStmt *block  : The statement that defines the block the s statement is
  *                   in.
  * int loop        : the loop depth of this statement.
  * PDB& pdb        : The pdb root object.
  *
  * OUTPUT:
  * list<Directive>&: A list of all the directives that need to be added to this
  *                   routine to complete each open directive.
  */
  list<Directive>& findOMPStmt(const int state, const pdbStmt *s, const pdbStmt *block, int loop, PDB& pdb)
  {
    if (verbosity == Debug)
			printLocation(s, block, loop);

    int currentLine = s->stmtBegin().line();
		Directive *nextDirective = &directives.front();
	  //int depth = loop - openDirectives.front().getDepth();
			if (state == STATE_CLOSED)
			{
    		if (verbosity == Debug)
				  printf("In state: STATE_CLOSED\n");
			  //if we have encountered an OMP directive
				if (directives.size() > 0)
				{
					if (!checkForDirective(state, s, block, loop, pdb))
					{
						//move to the next stmt
						return gotoNextStmt(STATE_CLOSED, s, block, loop, pdb);
					}
					//printf("nothing else to do...\n");
					return gotoNextStmt(STATE_CLOSED, s, block, loop, pdb);
				}
				else
				{
          return addDirectives;
				}
      }
			else if (state == STATE_OPEN)
			{
				int depth = loop - openDirectives.front().getDepth();
				if (verbosity == Debug)
				{
					printf("In state: STATE_OPEN\n");
					printf("last open directive: %d type: %d\n",
					openDirectives.front().getLine(), openDirectives.front().getType());
					printf("directive depth: %d\n", openDirectives.front().getDepth());
					printf("loop depth: %d\n", depth);
        }
    		if (openDirectives.size() == 0)
				{
          return findOMPStmt(STATE_CLOSED, s, block, loop, pdb);
				}
				if (directives.size() > 0 && (directives.front().getType() +
				openDirectives.front().getType() == 0))
				{
				  //printf("checking for directives...\n");
					checkForDirective(state, s, block, loop, pdb);
				}
				else
				{
					if (depth == 0 && openDirectives.front().getLine() <
					(s->stmtBegin().line() - 1))
					{
					  printf("we should see a directive.\n");
					  //parallel do open
						if (openDirectives.front().getType() != 1)
						{
							printf("closing do loop.\n");
							addDirectives.splice(addDirectives.end(),findOMPStmt(STATE_EXPECTING, s, block, loop, pdb));
							return addDirectives;
						}
						// end of routine or function
						else if (followingStmt(s)->kind() == pdbStmt::ST_FRETURN ||
						followingStmt(s)->kind() == pdbStmt::ST_FEXIT)
						{
							printf("closing parallel\n");
							addDirectives.splice(addDirectives.end(),findOMPStmt(STATE_EXPECTING, s, block, loop, pdb));
							//printf("back in open...\n");
							return addDirectives;
						}
						// wait to end parallel directive
						else
						{
						  return gotoNextStmt(STATE_OPEN, s, block, loop, pdb);
						}  
					}
					else
					{ 
					  //if (s->nextStmt() != NULL) { printf("NOT null"); }
						return gotoNextStmt(STATE_OPEN, s, block, loop, pdb);
					}
				}
			}
			else if (state == STATE_EXPECTING)
			{
    		if (verbosity == Debug)
				  printf("In state: STATE_EXPECTING\n");
			  
				//is there already a directive closing the one open?
				if (directives.front().getLine() == currentLine + 1 &&
				directives.front().getType() + openDirectives.front().getType() == 0)
        {
          directives.pop_front();
					openDirectives.pop_front();
					if (openDirectives.size() == 0)
					{
					  //printf("found.\n");
            return gotoNextStmt(STATE_CLOSED, s, block, loop, pdb);
					}
					else
					{
					  //printf("wanting.\n");
            return gotoNextStmt(STATE_OPEN, s, block, loop, pdb);
					}
				}
				else
				{
					addDirectives.splice(addDirectives.end(),findOMPStmt(STATE_ADD, s, block, loop, pdb));
					//printf("back in expecting...\n");
					return addDirectives;
				}  
			}
			else if (state == STATE_ADD)
			{
    		if (verbosity == Debug)
				{
				  printf("In state: STATE_ADD\n");
          printf("open type: %d", openDirectives.front().getType()); 
				}
				addDirectives.push_front(Directive(openDirectives.front(),
				openDirectives.front().getType(), loop, s));
				//printf("open size: %d\n", openDirectives.size());
				openDirectives.pop_front();
				//printf("open size: %d\n", openDirectives.size());
				if (openDirectives.size() < 1)
				{
					//printf("all directives closed.\n");
					return gotoNextStmt(STATE_CLOSED, s, block, loop, pdb);
				}
				else
				{
					return gotoNextStmt(STATE_OPEN, s, block, loop, pdb);
				}
			}
      else 
			{
			  return addDirectives;
			}
	  while (!directives.empty() and nextDirective->getLine() < s->stmtBegin().line())
		{
      if (verbosity == Debug)
			  printf("passed directive, removing it.\n");

			directives.pop_front();
		  nextDirective = &directives.front();
		}
			return addDirectives;
		/*
    while ( s->stmtBegin().line() >=
			 directives.front().getLine() && ( openDirectives.size() > 0 ||
			 directives.size() > 0))
		{
#ifdef DEBUG
			cerr << "open type: " <<  openDirectives.front().getType() << endl;		
#endif 
			while (
								directives.front().getType() == 1 ||
								directives.front().getType() == -1)
				{
					while (directives.front().getType() == 1)
					{
#ifdef DEBUG
					printf("h");
#endif
						if (verbosity == Debug)  
							cerr << "Opening Parallel OMP Directive." << endl;
						//openDirectives.push_front(Directive(directives.front(), loop, block));
						directives.pop_front();
					}
					while (directives.size() != 0 && directives.front().getType() == -1)
					{
						//if (openDirectives.size() == 1)
							//cerr << "ERROR: Superfluous closing OMP Directive on line: " << directives.front().getLine() << endl;
						if (verbosity == Debug)  
							cerr << "Closing Parallel OMP Directive." << endl;
						
						directives.pop_front();
					}
				}
				//if (verbosity == Debug)  
					//cerr << "checking whether directives have closed." << endl;

				//if we have found all the directives
				if (directives.size() == 0)
				{
					if (openDirectives.size() == 0)
					{
						if (verbosity == Debug)  
							cerr << "all directives have been closed." << endl;
					}
					else
					{
						//if (verbosity == Debug)  
							//cerr << "OMP Directive on line: " << openDirectives.front().getLine() << " has not been closed."<< endl;
					}
				}
				//Begin searching through statements
				//If this statement is inside a OMP Directive
				//printf("directive type: %d", directives.front().getType());
				while (directives.front().getLine() && directives.front().getType() < 0 &&
							openDirectives.size() != 0)
				{
					if (openDirectives.size() != 0)
					{
					  printf("inside if..\n");
						//atempt to close directive
						if (openDirectives.front().getType() + directives.front().getType() == 0)
						{
							if (verbosity == Debug)  
								cerr << "Closing OMP Directive type: " << directives.front().getType() << endl;
							openDirectives.pop_front();
							directives.pop_front();
						}//if
						else 
						{  
							cerr << "ERROR: Mismatched closing OMP Directive on line: " << directives.front().getType() << endl;
							if (verbosity >= Verbose)  
								cerr << "ERROR: mismatched closing OMP Directive on line: " <<
								s->stmtBegin().line() << " type: " << directives.front().getType() << endl;
							openDirectives.pop_front();
							directives.pop_front();
						}//else
					}//if 
					else  
						cerr << "ERROR: Superfluous closing OMP Directive on line: " << directives.front().getLine() << endl;
			  }//end while
    
			if (openDirectives.size() != 0 && openDirectives.front().getDepth() ==
            loop && openDirectives.front().getType() > 1 && verbosity == Debug)
        cerr << "Could be a missing directive." << endl;
      
			int nextStmtLine = block->stmtEnd().line();
		
      if (openDirectives.size() > 0 && openDirectives.front().getDepth() ==
			loop)
			  printf("shoud we consider closing directive?\n");

			//Are we expecting any more pragmas to be closed before this statement?
      while (openDirectives.size() != 0 && 
				openDirectives.front().getDepth() == loop && 
				openDirectives.front().getType() > 1 && 
				( !((nextStmtLine-1) >= directives.front().getLine() &&
      		directives.front().getType() + openDirectives.front().getType() == 0
					)
				)
			)
      {
        if (verbosity >= Verbose)
				{
          cerr << "We are expecting there to be a directive closing the one on line: " << openDirectives.front().getLine() << endl;
					cerr << "...in loop " << loop << ", directive in loop " << openDirectives.front().getDepth() << endl;
				}
        //Create a closing for/do pragma
        //createDirective(s,openDirectives.front());
        
        //if (s->downStmt() == NULL)
        //  cerr << "Directives opening a loop are only allowed immediately before the loop body." << endl;
        //else
        addDirectives.push_front(Directive(openDirectives.front(), openDirectives.front().getType(), loop, openDirectives.front().getBlock()));
        
        openDirectives.pop_front();
				if (directives.size() == 0)
				  return addDirectives;
      }
      
      //Are we expecting to close any parallel OMP directives.
      while (openDirectives.size() != 0 && openDirectives.front().getDepth() ==
            loop && openDirectives.front().getType() == 1 && s->nextStmt() == NULL)
      {
        //cerr << "We are expecting there to be a parallel directive closing the one on line: " << openDirectives.front().getLine() << endl;
        //Create a closing pragma
        //createDirective(s,openDirectives.front());
        
        
        //printf("pop 1\n");
        openDirectives.pop_front();
      }
      
      while (directives.size() != 0 && s->stmtBegin().line() >=
      directives.front().getLine() && directives.front().getType() > 1)
          
      {
          //printf("pop 2\n");
          //open new directive        
          if (s->kind() != pdbStmt::ST_FDO && s->kind() != pdbStmt::ST_FOR)
          {  
            if (verbosity == Debug)    
              cerr << "Directives opening a loop are only allowed immediately before the loop body." << endl;
          }
          else
          {
            if (verbosity == Debug)  
              cerr << "Opening OMP loop directive, loop " << loop << endl;
            openDirectives.push_front(Directive(directives.front(), loop, s));
          }
          directives.pop_front();
      }
      }//end while
      
			int nextStmtLine = block->stmtEnd().line();

      int i = 0;
      if (verbosity == Debug) 
      {
        while (i < loop) 
        { cerr << "\t"; i++; }
        cerr << s->stmtBegin().line() << "(" << s->stmtBegin().col() << ")";
        if (s->downStmt() != NULL)
          cerr << " |" << (*s->downStmt()).stmtBegin().line();
        else 
          cerr << " |NA";
        if (s->nextStmt() != NULL)
          cerr <<" -" << (*s->nextStmt()).stmtBegin().line();
        else
          cerr << " -NA";
        cerr << " L" << block->stmtBegin().line() << "  N" << nextStmtLine;
				cerr << "  D" << directives.size();
        cerr << "  O" << openDirectives.size() << "  L" << loop << endl;
      }  
      //Move to the next statement
      //Must maintain monotomy, ie don't follow loops.
      if ((s->downStmt() != NULL || s->kind() == pdbItem::RO_EXT) && s->downStmt()->stmtBegin().line() >=
      s->stmtBegin().line())
      {  
        if (verbosity == Debug)
          cerr << "recurising (a) " << endl;
        addDirectives.splice(addDirectives.end(), findOMPStmt(s->downStmt(), s, loop + 1, pdb));
        if (s->nextStmt() == NULL)
        {
          if (verbosity == Debug)
            cerr << "ending block: " << s->stmtBegin().line() << endl;
          pdbFile f(-1);
          pdbStmt endBlock(-1);
          endBlock.stmtBegin(pdbLoc(&f,s->stmtEnd().line(),s->stmtEnd().col()));
          endBlock.stmtEnd(pdbLoc(&f,s->stmtEnd().line(),s->stmtEnd().col()));
          endBlock.nextStmt(s->nextStmt());
          endBlock.downStmt(NULL);
          endBlock.extraStmt(NULL);
        if (verbosity == Debug)
          cerr << "recurising (b) " << endl;
          addDirectives.splice(addDirectives.end(), findOMPStmt(&endBlock, block, loop, pdb));  
        }
      }
      if (s->extraStmt() != NULL && s->extraStmt()->stmtBegin().line() >=
            s->stmtBegin().line()) 
			{
        if (verbosity == Debug)
          cerr << "recurising (c) " << endl;
        addDirectives.splice(addDirectives.end(), findOMPStmt(s->extraStmt(), s, loop + 1, pdb));
			}

      if (s->nextStmt() != NULL && s->nextStmt()->stmtBegin().line() >=
                  s->stmtBegin().line())
			{
        if (verbosity == Debug)
          cerr << "recurising (d) " << endl;
        addDirectives.splice(addDirectives.end(), findOMPStmt(s->nextStmt(), block, loop, pdb));  
      }
      //Need to make sure that the end of this block is analyzed.
      if ((s->kind() == pdbStmt::ST_FOR || s->kind() == pdbStmt::ST_FDO)
      && (s->nextStmt() == NULL || s->nextStmt()->stmtBegin().line() == 0))
      {
        if (verbosity == Debug)
          cerr << "ending block" << endl;
          
        pdbFile file(-1);
        pdbStmt closeBlock(-1);
        closeBlock.stmtBegin(pdbLoc(&file,s->stmtEnd().line(),s->stmtEnd().col()));
        closeBlock.stmtEnd(pdbLoc(&file,s->stmtEnd().line(),s->stmtEnd().col()));
        closeBlock.nextStmt(NULL);
        closeBlock.downStmt(NULL);
        closeBlock.extraStmt(NULL);
        if (verbosity == Debug)
          cerr << "recurising (e) "<< endl;
        addDirectives.splice(addDirectives.end(), findOMPStmt(&closeBlock, block, loop, pdb));  
      }
      
  
      return addDirectives;*/
  }      
  bool checkForDirective(const int state, const pdbStmt *s, const pdbStmt *block, int loop, PDB& pdb)
	{
    int currentLine = s->stmtBegin().line();
		Directive nextDirective = directives.front();
		if (currentLine >= nextDirective.getLine())
		{
			//if the directive is valid
			if (//loop == nextDirective.getDepth() && 
			    nextDirective.getType() > 0 && (directives.size() > 0 ))
			{
				if (verbosity == Debug)
				{
					cerr << "open type: " <<  directives.front().getType() << endl;		
				}	
        if (verbosity == Debug)  
					cerr << "Opening OMP loop directive, loop " << loop << endl;

				openDirectives.push_front(directives.front());
        openDirectives.front().setDepth(loop);
        openDirectives.front().setBlock(s);
				directives.pop_front();
				//list<Directive> *newDirectives = findOMPStmt(STATE_OPEN,s,block,loop,pdb);
				addDirectives.splice(addDirectives.end(),findOMPStmt(STATE_OPEN,s,block,loop,pdb));
			}
			else
			{
				printf("ERROR: Mismatched OMP directives at line: %d.\n", s->stmtBegin().line());
				directives.pop_front();
				addDirectives.splice(addDirectives.end(),findOMPStmt(state, s, block, loop, pdb));
			}
					//printf("back in check returning true...\n");
			return true;
		}
					//printf("back in check returning false...\n");
		return false;
	}
	const pdbStmt* followingStmt(const pdbStmt *s)
	{
		if ((s->downStmt() != NULL || s->kind() == pdbItem::RO_EXT) && s->downStmt()->stmtBegin().line() >=
		s->stmtBegin().line())
		{
		  return s->downStmt();
		}
		else if (s->extraStmt() != NULL && s->extraStmt()->stmtBegin().line() >=
					s->stmtBegin().line()) 
		{
		  return s->extraStmt();
		}
		else if (s->nextStmt() != NULL && s->nextStmt()->stmtBegin().line() >=
								s->stmtBegin().line())
    {
      return s->nextStmt();
		}
		else if ((s->kind() == pdbStmt::ST_FOR || s->kind() == pdbStmt::ST_FDO)
		&& (s->nextStmt() == NULL || s->nextStmt()->stmtBegin().line() == 0))
		{
      //return s->stmtEnd();
			return NULL;
		}
		else
		{
		  //printf("Fatal Error: Cannot parse PDB file.");
      //exit(1);
			return NULL;
		}
	}
  list<Directive>& gotoNextStmt(const int state, const pdbStmt *s, const pdbStmt *block, int loop, PDB& pdb)
	{
		if (verbosity == Debug)
			printf("going to next stmt, at line: %d \n", s->stmtBegin().line());
	  //are we at the start of a loop
    if ((s->kind() == pdbStmt::ST_FOR || s->kind() == pdbStmt::ST_FDO ||
		s->kind() == pdbStmt::ST_FIF) && s->downStmt() != NULL)
    {
			//enter the loop
			if (verbosity == Debug)
				printf("recurising (down) \n");
			addDirectives.splice(addDirectives.end(), findOMPStmt(state, s->downStmt(), s, loop + 1, pdb));
			
			//process the else in an if stmt.
			if (s->extraStmt() != NULL)
			{
				if (verbosity == Debug)
					printf("recurising (extra) \n");
				addDirectives.splice(addDirectives.end(), findOMPStmt(state, s->extraStmt(), s, loop + 1, pdb));
			}
			
			//process the end of the loop by creating a stmt located there.
				if (verbosity == Debug)
					printf("ending block: %d\n", s->stmtEnd().line());
				pdbFile f(-1);
				pdbStmt endBlock(-1);
				endBlock.stmtBegin(pdbLoc(&f,s->stmtEnd().line(),s->stmtEnd().col()));
				endBlock.stmtEnd(pdbLoc(&f,s->stmtEnd().line(),s->stmtEnd().col()));
				endBlock.nextStmt(s->nextStmt());
				endBlock.downStmt(NULL);
				endBlock.extraStmt(NULL);
			if (verbosity == Debug)
				printf("recurising (end block) \n");
      //end of the loop
      addDirectives.splice(addDirectives.end(), findOMPStmt(state, &endBlock, block, loop, pdb));  

		  if (s->nextStmt() != NULL)
			{
				if (verbosity == Debug)
					printf("recurising (next after loop) \n");
				addDirectives.splice(addDirectives.end(), findOMPStmt(state, s->nextStmt(), block, loop, pdb));  
			}
		}
		else 
		{
		  if (s->nextStmt() != NULL)
			{
				if (verbosity == Debug)
					printf("recurising (next) \n");
				addDirectives.splice(addDirectives.end(), findOMPStmt(state, s->nextStmt(), block, loop, pdb));  
			}
			if (s->extraStmt() != NULL)
			{
				if (verbosity == Debug)
					printf("recurising (extra) \n");
				addDirectives.splice(addDirectives.end(), findOMPStmt(state, s->extraStmt(), s, loop + 1, pdb));
			}
			else
			{
				if (verbosity == Debug)
					printf("no more statements, returning \n");
        addDirectives;
			}
    }
		return addDirectives;
		}
	  /*
		if (verbosity == Debug)
			printf("going to next stmt, at line: %d \n", s->stmtBegin().line());
		//Move to the next statement
		//Must maintain monotomy, ie don't follow loops.
    //if (s->nextStmt() != NULL) { printf("[1] moving to next stmt.\n"); }
		if ((s->downStmt() != NULL || s->kind() == pdbItem::RO_EXT) && s->downStmt()->stmtBegin().line() >=
		s->stmtBegin().line())
		{  
			if (verbosity == Debug)
				printf("recurising (a) \n");
			addDirectives.splice(addDirectives.end(), findOMPStmt(state, s->downStmt(), s, loop + 1, pdb));
			if (s->nextStmt() == NULL || s->nextStmt()->stmtBegin().line() >= s->stmtBegin().line())
			{
				if (verbosity == Debug)
					printf("ending block: %d\n", s->stmtEnd().line());
				pdbFile f(-1);
				pdbStmt endBlock(-1);
				endBlock.stmtBegin(pdbLoc(&f,s->stmtEnd().line(),s->stmtEnd().col()));
				endBlock.stmtEnd(pdbLoc(&f,s->stmtEnd().line(),s->stmtEnd().col()));
				endBlock.nextStmt(s->nextStmt());
				endBlock.downStmt(NULL);
				endBlock.extraStmt(NULL);
			if (verbosity == Debug)
				printf("recurising (b) \n");
				addDirectives.splice(addDirectives.end(), findOMPStmt(state, &endBlock, block, loop, pdb));  
			}
		}
    //if (s->nextStmt() != NULL) { printf("[3] moving to next stmt.\n"); }
		else if (s->extraStmt() != NULL && s->extraStmt()->stmtBegin().line() >=
					s->stmtBegin().line()) 
		{
			if (verbosity == Debug)
				printf("recurising (c) \n");
			addDirectives.splice(addDirectives.end(), findOMPStmt(state, s->extraStmt(), s, loop + 1, pdb));
		}
    //if (s->nextStmt() != NULL) { printf("moving to next stmt.\n"); }
		else if (s->nextStmt() != NULL && s->nextStmt()->stmtBegin().line() >=
								s->stmtBegin().line())
		{
			if (verbosity == Debug)
				printf("recurising (d) \n");
			addDirectives.splice(addDirectives.end(), findOMPStmt(state, s->nextStmt(), block, loop, pdb));  
		}
		//Need to make sure that the end of this block is analyzed.
		else if ((s->kind() == pdbStmt::ST_FOR || s->kind() == pdbStmt::ST_FDO)
		&& (s->nextStmt() == NULL || s->nextStmt()->stmtBegin().line() == 0))
		{
			if (verbosity == Debug)
				printf("ending block\n");
				
			pdbFile file(-1);
			pdbStmt closeBlock(-1);
			closeBlock.stmtBegin(pdbLoc(&file,block->stmtEnd().line(),s->stmtEnd().col()));
			closeBlock.stmtEnd(pdbLoc(&file,block->stmtEnd().line(),s->stmtEnd().col()));
			closeBlock.nextStmt(NULL);
			closeBlock.downStmt(NULL);
			closeBlock.extraStmt(NULL);
			if (verbosity == Debug)
				printf("recurising (e) \n");
			addDirectives.splice(addDirectives.end(), findOMPStmt(state, &closeBlock,
			block, loop - 1, pdb));  
		}
		if (verbosity == Debug)
			printf("stopped recurising.\n");
		
		return addDirectives;*/
  void printLocation(const pdbStmt* s, const pdbStmt* block, int loop)
	{
      int i = 0;
			int nextStmtLine = block->stmtEnd().line();
      if (verbosity == Debug) 
      {
        while (i < loop) 
        { printf("\t"); i++; }
        printf( "%d(%d)",s->stmtBegin().line(), s->stmtBegin().col());
        if (s->downStmt() != NULL)
          printf(" |%d",(*s->downStmt()).stmtBegin().line());
        else 
          printf(" |NA");
        if (s->nextStmt() != NULL)
          printf(" -%d",(*s->nextStmt()).stmtBegin().line());
        else
          printf(" -NA");
        if (s->extraStmt() != NULL)
          printf(" *%d",(*s->extraStmt()).stmtBegin().line());
        else
          printf(" *NA");
        printf(" L%d  N%d", block->stmtBegin().line(), nextStmtLine);
				printf("  D%d",directives.size());
        printf("  O%d  L%d\n", openDirectives.size(), loop);
      }
	}

  void createDirective(const pdbStmt* s, Directive match)
  {
    string text;
    string type = "end";
    
    if (language == Fortran)
      text = "!$";
    else
      text = "#pragma ";

    text += "omp end ";
    int length;
    if (match.getType() == 1)
    {
      type += "parallel";
      text += "parallel";
      length = 18;
    }
    else if (match.getType() == 2)
    {
      type += "do";
      text += "do";
      length = 11;
    }
    else
    {
      type += "for";
      text += "for";
      length = 12;
    }  
    /**output << "pr#" << ++pragma_id << " omp\n"
            "ploc so#1 " << (s->stmtBegin().line()-1) << " " << "1\n"
            "pkind " << type << " \n"
            "ppos so#1 " << (s->stmtBegin().line()-1) << " " << "1  so#1 " <<
            s->stmtBegin().line()-1 << " " << length << "\n"
            "ptext " << text << "\n" << endl;
  */}
};

void placeDirective(Directive directive)
{
  if (verbosity == Debug)  
    cerr << "lang: " << lang << " type: " << directive.getType() << endl;
  if (lang == Fortran)
  {
    if (directive.getType() == -1)
      *output << endl << "!$omp end parallel" << endl;
    else if (directive.getType() == -2)
      *output << endl << "!$omp end parallel do" << endl;
    else if (directive.getType() == -3)
      *output << endl << "!$omp end do" << endl;
    else if (directive.getType() == -4)
      *output << endl << "!$omp end for" << endl;
  }
  else
  {
    if (directive.getType() == -1)
      *output << endl << "#pragma omp end parallel" << endl;
    else if (directive.getType() == -2)
      *output << endl << "#pragma omp end parallel do" << endl;
    else if (directive.getType() == -3)
      *output << endl << "#pragma omp end do" << endl;
    else if (directive.getType() == -4)
      *output << endl << "#pragma omp end for" << endl;
  }
}
void printHelp()
{
  cout << "Usage: tau_ompcheck pdbfile soucrefile [-v|-d|-o outfile]" << endl;
  cout << endl;
  cout << "Finds uncompleted do/for omp directives and inserts closing" << endl;
  cout << "directives for each one uncompleted. do/for directives are" << endl;
  cout << "expected immediately before a do/for loop. Closing directives are" << endl;
  cout << "then placed immediately following the same do/for loop." << endl;
  cout << endl;
  cout << "Arguments: " << endl;
  cout << "pdbfile:     A pdbfile generated from the source file you wish to check." << endl;
  cout << "             This pdbfile must contain comments form which the omp" << endl;
  cout << "             directives are gathered. See pdbcomment for information on" << endl;
  cout << "             how to obtain comment from a pdbfile." << endl;
  cout << "sourcefile:  A fortran, C or C++ source file to analyized." << endl;
  cout << endl;
  cout << "Options:" << endl; 
  cout << "-v           verbose output." << endl;
  cout << "-d           debuging information, we suggest you pipe this" << endl;
  cout << "             unrestrained output to a file." << endl;
  cout << "-o outfile   write the output to the specified outfile." << endl;
}

int main(int argc, char *argv[]) 
{
  if (argc < 3)
  {
    printHelp();
    exit(1);
  }
  
  ofstream of;
  file = argv[1];
  source = argv[2];
  output = &cout;
  verbosity = Standard;
  if (argc > 3)
  {
    int i = 3;
    while (i < argc && argv[i] != "")
    {
      if (string(argv[i]) == "-v")
      {  
        verbosity = Verbose;
      }
      else if (string(argv[i]) == "-d")
      {
        verbosity = Debug;
      }
      else if (string(argv[i]) == "-o" && argv[i+1] != NULL)
      {
        of.open(argv[i+1]);
        output = &of;
        i++; // grab two arguments
      }
      i++;
    }
  }
  
  PDB p(argv[1]); if ( !p ) return 1;
  //p.write(cout);    
  
  if (verbosity >= Verbose)
    cerr << "Language " << p.language() << endl;

  //cerr << argv[1] << "  " << argv[2] << endl;
    
  PDB::filevec& files = p.getFileVec();
  
  pragma_id = p.getPragmaVec().size();

  list<Directive> directivesToBeAdded;

/* NOT processing C/C++ code

  if (p.language() == PDB::LA_C || p.language() == PDB::LA_CXX || p.language() == PDB::LA_C_or_CXX)
  {  
    lang = C;
    
    //for(PDB::filevec::iterator fi=files.begin(); fi!=files.end(); ++fi) {
    //    if ( ! (*fi)->isSystemFile() ) {
      
          for (PDB::croutinevec::iterator r = p.getCRoutineVec().begin();
            r!=p.getCRoutineVec().end(); r++)
          {
            if ((*r)->body() != NULL && (*r)->kind() != pdbItem::RO_EXT)
            {
              if (verbosity >= Verbose)
                cerr << "Processing routine: " << (*r)->name() << endl;
              
              CompleteDirectives c = CompleteDirectives(file,*r,lang);
              
              //Retrive the statement within the routines.
              const pdbStmt *v = (*r)->body();
              directivesToBeAdded.splice(directivesToBeAdded.end(), c.findOMPStmt(v,p));
            }
          }
        //}
      //}
  }  
*/
  if (p.language() == PDB::LA_FORTRAN)
  {  
    lang = Fortran;
    //for(PDB::filevec::iterator fi=files.begin(); fi!=files.end(); ++fi) {
    //    if ( ! (*fi)->isSystemFile() ) {
          
  
          for (PDB::froutinevec::iterator r = p.getFRoutineVec().begin();
            r!=p.getFRoutineVec().end(); r++)
          {
            if (verbosity >= Verbose)
              cerr << "preprocesing rountine: " << (*r)->name() << endl;
            if ((*r)->body() != NULL && (*r)->kind() != pdbItem::RO_EXT)
            {
              if (verbosity >= Verbose)
                cerr << "Processing routine: " << (*r)->name() << endl;
              
              CompleteDirectives c = CompleteDirectives(file,*r,lang);
              
              //Retrive the statement within the routines.
              const pdbStmt *v = (*r)->body();
	            //printf("hello\n");
							directivesToBeAdded.splice(directivesToBeAdded.end(), c.findOMPStmt(v,p));
            }
          }
        //}
      //}
  }
  
  if (verbosity >= Verbose)
    cerr << "----------------------\nDirectives to be added:" << endl; 
  
  list<Directive> printDirectives = directivesToBeAdded;
  
  while (printDirectives.size() > 0 && verbosity >= Verbose)
  {
    cerr << "Type: " << printDirectives.front().getType() << " Location: ";
    cerr << printDirectives.front().getLine() << ",";
    cerr << printDirectives.front().getCol() << "  Bl: ";
    cerr << printDirectives.front().getBlock()->stmtEnd().line() << endl;
    printDirectives.pop_front();
  }
  //Write file with completed directives

  directivesToBeAdded.sort();
  
  ifstream input(source, ios::in);
  int currentLine = 1, currentCol;
  string buffer;
  char c;
  //input.getline(buffer, 1000);
  //cerr << directivesToBeAdded.size() << endl;
  while (directivesToBeAdded.size() != 0)
  {
    while (directivesToBeAdded.front().getLine() > currentLine)
    {
      //pass line.
      currentLine++;
      getline(input, buffer);
      *output << buffer << endl;
    }
    currentCol = 1;
    currentLine++;
    getline(input, buffer);
    string::iterator it = buffer.begin();
    while (directivesToBeAdded.front().getCol() > currentCol && it !=
    buffer.end())
    {
      //pass char
      c = *it;
      *output << c;
      currentCol++;
      it++;
    }
    //insert directive
    placeDirective(directivesToBeAdded.front());
    directivesToBeAdded.pop_front();
    //pass remaining char
    while (it != buffer.end())
    {
      c = *it;
      it++;
      *output << c;
    }
		*output << endl;
  }
  while (getline(input, buffer) != 0)
  {
    //pass remaining lines
    *output << buffer << endl;
  }
}
/***************************************************************************
 * $RCSfile: tau_ompcheck.cpp,v $   $Author: scottb $
 * $Revision: 1.28 $   $Date: 2008/10/24 23:51:55 $
 * VERSION_ID: $Id: tau_ompcheck.cpp,v 1.28 2008/10/24 23:51:55 scottb Exp $
 ***************************************************************************/
