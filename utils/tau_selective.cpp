/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.acl.lanl.gov/tau		           **
*****************************************************************************
**    Copyright 1999  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: tau_selective.cpp				  **
**	Description 	: Provides selective instrumentation support in   **
**                        TAU.                                            **
**	Author		: Sameer Shende					  **
**	Contact		: sameer@cs.uoregon.edu sameer@acl.lanl.gov 	  **
**	Documentation	: See http://www.acl.lanl.gov/tau	          **
***************************************************************************/

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <list>
#include <string>
using namespace std;

#define INBUF_SIZE 2048
#define BEGIN_EXCLUDE_TOKEN "BEGIN_EXCLUDE_LIST"
#define END_EXCLUDE_TOKEN   "END_EXCLUDE_LIST"
#define BEGIN_INCLUDE_TOKEN "BEGIN_INCLUDE_LIST"
#define END_INCLUDE_TOKEN   "END_INCLUDE_LIST"
list<string> excludelist;
list<string> includelist;

/* -------------------------------------------------------------------------- */
/* -- Read the instrumentation requests file. ------------------------------- */

/* -------------------------------------------------------------------------- */
int processInstrumentationRequests(char *fname)
{
  ifstream input(fname); 
  char inbuf[INBUF_SIZE];

  if (!input)
  {
    cout << "ERROR: Cannot open file" << fname<<endl;
    return 0; 
  }
  while (input.getline(inbuf, INBUF_SIZE) || input.gcount()) {  
    if ((inbuf[0] == '#') || (inbuf[0] == '\0')) continue;
    if (strcmp(inbuf,BEGIN_EXCLUDE_TOKEN) == 0) 
    {
      while(input.getline(inbuf,INBUF_SIZE) || input.gcount())
      {
	if (strcmp(inbuf, END_EXCLUDE_TOKEN) == 0)
	  break; /* Found the end of exclude list. */  
        if ((inbuf[0] == '#') || (inbuf[0] == '\0')) continue;
        excludelist.push_back(string(inbuf)); 
      }
    }
    if (strcmp(inbuf,BEGIN_INCLUDE_TOKEN) == 0) 
    {
      while(input.getline(inbuf,INBUF_SIZE) || input.gcount())
      {
	if (strcmp(inbuf, END_INCLUDE_TOKEN) == 0)
	  break; /* Found the end of exclude list. */  
        if ((inbuf[0] == '#') || (inbuf[0] == '\0')) continue;
        includelist.push_back(string(inbuf)); 
      }
    }
    /* next token */
  }
  input.close();
  return 0;
}

/* -------------------------------------------------------------------------- */
/* -- print the contents of the exclude list -------------------------------- */
/* -------------------------------------------------------------------------- */
void printExcludeList(void)
{
  for (list<string>::iterator it = excludelist.begin(); 
	it != excludelist.end(); it++)
    cout <<"Exclude Item: "<<*it<<endl;

}

/* -------------------------------------------------------------------------- */
/* -- print the contents of the include list -------------------------------- */
/* -------------------------------------------------------------------------- */

void printIncludeList(void)
{
  for (list<string>::iterator it = includelist.begin(); 
	it != includelist.end(); it++)
    cout <<"Include Item: "<<*it<<endl;

}

/* -------------------------------------------------------------------------- */
/* -- report a match or a mismatch between two strings ---------------------- */
/* -------------------------------------------------------------------------- */
bool matchName(const string& str1, const string& str2)
{
  int length1, length2, size_to_compare;
  const char *s1;
  length1 = str1.length();
  length2 = str2.length();

  size_to_compare = (length1 < length2) ? length1 : length2;
 
  if (strncmp(str1.c_str(), str2.c_str(), size_to_compare) == 0)
    return true;
  else 
    return false;
}

/* -------------------------------------------------------------------------- */
/* -- should the routine be instrumented? Returns true or false ------------- */
/* -------------------------------------------------------------------------- */
bool instrumentEntity(const string& function_name)
{

  if (!excludelist.empty())
  { /* if exclude list has entries */
    for (list<string>::iterator it = excludelist.begin();
        it != excludelist.end(); it++)
    { /* iterate through the entries and see if the names match */
      if (matchName(*it, function_name)) 
      {
        /* names match! This routine should not be instrumented */
#ifdef DEBUG
	cout <<"Don't instrument: "<<function_name<<" it is in exclude list"<<endl;
#endif /* DEBUG */
        return false;
      }
    }
  }  

  /* There is no entry in the exclude list that matches function_name. */ 
  if (includelist.empty())
  {
#ifdef DEBUG
    cout <<"Instrumenting ... "<< function_name
	 << " INCLUDE list is empty" <<endl;
#endif /* DEBUG */
    /* There are no entries in the include list */
    return true; /* instrument the given routine */
  }
  else
  {
    /* wait! The include list is not empty. We must see if the given name
       appears in the list of routines that should be instrumented 
       For this, we need to iterate through the include list.  */
    for (list<string>::iterator it = includelist.begin();
        it != includelist.end(); it++)
    { /* iterate through the entries and see if the names match */
      if (matchName(*it, function_name)) 
      {
        /* names match! This routine should be instrumented */
#ifdef DEBUG
	cout <<"Instrument "<< function_name<<" matchName is true "<<endl;
#endif /* DEBUG */
        return true;
      }
      else 
      {
#ifdef DEBUG
	cout <<"Iterate over include_list: "
	     << function_name<<" matchName is false "<<endl;
#endif /* DEBUG */
      }
    }
    /* the name doesn't match any routine name in the include list. 
       Do not instrument it. */
    return false;
  }
 
}


/***************************************************************************
 * $RCSfile: tau_selective.cpp,v $   $Author: sameer $
 * $Revision: 1.1 $   $Date: 2002/03/11 22:47:23 $
 * VERSION_ID: $Id: tau_selective.cpp,v 1.1 2002/03/11 22:47:23 sameer Exp $
 ***************************************************************************/
