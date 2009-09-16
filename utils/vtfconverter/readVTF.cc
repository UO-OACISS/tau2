 /****************************************************************************
 ****************************************************************************/

/****************************************************************************
 *	readVTF.cc
 *
 *	Author: Ryan K. Harris & Wyatt Spear
 *
 *	readVTF is the module which utilizes the "vtf3.h" of Vampir	to read
 *	in VTF files, parse them, and pull out the desired trace info. Then
 *	the binarytree data structure module is called into which we store the
 *	information. Each distinct(unique) cpu is stored as a node,
 *	sorted by it's unique CPUID #. Within each CPU node is a private member
 *	tree which stores all functions run on that cpu as nodes on a
 *	'function' tree.
 *
 *	Large portions of this code is taken from Stephan Seidl
 ****************************************************************************/

#include "readVTF.h"
#include <stdlib.h>

int debug = 0;

/* Define user handler code for the user handlers declared in header */

/****************************************************************************
	MyDowntoHandler -- this is the newer equivalent to an exchangetype == 1.
****************************************************************************/
int readVTF::MyDowntoHandler(void *ccbvoid,
			     double time,
			     int statetoken,
			     unsigned int cpuid,
			     int scltoken)
{

  readVTF* this_readVTF_obj = (readVTF*)ccbvoid;

  if(this_readVTF_obj->lastcall != -1)
    {
      cout << "DANGER! 1\n";
    }

  this_readVTF_obj->sampstatetoken = statetoken;
  this_readVTF_obj->sampcpupid = cpuid;
  this_readVTF_obj->sampscltoken = scltoken;
  this_readVTF_obj->samptime = time;
  this_readVTF_obj->sampindex = 0;
  this_readVTF_obj->lastcall = 0;
  if(this_readVTF_obj->globmets == 0)
    {
      this_readVTF_obj->DowntoBatch(this_readVTF_obj);
      this_readVTF_obj->sampindex = -1;
      this_readVTF_obj->lastcall = -1;

    }
  return(1);
}

/****************************************************************************
	DowntoBatch -- Records collected time and sample data as per 
	exchangetype == 1.
****************************************************************************/
int readVTF::DowntoBatch(void *ccbvoid)
{
  /* The following line must be executed in order to gain
     access to the member var's, i.e.
     'this_readVTF_obj.some_counter_var' where
     some_counter_var would be in the private section of the
     class definition. */

  readVTF* this_readVTF_obj = (readVTF*)ccbvoid;

  double time = this_readVTF_obj->samptime;
  int statetoken = this_readVTF_obj->sampstatetoken;
  unsigned int cpuid = this_readVTF_obj->sampcpupid;
  //int scltoken = this_readVTF_obj->sampscltoken;
  int nummets = this_readVTF_obj->globmets;

  /* The following 'exchangetype' var is declared to
     make this handler compatable (able to use) the
     current implementation of the stack class. */
  int exchangetype = 1;

  double timeStamp = time;
  double startingTime = this_readVTF_obj->startTime;
  double endingTime = this_readVTF_obj->endTime;
  if(this_readVTF_obj->clockPeriod != 0)
    {
      timeStamp = timeStamp * this_readVTF_obj->clockPeriod;
    }
  /* Convert to microSeconds == 10E-6. */
  timeStamp = timeStamp * 1000000;


  /*cout << "DT. TS: " << timeStamp << " ";
    for(int i = 0; i < nummets; i++)
    {
    cout << " M" << i <<  ": " << this_readVTF_obj->sampholder[i];
    }
    cout << endl;*/

  /* If threaded, handle different. */
  if(this_readVTF_obj->threadNumArrayDim)
    {
      int cpuID;
      int threadID;

      cpuID = (cpuid << 16) >> 16;
      threadID = cpuid >> 16;
      //cout << cpuid << " " << cpuID << " " << threadID << endl;

      /* Check that the desired function, and group
	 exist in the appropriate lookUpTrees; otherwise, insert
	 in an attempt at traceRecord correction. */

      node * a_FuncNode =
	this_readVTF_obj->functionLookUp->getFuncNode(statetoken);
      if(a_FuncNode == 0)
	{
	  string the_statename;
	  the_statename = "UNDEFINED_FUNCTION";

	  this_readVTF_obj->functionLookUp->insert(	this_readVTF_obj->nextToken,
							statetoken,
							&the_statename,
							&the_statename);

	  this_readVTF_obj->nextToken = this_readVTF_obj->nextToken + 5;

	  a_FuncNode =
	    this_readVTF_obj->functionLookUp->getFuncNode(statetoken);
	}//if

      int group = (*a_FuncNode).getGroupID();
      node * a_GroupNode =
	this_readVTF_obj->activityLookUp->getFuncNode(group);
      if(a_GroupNode == 0)
	{
	  string the_activityname;
	  the_activityname = "UNDEFINED_GROUP";

	  this_readVTF_obj->activityLookUp->insert(	group,
							group,
							&the_activityname,
							&the_activityname);
	}//if

      a_FuncNode =
	this_readVTF_obj->functionLookUp->getFuncNode(statetoken);

      group = (*a_FuncNode).getGroupID();

      string the_statename;
      the_statename = *(a_FuncNode->getFunctionName());

      /* Get the groupName (activityname) from the
	 activityLookUp tree. */

      a_GroupNode =
	this_readVTF_obj->activityLookUp->getFuncNode(group);
      string the_activityname;

      /* The following function makes use of the 'getFunctionName()'
	 method, but keep in mind that we have stored the
	 activity (group)name there in this tree. */

      the_activityname = *(a_GroupNode->getFunctionName());

      /* Now attempt to insert the new 'cpu' node and
	 'function' nodes into the cpu_tree and
	 associated private funcTree of that 'cpu' node. */

      this_readVTF_obj->cpu_tree->insert(	cpuID,
						group,
						statetoken,
						&the_statename,
						&the_activityname,
						threadID,
						startingTime,
						endingTime);


      /* Now knowing that it exists. */
      nodeC* a_CPUnode =
	this_readVTF_obj->cpu_tree->getCPUNode(cpuID);

      nodeC * a_thread =
	a_CPUnode->threadTree->getCPUNode(threadID);

      /* Now increment the 'function' node within this
	 'thread' nodes subTree. */

      /* Now happens inside 'stack' structures.

      a_thread->incFuncCount(statetoken);
      */

      a_thread->push(statetoken, timeStamp, this_readVTF_obj->sampholder, nummets, exchangetype);

      return(1);
    }//if threaded
  else
    {

      /* Check that the desired function, and group
	 exist in the appropriate lookUpTrees; otherwise, insert
	 in an attempt at traceRecord correction. */

      node * a_FuncNode =
	this_readVTF_obj->functionLookUp->getFuncNode(statetoken);
      if(a_FuncNode == 0)
	{
	  string the_statename;
	  the_statename = "UNDEFINED_FUNCTION";

	  this_readVTF_obj->functionLookUp->insert(	this_readVTF_obj->nextToken,
							statetoken,
							&the_statename,
							&the_statename);

	  this_readVTF_obj->nextToken = this_readVTF_obj->nextToken + 5;

	  a_FuncNode =
	    this_readVTF_obj->functionLookUp->getFuncNode(statetoken);
	}//if

      int group = (*a_FuncNode).getGroupID();
      node * a_GroupNode =
	this_readVTF_obj->activityLookUp->getFuncNode(group);
      if(a_GroupNode == 0)
	{
	  string the_activityname;
	  the_activityname = "UNDEFINED_GROUP";

	  this_readVTF_obj->activityLookUp->insert(	group,
							group,
							&the_activityname,
							&the_activityname);
	}//if

      a_FuncNode =
	this_readVTF_obj->functionLookUp->getFuncNode(statetoken);

      group = (*a_FuncNode).getGroupID();

      string the_statename;
      the_statename = *(a_FuncNode->getFunctionName());

      /* Get the groupName (activityname) from the
	 activityLookUp tree. */

      a_GroupNode =
	this_readVTF_obj->activityLookUp->getFuncNode(group);
      string the_activityname;

      /* The following function makes use of the 'getFunctionName()'
	 method, but keep in mind that we have stored the
	 activity (group)name there in this tree. */

      the_activityname = *(a_GroupNode->getFunctionName());

      /* Now attempt to insert the new 'cpu' node and
	 'function' nodes into the cpu_tree and
	 associated private funcTree of that 'cpu' node. */

      this_readVTF_obj->cpu_tree->insert(	cpuid,
						group,
						statetoken,
						&the_statename,
						&the_activityname,
						startingTime,
						endingTime);

      nodeC* a_CPUnode =
	this_readVTF_obj->cpu_tree->getCPUNode(cpuid);

      (*a_CPUnode).push(statetoken, timeStamp, this_readVTF_obj->sampholder, nummets, exchangetype);


      return(1);
    }//else (not threaded)

  return(1);
}//MyDowntoHandler


/****************************************************************************
	MyUpfromHandler -- this is the newer equivalent to an exchangetype == 2.
		The difference between this function and Upto is that Upfrom expects
		the token (functionID) of the function being left, so I pass a
		special exchangetype == 5 to tell 'stack' how to handle.
****************************************************************************/
int readVTF::MyUpfromHandler(void *ccbvoid,
			     double time,
			     int statetoken,
			     unsigned int cpuid,
			     int scltoken)
{
  readVTF* this_readVTF_obj = (readVTF*)ccbvoid;

  if(this_readVTF_obj->lastcall != -1)
    {
      cout << "DANGER! 1\n";
    }

  this_readVTF_obj->sampstatetoken = statetoken;
  this_readVTF_obj->sampcpupid = cpuid;
  this_readVTF_obj->sampscltoken = scltoken;
  this_readVTF_obj->samptime = time;
  this_readVTF_obj->sampindex = 0;
  this_readVTF_obj->lastcall = 1;
  if(this_readVTF_obj->globmets == 0)
    {
      this_readVTF_obj->UpfromBatch(this_readVTF_obj);
      this_readVTF_obj->sampindex = -1;
      this_readVTF_obj->lastcall = -1;
    }
  return(1);
}

/****************************************************************************
	UpFromBatch -- Records collected time and sample data as per 
	upFrom.
****************************************************************************/
int readVTF::UpfromBatch(void *ccbvoid)
{
  /* The following line must be executed in order to gain
     access to the member var's, i.e.
     'this_readVTF_obj.some_counter_var' where
     some_counter_var would be in the private section of the
     class definition. */

  readVTF* this_readVTF_obj = (readVTF*)ccbvoid;

  double time = this_readVTF_obj->samptime;
  int statetoken = this_readVTF_obj->sampstatetoken;
  unsigned int cpuid = this_readVTF_obj->sampcpupid;
  //int scltoken = this_readVTF_obj->sampscltoken;
  int nummets = this_readVTF_obj->globmets;


  /* The following 'exchangetype' var is declared to
     make this handler compatable (able to use) the
     current implementation of the stack class.*/
  int exchangetype = 5;

  double timeStamp = time;
  double startingTime = this_readVTF_obj->startTime;
  double endingTime = this_readVTF_obj->endTime;
  if(this_readVTF_obj->clockPeriod != 0)
    {
      timeStamp = timeStamp * this_readVTF_obj->clockPeriod;
    }
  /* Convert to microSeconds == 10E-6. */
  timeStamp = timeStamp * 1000000;

  /*cout << "UF. TS: " << timeStamp << " ";
    for(int i = 0; i < nummets; i++)
    {
    cout << " M" << i <<  ": " << this_readVTF_obj->sampholder[i];
    }
    cout << endl;*/


  /* If threaded, handle different. */
  if(this_readVTF_obj->threadNumArrayDim)
    {
      int cpuID;
      int threadID;

      cpuID = (cpuid << 16) >> 16;
      threadID = cpuid >> 16;
      //cout << cpuid << " " << cpuID << " " << threadID << endl;
      /* Check that the desired function, and group
	 exist in the appropriate lookUpTrees; otherwise, insert
	 in an attempt at traceRecord correction. */

      node * a_FuncNode =
	this_readVTF_obj->functionLookUp->getFuncNode(statetoken);
      if(a_FuncNode == 0)
	{
	  string the_statename;
	  the_statename = "UNDEFINED_FUNCTION";

	  this_readVTF_obj->functionLookUp->insert(	this_readVTF_obj->nextToken,
							statetoken,
							&the_statename,
							&the_statename);

	  this_readVTF_obj->nextToken = this_readVTF_obj->nextToken + 5;

	  a_FuncNode =
	    this_readVTF_obj->functionLookUp->getFuncNode(statetoken);
	}//if

      int group = (*a_FuncNode).getGroupID();
      node * a_GroupNode =
	this_readVTF_obj->activityLookUp->getFuncNode(group);
      if(a_GroupNode == 0)
	{
	  string the_activityname;
	  the_activityname = "UNDEFINED_GROUP";

	  this_readVTF_obj->activityLookUp->insert(	group,
							group,
							&the_activityname,
							&the_activityname);
	}//if


      /* Get the thread(threadID) from the cpu(cpuID). */

      /* It is remotely  possible that we might be dealing with
	 a horribly constructed traceFile, and that neither
	 the CPU nor the threadNode were ever created, so
	 we include if-statements to check here. */

      nodeC * a_CPUnode =
	this_readVTF_obj->cpu_tree->getCPUNode(cpuID);
      if(a_CPUnode == 0)
	{
	  /* Doesn't exist, insert it. */
	  int group = (*a_FuncNode).getGroupID();
	  string the_statename;
	  the_statename = *(a_FuncNode->getFunctionName());
	  string the_activityname;
	  the_activityname = *(a_GroupNode->getFunctionName());
	  this_readVTF_obj->cpu_tree->insert(	cpuID,
						group,
						statetoken,
						&the_statename,
						&the_activityname,
						threadID,
						startingTime,
						endingTime);
	  a_CPUnode =
	    this_readVTF_obj->cpu_tree->getCPUNode(cpuID);
	}//if

      nodeC * a_thread =
	a_CPUnode->threadTree->getCPUNode(threadID);

      a_thread->push(statetoken, timeStamp, this_readVTF_obj->sampholder, nummets, exchangetype);

      return(1);
    }//if threaded
  else
    {

      /* Check that the desired function, and group
	 exist in the appropriate lookUpTrees; otherwise, insert
	 in an attempt at traceRecord correction. */

      node * a_FuncNode =
	this_readVTF_obj->functionLookUp->getFuncNode(statetoken);
      if(a_FuncNode == 0)
	{
	  string the_statename;
	  the_statename = "UNDEFINED_FUNCTION";

	  this_readVTF_obj->functionLookUp->insert(	this_readVTF_obj->nextToken,
							statetoken,
							&the_statename,
							&the_statename);

	  this_readVTF_obj->nextToken = this_readVTF_obj->nextToken + 5;

	  a_FuncNode =
	    this_readVTF_obj->functionLookUp->getFuncNode(statetoken);
	}//if

      int group = (*a_FuncNode).getGroupID();
      node * a_GroupNode =
	this_readVTF_obj->activityLookUp->getFuncNode(group);
      if(a_GroupNode == 0)
	{
	  string the_activityname;
	  the_activityname = "UNDEFINED_GROUP";

	  this_readVTF_obj->activityLookUp->insert(	group,
							group,
							&the_activityname,
							&the_activityname);
	}//if

      /* Now push node onto this cpu's internal stack.
	 The stack is designed to handle everything
	 within it's structure and methods, it has
	 access to the other data structures internal
	 to this 'cpu' node. */

      nodeC * a_CPUnode =
	this_readVTF_obj->cpu_tree->getCPUNode(cpuid);
      if(a_CPUnode == 0)
	{
	  /* Doesn't exist, insert it. */
	  int group = (*a_FuncNode).getGroupID();
	  string the_statename;
	  the_statename = *(a_FuncNode->getFunctionName());
	  string the_activityname;
	  the_activityname = *(a_GroupNode->getFunctionName());
	  this_readVTF_obj->cpu_tree->insert(	cpuid,
						group,
						statetoken,
						&the_statename,
						&the_activityname,
						startingTime,
						endingTime);
	}//if

      (*a_CPUnode).push(statetoken, timeStamp, this_readVTF_obj->sampholder, nummets, exchangetype);

      return(1);
    }//else (not threaded)

  return(1);
}//MyUpfromHandler


/****************************************************************************
	MyUptoHandler -- this is the newer equivalent to an exchangetype == 2.
		The difference between this function and Upfrom is that Upto expects
		the token(functionID) of the target function.
****************************************************************************/
int readVTF::MyUptoHandler(void *ccbvoid,
			   double time,
			   int statetoken,
			   unsigned int cpuid,
			   int scltoken)
{

  readVTF* this_readVTF_obj = (readVTF*)ccbvoid;

  if(this_readVTF_obj->lastcall != -1)
    {
      cout << "DANGER! 1\n";
    }

  this_readVTF_obj->sampstatetoken = statetoken;
  this_readVTF_obj->sampcpupid = cpuid;
  this_readVTF_obj->sampscltoken = scltoken;
  this_readVTF_obj->samptime = time;
  this_readVTF_obj->sampindex = 0;
  this_readVTF_obj->lastcall = 2;
  if(this_readVTF_obj->globmets == 0)
    {
      this_readVTF_obj->UptoBatch(this_readVTF_obj);
      this_readVTF_obj->sampindex = -1;
      this_readVTF_obj->lastcall = -1;
    }
  return(1);
}


/****************************************************************************
	UptoBatch -- Records collected time and sample data as per 
	Upto
****************************************************************************/
int readVTF::UptoBatch(void *ccbvoid)
{
  /* The following line must be executed in order to gain
     access to the member var's, i.e.
     'this_readVTF_obj.some_counter_var' where
     some_counter_var would be in the private section of the
     class definition. */

  readVTF* this_readVTF_obj = (readVTF*)ccbvoid;

  double time = this_readVTF_obj->samptime;
  int statetoken = this_readVTF_obj->sampstatetoken;
  unsigned int cpuid = this_readVTF_obj->sampcpupid;
  //int scltoken = this_readVTF_obj->sampscltoken;
  int nummets = this_readVTF_obj->globmets;

  /* The following 'exchangetype' var is declared to
     make this handler compatable (able to use) the
     current implementation of the stack class. */
  int exchangetype = 2;

  double timeStamp = time;
  double startingTime = this_readVTF_obj->startTime;
  double endingTime = this_readVTF_obj->endTime;
  if(this_readVTF_obj->clockPeriod != 0)
    {
      timeStamp = timeStamp * this_readVTF_obj->clockPeriod;
    }
  /* Convert to microSeconds == 10E-6. */
  timeStamp = timeStamp * 1000000;

  /*cout << "UT. TS: " << timeStamp << " ";
    for(int i = 0; i < nummets; i++)
    {
    cout << " M" << i <<  ": " << this_readVTF_obj->sampholder[i];
    }
    cout << endl;*/


  /* If threaded, handle different. */
  if(this_readVTF_obj->threadNumArrayDim)
    {
      int cpuID;
      int threadID;

      cpuID = (cpuid << 16) >> 16;
      threadID = cpuid >> 16;
      //cout << cpuid << " " << cpuID << " " << threadID << endl;
      /* Check that the desired function, and group
	 exist in the appropriate lookUpTrees; otherwise, insert
	 in an attempt at traceRecord correction. */

      node * a_FuncNode =
	this_readVTF_obj->functionLookUp->getFuncNode(statetoken);
      if(a_FuncNode == 0)
	{
	  string the_statename;
	  the_statename = "UNDEFINED_FUNCTION";

	  this_readVTF_obj->functionLookUp->insert(	this_readVTF_obj->nextToken,
							statetoken,
							&the_statename,
							&the_statename);

	  this_readVTF_obj->nextToken = this_readVTF_obj->nextToken + 5;

	  a_FuncNode =
	    this_readVTF_obj->functionLookUp->getFuncNode(statetoken);
	}//if

      int group = (*a_FuncNode).getGroupID();
      node * a_GroupNode =
	this_readVTF_obj->activityLookUp->getFuncNode(group);
      if(a_GroupNode == 0)
	{
	  string the_activityname;
	  the_activityname = "UNDEFINED_GROUP";

	  this_readVTF_obj->activityLookUp->insert(	group,
							group,
							&the_activityname,
							&the_activityname);
	}//if


      /* Get the thread(threadID) from the cpu(cpuID). */

      /* It is remotely  possible that we might be dealing with
	 a horribly constructed traceFile, and that neither
	 the CPU nor the threadNode were ever created, so
	 we include if-statements to check here. */

      nodeC * a_CPUnode =
	this_readVTF_obj->cpu_tree->getCPUNode(cpuID);
      if(a_CPUnode == 0)
	{
	  /* Doesn't exist, insert it. */
	  int group = (*a_FuncNode).getGroupID();
	  string the_statename;
	  the_statename = *(a_FuncNode->getFunctionName());
	  string the_activityname;
	  the_activityname = *(a_GroupNode->getFunctionName());
	  this_readVTF_obj->cpu_tree->insert(	cpuID,
						group,
						statetoken,
						&the_statename,
						&the_activityname,
						threadID,
						startingTime,
						endingTime);
	  a_CPUnode =
	    this_readVTF_obj->cpu_tree->getCPUNode(cpuID);
	}//if

      nodeC * a_thread =
	a_CPUnode->threadTree->getCPUNode(threadID);

      a_thread->push(statetoken, timeStamp, this_readVTF_obj->sampholder, nummets, exchangetype);

      return(1);
    }//if threaded
  else
    {

      /* Check that the desired function, and group
	 exist in the appropriate lookUpTrees; otherwise, insert
	 in an attempt at traceRecord correction. */

      node * a_FuncNode =
	this_readVTF_obj->functionLookUp->getFuncNode(statetoken);
      if(a_FuncNode == 0)
	{
	  string the_statename;
	  the_statename = "UNDEFINED_FUNCTION";

	  this_readVTF_obj->functionLookUp->insert(	this_readVTF_obj->nextToken,
							statetoken,
							&the_statename,
							&the_statename);

	  this_readVTF_obj->nextToken = this_readVTF_obj->nextToken + 5;

	  a_FuncNode =
	    this_readVTF_obj->functionLookUp->getFuncNode(statetoken);
	}//if

      int group = (*a_FuncNode).getGroupID();
      node * a_GroupNode =
	this_readVTF_obj->activityLookUp->getFuncNode(group);
      if(a_GroupNode == 0)
	{
	  string the_activityname;
	  the_activityname = "UNDEFINED_GROUP";

	  this_readVTF_obj->activityLookUp->insert(	group,
							group,
							&the_activityname,
							&the_activityname);
	}//if

      /* Now push node onto this cpu's internal stack.
	 The stack is designed to handle everything
	 within it's structure and methods, it has
	 access to the other data structures internal
	 to this 'cpu' node. */

      nodeC * a_CPUnode =
	this_readVTF_obj->cpu_tree->getCPUNode(cpuid);
      if(a_CPUnode == 0)
	{
	  /* Doesn't exist, insert it. */
	  int group = (*a_FuncNode).getGroupID();
	  string the_statename;
	  the_statename = *(a_FuncNode->getFunctionName());
	  string the_activityname;
	  the_activityname = *(a_GroupNode->getFunctionName());
	  this_readVTF_obj->cpu_tree->insert(	cpuid,
						group,
						statetoken,
						&the_statename,
						&the_activityname,
						startingTime,
						endingTime);
	}//if

      (*a_CPUnode).push(statetoken, timeStamp, this_readVTF_obj->sampholder, nummets, exchangetype);

      return(1);
    }//else (not threaded)

  return(1);
}//MyUptoHandler


/****************************************************************************
	MyExchangeHandler - Collective handler for all exchange types.  Not yet
	compatable with user defined metric data
****************************************************************************/
int readVTF::MyExchangeHandler(void *ccbvoid,
			       double time,
			       unsigned int cpuid,
			       int exchangetype,
			       int statetoken,
			       int job,
			       int scltoken)
{

  readVTF* this_readVTF_obj = (readVTF*)ccbvoid;
  int nummets = this_readVTF_obj->globmets;
  double timeStamp = time;
  double startingTime = this_readVTF_obj->startTime;
  double endingTime = this_readVTF_obj->endTime;
  if(this_readVTF_obj->clockPeriod != 0)
    {
      timeStamp = timeStamp * this_readVTF_obj->clockPeriod;
    }
  /* Convert to microSeconds == 10E-6. */
  timeStamp = timeStamp * 1000000;

  /*
    cout << "*****EXCHANGE*****\n";
    cout << "time: " << time << "\n";
    cout << "cpuid: " << cpuid << "\n";
    cout << "exchangetype: " << exchangetype << "\n";
    cout << "statetoken: " << statetoken << "\n";
    cout << "job: " << job << "\n";
    cout << "scltoken: " << scltoken << "\n";
  */

  /* I want to keep track of how many times each
     function is called and increment their individual
     counters respectively. */


  //FOR DEBUGGING
  /*
    if(cpuid == 38)
    {
    cerr << "cpuID: " << cpuid << "\n";
    cerr << "functionID: " << statetoken << "\n";
    cerr << "exchange event: type: " << exchangetype << "\n";
    cerr << "timeStamp: " << time << "\n";
    }
  */

  /* If threaded, handle different. */
  if(this_readVTF_obj->threadNumArrayDim)
    {
      int cpuID;
      int threadID;

      cpuID = (cpuid << 16) >> 16;
      threadID = cpuid >> 16;

      /* Check that the desired function, and group
	 exist in the appropriate lookUpTrees; otherwise, insert
	 in an attempt at traceRecord correction. */

      node * a_FuncNode =
	this_readVTF_obj->functionLookUp->getFuncNode(statetoken);
      if(a_FuncNode == 0)
	{
	  string the_statename;
	  the_statename = "UNDEFINED_FUNCTION";

	  this_readVTF_obj->functionLookUp->insert(	this_readVTF_obj->nextToken,
							statetoken,
							&the_statename,
							&the_statename);

	  this_readVTF_obj->nextToken = this_readVTF_obj->nextToken + 5;

	  a_FuncNode =
	    this_readVTF_obj->functionLookUp->getFuncNode(statetoken);
	}//if

      int group = (*a_FuncNode).getGroupID();
      node * a_GroupNode =
	this_readVTF_obj->activityLookUp->getFuncNode(group);
      if(a_GroupNode == 0)
	{
	  string the_activityname;
	  the_activityname = "UNDEFINED_GROUP";

	  this_readVTF_obj->activityLookUp->insert(	group,
							group,
							&the_activityname,
							&the_activityname);
	}//if


      if(exchangetype == 2)
	{
	  /* Get the thread(threadID) from the cpu(cpuID). */

	  /* It is remotely  possible that we might be dealing with
	     a horribly constructed traceFile, and that neither
	     the CPU nor the threadNode were ever created, so
	     we include if-statements to check here. */

	  nodeC * a_CPUnode =
	    this_readVTF_obj->cpu_tree->getCPUNode(cpuID);
	  if(a_CPUnode == 0)
	    {
	      /* Doesn't exist, insert it. */
	      int group = (*a_FuncNode).getGroupID();
	      string the_statename;
	      the_statename = *(a_FuncNode->getFunctionName());
	      string the_activityname;
	      the_activityname = *(a_GroupNode->getFunctionName());
	      this_readVTF_obj->cpu_tree->insert(	cpuID,
							group,
							statetoken,
							&the_statename,
							&the_activityname,
							threadID,
							startingTime,
							endingTime);
	      a_CPUnode =
		this_readVTF_obj->cpu_tree->getCPUNode(cpuID);
	    }//if

	  nodeC * a_thread =
	    a_CPUnode->threadTree->getCPUNode(threadID);

	  a_thread->push(statetoken, timeStamp, this_readVTF_obj->sampholder, nummets, exchangetype);

	}//exchangetype == 2

      if(exchangetype == 1)
	{
	  a_FuncNode =
	    this_readVTF_obj->functionLookUp->getFuncNode(statetoken);

	  int group = (*a_FuncNode).getGroupID();

	  string the_statename;
	  the_statename = *(a_FuncNode->getFunctionName());

	  /* Get the groupName (activityname) from the
	     activityLookUp tree. */

	  a_GroupNode =
	    this_readVTF_obj->activityLookUp->getFuncNode(group);
	  string the_activityname;

	  /* The following function makes use of the 'getFunctionName()'
	     method, but keep in mind that we have stored the
	     activity (group)name there in this tree. */

	  the_activityname = *(a_GroupNode->getFunctionName());

	  /* Now attempt to insert the new 'cpu' node and
	     'function' nodes into the cpu_tree and
	     associated private funcTree of that 'cpu' node. */

	  this_readVTF_obj->cpu_tree->insert(	cpuID,
						group,
						statetoken,
						&the_statename,
						&the_activityname,
						threadID,
						startingTime,
						endingTime);


	  /* Now knowing that it exists. */
	  nodeC* a_CPUnode =
	    this_readVTF_obj->cpu_tree->getCPUNode(cpuID);

	  nodeC * a_thread =
	    a_CPUnode->threadTree->getCPUNode(threadID);

	  /* Now increment the 'function' node within this
	     'thread' nodes subTree. */

	  /* Now happens inside 'stack' structures.

	  a_thread->incFuncCount(statetoken);
	  */

	  a_thread->push(statetoken, timeStamp, this_readVTF_obj->sampholder, nummets, exchangetype);

	}//exchangetype == 1

      return(1);
    }//if threaded
  else
    {

      /* Check that the desired function, and group
	 exist in the appropriate lookUpTrees; otherwise, insert
	 in an attempt at traceRecord correction. */

      node * a_FuncNode =
	this_readVTF_obj->functionLookUp->getFuncNode(statetoken);
      if(a_FuncNode == 0)
	{
	  string the_statename;
	  the_statename = "UNDEFINED_FUNCTION";

	  this_readVTF_obj->functionLookUp->insert(	this_readVTF_obj->nextToken,
							statetoken,
							&the_statename,
							&the_statename);

	  this_readVTF_obj->nextToken = this_readVTF_obj->nextToken + 5;

	  a_FuncNode =
	    this_readVTF_obj->functionLookUp->getFuncNode(statetoken);
	}//if

      int group = (*a_FuncNode).getGroupID();
      node * a_GroupNode =
	this_readVTF_obj->activityLookUp->getFuncNode(group);
      if(a_GroupNode == 0)
	{
	  string the_activityname;
	  the_activityname = "UNDEFINED_GROUP";

	  this_readVTF_obj->activityLookUp->insert(	group,
							group,
							&the_activityname,
							&the_activityname);
	}//if


      if(exchangetype == 2)
	{
	  /* Now push node onto this cpu's internal stack.
	     The stack is designed to handle everything
	     within it's structure and methods, it has
	     access to the other data structures internal
	     to this 'cpu' node. */

	  nodeC * a_CPUnode =
	    this_readVTF_obj->cpu_tree->getCPUNode(cpuid);
	  if(a_CPUnode == 0)
	    {
	      /* Doesn't exist, insert it. */
	      int group = (*a_FuncNode).getGroupID();
	      string the_statename;
	      the_statename = *(a_FuncNode->getFunctionName());
	      string the_activityname;
	      the_activityname = *(a_GroupNode->getFunctionName());
	      this_readVTF_obj->cpu_tree->insert(	cpuid,
							group,
							statetoken,
							&the_statename,
							&the_activityname,
							startingTime,
							endingTime);
	    }//if

	  (*a_CPUnode).push(statetoken, timeStamp, this_readVTF_obj->sampholder, nummets, exchangetype);
	}

      if(exchangetype == 1)
	{
	  /* When I encounter an exchange, I want to:
	     (i) try inserting this 'cpu' node into the cpu_tree
	     if it isn't already.
	     (ii) insert the 'function' node into this 'cpu'
	     nodes internal funcTree.
	     (iii) increment that internal 'function' nodes
	     call_counter by 1.

	     This is actually easy because of the structures
	     and functionality I have already built in. All
	     we need to do is call:
	     this_readVTF_obj->cpu_tree->insert()
	     and pass in the appropriate parameters, then the
	     insert method does all of the work for us. It checks
	     whether the 'cpu' node already exists in the tree,
	     if no, it creates is. If yes, it moves right on
	     and trys to insert the 'function' node into the
	     'cpu' nodes internal funcTree. If it already exists,
	     it simply returns. The last thing we need to do here
	     is increment the 'function' nodes call_counter,
	     I have not designed insert() to do this for us,
	     although that would be easy enough if it seems
	     appropriate later.

	     The way we get the 'function' node information:
	     remember, we have a seperate
	     binary tree sitting in this readVTF object
	     which is built entirely out of 'function' nodes,
	     but it contains all of the 'function' nodes in the VTF;
	     therefore, it is a 'look up tree'. I use the 'statetoken'
	     parameter which is passed to my 'Exchange' handler to
	     pull the appropriate function out of the look up tree,
	     then I pull: groupID, and function name from it.
	     I pass the info to the cpu_tree->insert() method. */

	  a_FuncNode =
	    this_readVTF_obj->functionLookUp->getFuncNode(statetoken);

	  int group = (*a_FuncNode).getGroupID();

	  string the_statename;
	  the_statename = *(a_FuncNode->getFunctionName());

	  /* Get the groupName (activityname) from the
	     activityLookUp tree. */

	  a_GroupNode =
	    this_readVTF_obj->activityLookUp->getFuncNode(group);
	  string the_activityname;

	  /* The following function makes use of the 'getFunctionName()'
	     method, but keep in mind that we have stored the
	     activity (group)name there in this tree. */

	  the_activityname = *(a_GroupNode->getFunctionName());

	  /* Now attempt to insert the new 'cpu' node and
	     'function' nodes into the cpu_tree and
	     associated private funcTree of that 'cpu' node. */

	  this_readVTF_obj->cpu_tree->insert(	cpuid,
						group,
						statetoken,
						&the_statename,
						&the_activityname,
						startingTime,
						endingTime);

	  nodeC* a_CPUnode =
	    this_readVTF_obj->cpu_tree->getCPUNode(cpuid);

	  /* Now increment the 'function' node within this
	     'cpu' nodes subTree. */

	  /* Actually incrementing the functions now takes
	     place inside of the 'stack' structures based
	     on the interval implementation.

	     (*a_CPUnode).incFuncCount(statetoken);
	  */

	  (*a_CPUnode).push(statetoken, timeStamp, this_readVTF_obj->sampholder, nummets, exchangetype);

	}//if

      return(1);
    }//else (not threaded)

}//MyExchangeHandler



/****************************************************************************
	MyDefactHandler -- Processes activity definitions
****************************************************************************/
int readVTF::MyDefactHandler(void *ccbvoid,
			     int activitytoken,
			     const char *activityname)
{
  readVTF* this_readVTF_obj = (readVTF*)ccbvoid;
  /*
    cout <<  "*****ACTIVITY*****\n";
    cout << "activitytoken: " << activitytoken << "\n";
    cout << "activityname: " << activityname << "\n";
  */

  /* Doesn't really do anything usefull, just filler. */
  //int the_size = this_readVTF_obj->cpu_tree->size();

  /* OK, I'm going to reuse the binarytree structure
     in conjunction with the node class ( 'function' nodes)
     to store the activities for look up. But the binary
     tree class requires that each function's statetoken
     which we pass in be unique, therefore we will pass
     in the activitytoken for both activitytoken and
     statetoken, which is fine, as we don't plan to
     reference the statetoken when using the activityLookUp
     tree. */

  string the_activityname(activityname);

  this_readVTF_obj->nextToken = activitytoken + 1000;

  this_readVTF_obj->activityLookUp->insert(	activitytoken,
						activitytoken,
						&the_activityname,
						&the_activityname);

  return(1);
}//MyDefactHandler


/****************************************************************************
	MyDefstateHandler -- Processes state definitions
****************************************************************************/
int readVTF::MyDefstateHandler(void *ccbvoid,
			       int activitytoken,
			       int statetoken,
			       const char *statename,
			       int scltoken)
{
  readVTF* this_readVTF_obj = (readVTF*)ccbvoid;
  /*
    cout << "*****STATE*****\n";
    cout << "activitytoken: " << activitytoken << "\n";
    cout << "statetoken: " << statetoken << "\n";
    cout << "statename: " << statename << "\n";
    cout << "scltoken: " << scltoken << "\n";
  */

  /* Pull the desired info from the trace, call
     binarytree::insert(), passing it the info. It
     will in turn, create a new node storing the
     function info into our 'functionLookUp' tree. */
  string the_statename(statename);

  if(statetoken > this_readVTF_obj->maxstate)
    this_readVTF_obj->maxstate = statetoken;

  this_readVTF_obj->functionLookUp->insert(	activitytoken,
						statetoken,
						&the_statename,
						&the_statename);

  return(1);
}//MyDefstateHandler


/****************************************************************************
	MyDefthreadnumsHandler -- Processes thread number definitions
****************************************************************************/
int readVTF::MyDefthreadnumsHandler(void *ccbvoid,
				    int threadnumarraydim,
				    const int * threadnumarray)
{
  /* The following line must be executed in order to gain
     access to the member var's, i.e.
     'this_readVTF_obj.some_counter_var' where
     some_counter_var would be in the private section of the
     class definition. */
  readVTF* this_readVTF_obj = (readVTF*)ccbvoid;

  /* All we're gonna do with this right now is count the #
     of thread records we come accross. */
  this_readVTF_obj->threadNumArrayDim = threadnumarraydim;
  //this_readVTF_obj->threadNumArray = threadnumarray;

  return(1);
}//MyDefthreadnumsHandler


/****************************************************************************
	MyCPUGrpHandler -- Processes cpu group definitions.
	Used for thread identification
****************************************************************************/
int readVTF::MyCPUGrpHandler(void *ccbvoid,
			     unsigned int cpugrpid,
			     int cpuorcpugrpidarraydim,
			     const unsigned int *cpuorcpugrpidarray,
			     const char *cpugrpname)
{
  readVTF* this_readVTF_obj = (readVTF*)ccbvoid;

  string grpname = cpugrpname;

  /*All known relevant group definitions are either "TAU sample group name"
    which is the default, or of the form "Node X" where X is the CPU#.  In 
    case "Node" thread difinitions are found, this will record the associated
    threads for future use*/
  if(grpname.substr(0,5) == "Node " ||  grpname == "TAU sample group name")
    {
      int curgroups = this_readVTF_obj->ngroups;
      this_readVTF_obj->ngroups++;

      int i;
      int  * temp = new int[this_readVTF_obj->ngroups];
      for(i = 0; i<curgroups; i++)
	{
	  temp[i] = this_readVTF_obj->threadlist[i];
	}
      //delete[] this_readVTF_obj->threadlist;
      this_readVTF_obj->threadlist = new int[this_readVTF_obj->ngroups];
      for(i = 0; i<curgroups; i++)
	{
	  this_readVTF_obj->threadlist[i] = temp[i];
	}

      delete[] temp;
      this_readVTF_obj->threadlist[this_readVTF_obj->ngroups-1]=cpuorcpugrpidarraydim;
    }

  //this_readVTF_obj->threadNumArrayDim = threadnumarraydim;

  //this_readVTF_obj->threadNumArray = threadnumarray;

  return(1);
}//MyCPUGrpHandler


/****************************************************************************
	MyDefclkperiodHandler -- Processes clock period definition
****************************************************************************/
int readVTF::MyDefclkperiodHandler(void *ccbvoid,
				   double clkperiod)
{
  readVTF * this_readVTF_obj = (readVTF*)ccbvoid;
  this_readVTF_obj->clockPeriod = clkperiod;
  return(1);
}//MyDefclkperiodHandler


/****************************************************************************
	MyDeftimeoffsetHandler -- Processes time offset definition
****************************************************************************/
int readVTF::MyDeftimeoffsetHandler(void * ccbvoid,
				    double timeoffset)
{
  readVTF * this_readVTF_obj = (readVTF*)ccbvoid;
  this_readVTF_obj->timeOffset = timeoffset;
  return(1);
}//MyDeftimeoffsetHandler


/****************************************************************************
	MyUnrecognizableHandler -- Processes unrecognizable entries
****************************************************************************/
int readVTF::MyUnrecognizableHandler(void * ccbvoid,
				     double lastvalidtime,
				     int numberofunrecognizablechars,
				     int typeofunrecognizablerecord,
				     const char * unrecognizablerecord)
{
  //readVTF * this_readVTF_obj = (readVTF*)ccbvoid;

  string unrecon(unrecognizablerecord);

  cerr << "\nUn-recognizable Record Found:\n";
  cerr << unrecon << "\n";
  cerr << "Record Type: " << typeofunrecognizablerecord << "\n\n";
  return(1);
}//MyUnrecognizableHandler


/****************************************************************************
	MyCPUNameHandler -- Processes CPU Name definitions
****************************************************************************/
int readVTF::MyCPUNameHandler(void *fcb,
			      unsigned int cpuid,
			      const char *cpuname)
{
  //cout << cpuid << " " << cpuname << endl;
  return(1);
}


/****************************************************************************
	MyDefsampHandler -- Processes sample definitions
****************************************************************************/
int readVTF::MyDefsampHandler(void * ccbvoid,
			      int sampletoken,
			      int sampleclasstoken,
			      int iscpugrpsamp,
			      unsigned int cpuorcpugrpid,
			      int valuetype,
			      const void *valuebounds,
			      int dodifferentiation,
			      int datarephint,
			      const char *samplename,
			      const char *sampleunit)
{
  readVTF* this_readVTF_obj = (readVTF*)ccbvoid;

  string the_samplename(samplename);

  if(dodifferentiation == 0)
    {
      int test = sampletoken;
      if(test <= this_readVTF_obj->maxstate)
	{ test += this_readVTF_obj->maxstate+1; }
      //This metric is not associated with other functions.  Insert it into the function tree
      // When this  sample type is found the sample data will be added to the totals for the relevant cpu_node
      this_readVTF_obj->functionLookUp->insert(	test,
						test,
						&the_samplename,
						&the_samplename);
      this_readVTF_obj->usermets++;//Update the number of user define metrics.

      //cout << "A 0 metric! " << sampletoken << " "<< the_samplename << endl;

    }//if

  else//A global differentiable metric
    {
      if(valuetype != 1)
	{
	  cout << "WARNING: Sample Data Type Not Unsigned Long Long" << endl;
	}		
	
      this_readVTF_obj->globmets++;//The number of such metrics defined
      delete[] this_readVTF_obj->sampholder;//Remove the old sample holder
      this_readVTF_obj->sampholder = new unsigned long long[this_readVTF_obj->globmets];//And create a resized one

      //Create an array of the names of the metrics used
      //(each metric type will be in its own folders/files under this name)
      //Add the new metric's name to the resized array on top of the previously encountered names
      string * temp = new string[this_readVTF_obj->globmets];
      int i;
      for(i = 0; i<(this_readVTF_obj->globmets-1); i++)
	{
	  temp[i] = this_readVTF_obj->sampnames[i];
	}

      delete[] this_readVTF_obj->sampnames;
      this_readVTF_obj->sampnames = new string[this_readVTF_obj->globmets];
      for(i = 0; i<(this_readVTF_obj->globmets-1); i++)
	{
	  this_readVTF_obj->sampnames[i] = temp[i];
	}

      this_readVTF_obj->sampnames[this_readVTF_obj->globmets-1]=the_samplename;
      delete[] temp;
    }
  return(1);
}//MyDefsampHandler


/****************************************************************************
	MySampHandler -- Processes sample data
****************************************************************************/
int readVTF::MySampHandler(void * ccbvoid,
			   double time,
			   unsigned int cpuorcpugrpid,
			   int samplearraydim,
			   const int *sampletokenarray,
			   const int *samplevaluetypearray,
			   const void *samplevaluearray)
{
  readVTF* this_readVTF_obj = (readVTF*)ccbvoid;

  //For every item associated with this sample set do the following
  for(int i = 0; i<samplearraydim; i++)
    {
      int test = sampletokenarray[i];
      if(test <= this_readVTF_obj->maxstate)
	{ test += this_readVTF_obj->maxstate+1; }
      //Get the function node associated with the sample token
      node * a_SampNode =
	this_readVTF_obj->functionLookUp->getFuncNode(test);//sampletokenarray[i]);

      //If the token has no node in the function look up tree...
      if(a_SampNode == 0)
	{
	  //It has to be a function-associated (differentiable) metric.
	  //If there are no such metrics defined we have an error... an undefined sample token
	  if(this_readVTF_obj->globmets == 0)
	    {
	      cout << "ERROR! Metric mismatch!" << endl;
	    }


	  //Otherwise place this sample's value in the sample holder array
	  this_readVTF_obj->sampholder[this_readVTF_obj->sampindex]=(((unsigned long long *)samplevaluearray)[i]);

	  //And increment the array's current index
	  this_readVTF_obj->sampindex++;
			
	  //printf("value=0x%x\n", ((unsigned long *)samplevaluearray)[i]);
			
	  //cout << this_readVTF_obj->sampholder[this_readVTF_obj->sampindex-1] << " " << ((unsigned long long *)samplevaluearray)[i] << " " << samplevaluetypearray[i]  << " " << sampletokenarray[i] << endl;

	  //Place the newest metric data in the array
	  //If the array is full we have the time and all metric data for the event
	  //Run the most recently encountered call with the time/metric tuple
	  if(this_readVTF_obj->sampindex >= this_readVTF_obj->globmets)
	    {
	      if(this_readVTF_obj->lastcall == 0)
		{
		  this_readVTF_obj->DowntoBatch(this_readVTF_obj);
		}
	      else
		if(this_readVTF_obj->lastcall == 1)
		  {
		    this_readVTF_obj->UpfromBatch(this_readVTF_obj);
		  }
		else
		  if(this_readVTF_obj->lastcall == 2)
		    {
		      this_readVTF_obj->UptoBatch(this_readVTF_obj);
		    }
		  else
		    {
		      cout << "ERROR!  Lastcall not set!" << endl;
		    }

	      this_readVTF_obj->sampindex = -1;
	      this_readVTF_obj->lastcall = -1;
	    }
	}//if
      else//If the sample is found in  the tree this is a user defined metric.  Update the counters
	{
	  double startingTime = this_readVTF_obj->startTime;
	  double endingTime = this_readVTF_obj->endTime;

	  if((startingTime == -1) || (time >= startingTime && time <= endingTime))

	    /*||
	      (startingTime == -1 && time <= endingTime) || 
	      (endingTime == -1  && time >= startingTime))*/
	    {
	      //Get the cpu tree node associated with this metric
	      nodeC * a_CPUnode =
		this_readVTF_obj->cpu_tree->getCPUNode(cpuorcpugrpid);
	      if(a_CPUnode == 0)
		return 1;
	      //cout << sampletokenarray[i] << " " << a_SampNode->getFunctionName() << endl;
	      a_CPUnode->insert(sampletokenarray[i],sampletokenarray[i],
				a_SampNode->getFunctionName(),
				a_SampNode->getFunctionName());//Make sure this sample token is in the right place

	      if(samplevaluetypearray[i] == VTF3_VALUETYPE_UINT)
		{
		  unsigned long long thedata = ((unsigned long long *)samplevaluearray)[i];
		  //cout << thedata << endl;
		  a_CPUnode->incSampStat(sampletokenarray[i],(double)thedata);//Increment the sample node with the data
		}//Convert the data
	      else
		if(samplevaluetypearray[i] == VTF3_VALUETYPE_FLOAT)
		  {
		    double thedata = (((double *)samplevaluearray)[i]);
		    //cout << thedata << endl;
		    a_CPUnode->incSampStat(sampletokenarray[i],thedata);//Increment the sample node with the data
		  }

	      a_SampNode->incCount();//Increment the sample node's count
	    }//if

	}//else
    }//for

  return(1);
}//MySampHandler


/****************************************************************************
	readfile -- Read the tracefile for translation
****************************************************************************/
void readVTF::readfile(void)
{
#ifdef DEBUG
  cout << "Begin READ... \n";
#endif /* DEBUG */

  /* Initialize Vampir functionality */
  (void)VTF3_InitTables();

  /* How many different record types exist */
  numrec_types = VTF3_GetRecTypeArrayDim();

  /* Allocate three arrays for:
     record type magic numbers,
     handler function pointers,
     first args to handler functions. */
  recordtypes = (int*)malloc((size_t) numrec_types * sizeof(int));
  handlers = (VTF3_handler_t*)malloc((size_t) numrec_types * sizeof(VTF3_handler_t));
  firsthandlerargs = (void**)malloc((size_t) numrec_types * sizeof(void*));

  /* Verify we didn't run out of memory */
  if(recordtypes == 0 || handlers == 0 || firsthandlerargs == 0)
    {
      cout << "no more memory\n";
      return;
    }//if

  /* Get record type magic numbers into the appropriate array */
  (void)VTF3_GetRecTypeArray(recordtypes);

  /* Store predefined copy handler function pointers in my array */
  (void)VTF3_GetCopyHandlerArray(handlers);

  /* Replace the default handlers with a pointer to our own set
     of user defined record handlers for the desired group of
     records/events. */
  for(int i = 0; i < numrec_types; ++i)
    {
      if(recordtypes[i] == VTF3_RECTYPE_DOWNTO)
	{
	  handlers[i] = (VTF3_handler_t)MyDowntoHandler;
	  /* the following assignment is not arbitrary, we must
	     pass in a copy of the instantiated class we
	     are working within. This is because we have
	     created the user record handlers as class member
	     functions, but they are declared static in their
	     prototypes. This causes errors when we try to directly
	     grab hold of member vars.*/
	  firsthandlerargs[i] = this;
	  continue;
	}//if

      if(recordtypes[i] == VTF3_RECTYPE_UPFROM)
	{
	  handlers[i] = (VTF3_handler_t)MyUpfromHandler;
	  firsthandlerargs[i] = this;
	  continue;
	}//if

      if(recordtypes[i] == VTF3_RECTYPE_UPTO)
	{
	  handlers[i] = (VTF3_handler_t)MyUptoHandler;
	  firsthandlerargs[i] = this;
	  continue;
	}//if

      if(recordtypes[i] == VTF3_RECTYPE_EXCHANGE)
	{
	  handlers[i] = (VTF3_handler_t)MyExchangeHandler;
	  firsthandlerargs[i] = this;
	  continue;
	}//if

      if(recordtypes[i] == VTF3_RECTYPE_DEFACT)
	{
	  handlers[i] = (VTF3_handler_t)MyDefactHandler;
	  firsthandlerargs[i] = this;
	  continue;
	}//if

      if(recordtypes[i] == VTF3_RECTYPE_DEFSTATE)
	{
	  handlers[i] = (VTF3_handler_t)MyDefstateHandler;
	  firsthandlerargs[i] = this;
	  continue;
	}//if

      if(recordtypes[i] == VTF3_RECTYPE_DEFTHREADNUMS)
	{
	  handlers[i] = (VTF3_handler_t)MyDefthreadnumsHandler;
	  firsthandlerargs[i] = this;
	  continue;
	}//if

      if(recordtypes[i] == VTF3_RECTYPE_DEFCLKPERIOD)
	{
	  handlers[i] = (VTF3_handler_t)MyDefclkperiodHandler;
	  firsthandlerargs[i] = this;
	  continue;
	}//IF

      if(recordtypes[i] == VTF3_RECTYPE_DEFTIMEOFFSET)
	{
	  handlers[i] = (VTF3_handler_t)MyDeftimeoffsetHandler;
	  firsthandlerargs[i] = this;
	  continue;
	}//if


      if(recordtypes[i] == VTF3_RECTYPE_UNRECOGNIZABLE)
	{
	  handlers[i] = (VTF3_handler_t)MyUnrecognizableHandler;
	  firsthandlerargs[i] = this;
	  continue;
	}//if

      //NEW!!!
      if(recordtypes[i] == VTF3_RECTYPE_DEFSAMP)
	{
	  handlers[i] = (VTF3_handler_t)MyDefsampHandler;
	  firsthandlerargs[i] = this;
	  continue;
	}//if

      if(recordtypes[i] == VTF3_RECTYPE_SAMP)
	{
	  handlers[i] = (VTF3_handler_t)MySampHandler;
	  firsthandlerargs[i] = this;
	  continue;
	}//if


      if(recordtypes[i] == VTF3_RECTYPE_DEFCPUGRP)
	{
	  handlers[i] = (VTF3_handler_t)MyCPUGrpHandler;
	  firsthandlerargs[i] = this;
	  continue;
	} //if

      if(recordtypes[i] == VTF3_RECTYPE_DEFCPUNAME)
	{
	  handlers[i] = (VTF3_handler_t)MyCPUNameHandler;
	  firsthandlerargs[i] = this;
	  continue;
	} //if

      /* All other records:
	 Install 0 as the handler for all others,
	 reset firsthandlerargs to 0. */
      handlers[i] = 0;
      firsthandlerargs[i] = 0;
    }//for

  /* Open the input file */
  ccb.fcbin = VTF3_OpenFileInput(	filename.c_str(),
					handlers,
					firsthandlerargs,
					substituteupfrom = 0);

  /* Check for failure while opening */
  if(ccb.fcbin == 0)
    {
      cout << "Couldn't open " << filename << "\n";
      return;
    }//if

  /* else, get file format */
  else
    {
      fileformat = VTF3_QueryFormat(ccb.fcbin);
      //cout << "fileformat is: " << fileformat << "\n";
    }//else

  do
    {
      bytesread = VTF3_ReadFileInput(ccb.fcbin);
      totalbytesread += bytesread;
    }
  while(bytesread != 0);

  /*
    cout << "total bytes read = " << totalbytesread << "\n";
    cout << "size of tree is: " << cpu_tree->size() << "\n";
    cpu_tree->printTree((*cpu_tree).getRootC());
    cout << "# of threadRecords: " << threadRecordCount << "\n";
  */

  /* Need to open the appropriate outFiles, named according to the
     n,c,t scheme (node, context, thread), write the associated
     info to that file, close that file, move on to the next. When
     creating the 'profile.@.@.@' type files we're not concerned
     with overwriting previous 'profile' files of similar names.
     It's perfectly fine if we completely rewrite the profiles
     each time. */

  int * threadA = 0;
  int * cpuA = 0;
  int threadrun = 0;
  if(ngroups > 1)//If we have more groups than the 'global listing' there are threads
    {
      //Create an array with one space for each thread
      //The total number of threads is provided by the last group definition
      threadrun = threadlist[ngroups-1];
      cpuA = new int[threadrun];
      threadA = new int[threadrun];
      int counter = 0;
      for(int i = 0;i<ngroups-1;i++)
	{
	  for(int j = 0; j<threadlist[i]; j++)
	    {
	      cpuA[counter] = i;
	      threadA[counter] = j;
	      counter++;
	    }
	}
    }


  if(globmets == 0)
    {
      string runname = "";
      int readSuccess = cpu_tree->writeTree((*cpu_tree).getRootC(), &destPath, usermets, &runname, -1, threadrun, threadA, cpuA);

      if(readSuccess == 0)
	{
#ifdef DEBUG
	  cout << "...completed READ\n";
#endif /* DEBUG */

	}
      else if(readSuccess > 0)
	{
	  cout << "READ failed\n";
	}
      else
	{
	  cout << "error in READ\n";
	}

    }
  else
    {
      //Create the folder for 'get time of day'.  Print the tree data.  All traces will have this metric
      string multi = (string)"MULTI_";
      string dash = (string)"_";
      string runname = multi + dash + ((string)"GET_TIME_OF_DAY");
      string syscom = (string)"mkdir " + destPath + runname;
      system(syscom.c_str());
      string modpath = destPath + (string)"/"+ runname;
      runname = dash+multi+((string)"GET_TIME_OF_DAY");
      int readSuccess = cpu_tree->writeTree((*cpu_tree).getRootC(), &modpath, usermets, &runname, -1, threadrun, threadA, cpuA);

      if(readSuccess == 0)
	{
#ifdef DEBUG
	  cout << "...completed READ\n";
#endif /* DEBUG */

	}
      else if(readSuccess > 0)
	{
	  cout << "READ failed\n";
	}
      else
	{
	  cout << "error in READ\n";
	}

      //For every other global metric (if any) create a directory and print the tree data.
      for(int i = 0; i<globmets; i++)
	{
	  runname = multi+dash+(string)sampnames[i];
	  syscom = (string)"mkdir "+destPath+runname;
	  string modpath = destPath + (string)"/"+ runname;
	  system(syscom.c_str());
	  runname = dash+multi+(string)sampnames[i];
	  int readSuccess = cpu_tree->writeTree((*cpu_tree).getRootC(), &modpath, usermets, &runname, i, threadrun, threadA, cpuA);

	  if(readSuccess == 0)
	    {
#ifdef DEBUG
	      cout << "...completed READ\n";
#endif  /*DEBUG*/

	    }
	  else if(readSuccess > 0)
	    {
	      cout << "READ failed\n";
	    }
	  else
	    {
	      cout << "error in READ\n";
	    }

	}//for
    }//else

  /* The following calls are for timestamp error detection. */
  int totalOutOrder = cpu_tree->countOutOrder((*cpu_tree).getRootC());
  int totalRepeats = cpu_tree->countRepeats((*cpu_tree).getRootC());

  if(debug)
    {
      cout << "totalOutOrder: " << totalOutOrder << "\n";
      cout << "totalRepeats: " << totalRepeats << "\n";
    }//if

  /* Free up any allocated memory */
  (void)free(recordtypes);
  (void)free(handlers);
  (void)free(firsthandlerargs);
  delete[] threadlist;
  delete[] sampholder;
  delete[] sampnames;

  /* Close all devices before leaving */
  (void)VTF3_Close(ccb.fcbin);

  return;
}//readfile()
