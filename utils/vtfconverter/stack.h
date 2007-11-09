/************************************************************************
************************************************************************/

/************************************************************************
	stack.h
	Author: Ryan K. Harris
 
	Definitions for the stack class.
	A stack is filled with trays.
 
	Notice that the stack constructor takes a reference to
	a nodeC (pass by address). What we're doing is passing in a
	reference to the parent nodeC ( 'cpu' node) of this instantiated
	stack. The reason I chose this approach is: we want every data
	structure within the parent 'cpu' node to be able to directly
	interact with every other data structure; namely, we want the
	internal stack to be able to interact directly with the internal
	'funcTree' (binarytree data structure holding the 'function' nodes
	associated to this cpuID).
*************************************************************************
	class stack
 
	Member functions:
		push -- push a tray onto the stack
				-- It also handles the time counters for each tray.
************************************************************************/
#ifndef __STACK_H
#define __STACK_H

#include "tray.h"
#include "nodeC.h"
#include "node.h"

class nodeC;

class stack
{
private:
	tray * topTray;		// Points to the tray on top
	tray * tempTray;	//For memory control
	nodeC * parent;		// 'cpu' node which owns this stack
	double startTime;	// Holds starting interval time if provided.
	double endTime;		// Holds ending interval time if provided.
	unsigned long long * startMets;//As startTime but for metrics
	unsigned long long * endMets;//As endTime but for metrics
	int startset;
	int endset;

public:
	/* Constructor: pass in one parameter, the cpuID
		of the 'cpu' node which owns this stack. */
	stack(nodeC * n_parent)
	{
		parent = n_parent;
		topTray = 0;
		tempTray = 0;

		startMets = 0;
		endMets = 0;
		startset = 0;
		endset = 0;

		startTime = -1;
		endTime = -1;
	}

	stack(nodeC * n_parent, double n_startTime, double n_endTime)
	{
		parent = n_parent;
		topTray = 0;

		startMets = 0;
		endMets = 0;
		startset = 0;
		endset = 0;

		startTime = n_startTime;
		endTime = n_endTime;
	}


	~stack()
	{
		//delete topTray;
		//delete lastEntry;
	}

	void push(int n_functionID, double n_timeStamp,
			  unsigned long long * a_samples, int n_samplesdim, int n_exchangeType);
	void newtray(int n_functionID, double n_timeStamp,
		unsigned long long * a_samples, int n_samplesdim);
	void routine(int n_functionID, double n_timeStamp,
		unsigned long long * a_samples, int n_samplesdim);
	void returnfrom(int n_functionID, double n_timeStamp,
		unsigned long long * a_samples, int n_samplesdim);
	void lastreturn(int n_functionID, double n_timeStamp, 
		unsigned long long * a_samples, int n_samplesdim);
	void norec(node * funcNode, int n_samplesdim, tray * tempTray);
}
;//stack

#endif
