/************************************************************************
************************************************************************/

/************************************************************************
	nodeC.h
	Author: Ryan K. Harris
 
	Definitions for the nodeC class.
	This is a variation on the node class, used to create distinct nodes
	for each cpu ID. These in turn will have an internal pointer to the
	root of a tree composed entirely of function nodes(objects of type
	node, not nodeC), specifically the nodes run on that cpu.
*************************************************************************
	class nodeC
 
	Member functions:
		getCPUID -- returns ID of CPU node as given in VTF
		getParent -- return parent node
		getLeft -- return left child
		getRight -- return right child
		setParent -- sets parent
		setLeft -- sets left child
		setRight -- sets right child
		getFuncTreeRoot -- returns handle to the root of the internal
			function tree for this CPU node.
		insert -- inserts a 'function' node into it's 'funcTree'
		incFuncCount -- increments the count of times called for the
			'function' node w/functionID.
		push -- an interface method for accessing the stack.push
			method.
		obsolPush -- used by obsolStack.
 
		incSampStat -- updates a sample node's user defined data values
************************************************************************/
#ifndef __NODEC_H
#define __NODEC_H

#include <string>
using std::string;

#include "binarytree.h"
#include "node.h"
#include "stack.h"

class stack;
class binarytree;

class nodeC
{
private:
	nodeC* parent;
	nodeC* leftChild;
	nodeC* rightChild;
	int cpuID;
	string functionName;
	string activityName;
	stack * funcStack;

public:
	binarytree* funcTree;
	binarytree * threadTree;
	int stackError;

	double lastTimeOne;
	double lastTimeTwo;
	int totalOutOrder;
	int totalRepeats;

	/* Constructor */
	nodeC(	const int n_cpuID,
		   const int n_groupID,
		   const int n_functionID,
		   string * n_functionName,
		   string * n_activityName,
		   double n_startTime,
		   double n_endTime);

	/* Constructor for the threaded version. */
	nodeC(	const int n_cpuID,
		   const int n_groupID,
		   const int n_functionID,
		   string * n_functionName,
		   string * n_activityName,
		   const int n_threaded,
		   double n_startTime,
		   double n_endTime);

	/* destructor */
	~nodeC();

	int getCPUID(void);

	nodeC* getParent(void);
	nodeC* getLeft(void);
	nodeC* getRight(void);

	void setParent(nodeC* n_parent);
	void setLeft(nodeC* n_left);
	void setRight(nodeC* n_right);

	node* getFuncTreeRoot(void);

	void insert(int n_groupID,
				int n_functionID,
				string * n_functionName,
				string * n_activityName);

	void incFuncCount(int n_functionID);
	void incSampStat(int n_sampleID, double n_sampleValue);
	void push(int n_functionID, double n_timeStamp, unsigned long long * a_samples, int n_samplesdim, int n_exchangeType);


}
;//class nodeC

#endif
