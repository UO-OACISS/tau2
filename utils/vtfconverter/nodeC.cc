/************************************************************************
************************************************************************/

/************************************************************************
	nodeC.cc
	Author: Ryan K. Harris
 
	The 'nodeC' source file implements the function code for the
	nodeC class. nodeC classes are nodes which represent distinct
	cpuID's.
************************************************************************/

#include "nodeC.h"

/************************************************************************
	nodeC class
 
	Member functions:
		getCPUID -- returns ID of CPU node as given in VTF
		getParent -- return parent node
		getLeft -- return left child
		getRight -- return right child
		setParent -- sets parent
		setLeft -- sets left child
		setRight -- sets right child
		incSampStat -- adds another sample observation to a user metric
		getFuncTreeRoot -- returns handle to the root of the internal
			function tree for this CPU node.
************************************************************************/

/************************************************************************
	nodeC::nodeC (constructor)
 
	Parameters: cpuID and 'function' node parameters: groupID,
		functionID, functionName.
************************************************************************/
nodeC::nodeC(	const int n_cpuID,
			  const int n_groupID,
			  const int n_functionID,
			  string * n_functionName,
			  string * n_activityName,
			  double n_startTime,
			  double n_endTime)
{
	cpuID = n_cpuID;

	parent = 0;
	leftChild = 0;
	rightChild = 0;

	functionName = *n_functionName;
	activityName = *n_activityName;
	//delete funcTree;
	funcTree = new binarytree();
	funcTree->insert(	n_groupID,
					  n_functionID,
					  &functionName,
					  &activityName);

	funcStack = new stack(this, n_startTime, n_endTime );
	threadTree = 0;
	stackError = 0;

	/* For timeStamp error detection. */
	lastTimeOne = -1;
	lastTimeTwo = -1;
	totalOutOrder = 0;
	totalRepeats = 0;
}//nodeC (constructor)

/************************************************************************
	nodeC::nodeC (constructor, threaded version)
		This version takes an extra parameter, 'threaded', an int
		which tells us that an extra level of tree structures is
		needed and that we must manipulate the cpuID differently.
 
	Parameters: cpuID and 'function' node parameters: groupID,
		functionID, functionName, threaded.
************************************************************************/
nodeC::nodeC(	const int n_cpuID,
			  const int n_groupID,
			  const int n_functionID,
			  string * n_functionName,
			  string * n_activityName,
			  const int n_threadID,
			  double n_startTime,
			  double n_endTime)
{

	cpuID = n_cpuID;

	parent = 0;
	leftChild = 0;
	rightChild = 0;

	functionName = *n_functionName;
	activityName = *n_activityName;

	/* We don't want to initialize the funcTree for this
		nodeC if it is intended to represent a process with
		threads. */
	/*
	funcTree = new binarytree();
	funcTree->insert(	n_groupID,
						n_functionID,
						functionname,
						activityname);
	*/
	funcTree = 0;

	/* Instead we initialize the internal nodeC tree threadTree. */
	threadTree = new binarytree();
	threadTree->insert(	n_threadID,
						n_groupID,
						n_functionID,
						&functionName,
						&activityName,
						n_startTime,
						n_endTime);

	/* Also don't want to instantiate the call stack, no need. */
	/*
	funcStack = new stack(this);
	*/

	stackError = 0;

	/* For timeStamp error detection. */
	lastTimeOne = -1;
	lastTimeTwo = -1;
	totalOutOrder = 0;
	totalRepeats = 0;
}//nodeC (constructor)

/************************************************************************
	nodeC::~nodeC (destructor)
 
	Parameters: none
************************************************************************/
nodeC::~nodeC(void)
{
	/* clean up our allocated memory */
	delete funcTree;
	delete threadTree;
	delete funcStack;
}//destructor

/************************************************************************
	nodeC::getCPUID
 
	Parameters: none
************************************************************************/
int nodeC::getCPUID(void)
{
	return(cpuID);
}//getCPUID

/************************************************************************
	nodeC::getParent
 
	Parameters: none
************************************************************************/
nodeC* nodeC::getParent(void)
{
	return(parent);
}//getParent

/************************************************************************
	nodeC::getLeft
 
	Parameters: none
************************************************************************/
nodeC* nodeC::getLeft(void)
{
	return(leftChild);
}//getLeft

/************************************************************************
	nodeC::getRight
 
	Parameters: none
************************************************************************/
nodeC* nodeC::getRight(void)
{
	return(rightChild);
}//getLeft

/************************************************************************
	nodeC::setParent
 
	Parameters: pass by address
************************************************************************/
void nodeC::setParent(nodeC *n_parent)
{
	parent = n_parent;
}//setParent(nodeC)

/************************************************************************
	nodeC::setLeft
 
	Parameters: pass by address
************************************************************************/
void nodeC::setLeft(nodeC *n_left)
{
	leftChild = n_left;
}//setLeft(nodeC)

/************************************************************************
	nodeC::setRight
 
	Parameters: pass by address
************************************************************************/
void nodeC::setRight(nodeC *n_right)
{
	rightChild = n_right;
}//setRight(nodeC)

/************************************************************************
	nodeC::getFuncTreeRoot -- returns the root of the internal
		function tree in the form of a node* the_root.
 
	Parameters: none
************************************************************************/
node* nodeC::getFuncTreeRoot(void)
{
	return((*funcTree).getRoot());
}//getFuncTreeRoot

/************************************************************************
	nodeC::insert -- insert 'function' node into this 'cpu' nodes
		private member 'funcTree' (type binarytree data structure).
		Returns silently if the node is already there or something
		goes wrong.
 
	Parameters: pass by value
************************************************************************/
void nodeC::insert(	int n_groupID,
					int n_functionID,
					string * n_functionName,
					string * n_activityName)
{
	funcTree->insert(n_groupID, n_functionID, n_functionName, n_activityName);

	//node* a_FuncNode = (*funcTree).getFuncNode(n_functionID);
	//(*a_FuncNode).setSamplesdim(dim);


	return;
}//insert()

/************************************************************************
	nodeC::incFuncCount -- will increment the count of # of times
		called for the 'function' node w/functionID.
 
	Parameters: pass by value
************************************************************************/
void nodeC::incFuncCount(int n_functionID)
{
	node* a_FuncNode = (*funcTree).getFuncNode(n_functionID);
	(*a_FuncNode).incCount();
	return;
}//incFuncCount



/************************************************************************
	nodeC::incSampStat -- will update a user defined metric
		called for the 'function' node w/sampleID.
 
	Parameters: pass by value
************************************************************************/
void nodeC::incSampStat(int n_sampleID, double n_sampleValue)
{
	node* a_FuncNode = (*funcTree).getFuncNode(n_sampleID);
	(*a_FuncNode).incSamp(n_sampleValue);
	return;
}//incFuncCount



/************************************************************************
	nodeC::push -- provide an interface through the nodeC class
		to its internal stack structure's push method.
 
	Parameters: pass by value
************************************************************************/
void nodeC::push(int n_functionID, double n_timeStamp, unsigned long long * a_samples, int n_samplesdim, int n_exchangeType)
{
	funcStack->push(n_functionID, n_timeStamp, a_samples, n_samplesdim, n_exchangeType);
	return;
}//push
