/************************************************************************
************************************************************************/

/************************************************************************
*	node.h																*
*	Author: Ryan K. Harris												*
*																		*
*	Definitions for the node (node) class.								*
*	A node contains (stores) all of the unique information for a		*
*	function after it has been read from the							*
*	VTF(Vampir Trace Format) file. The nodes will be placed into		*
*	a binary tree data structure.										*
*************************************************************************
*	class node
*
*	Public Member functions:
*		#getGroupID -- return this nodes groupID
*			int getGroupID(void)
*		#getFunctionID -- return this nodes functionID
*			int getFunctionID(void)
*		#getFunctionName -- return the name of the function
*			string * getFunctionName(void)
*		#getActivityName -- return the activity name of this func.
*			string * getActivityName(void)
*		#getParent -- return parent node of this node
*			node * getParent(void)
*		#getLeft -- return left child node of this node
*			node * getLeft(void)
*		#getRight -- return right child node of this node
*			node * getRight(void)
*		#setParent -- sets parent
*			void setParent(node * parent)
*		#setLeft -- sets leftChild
*			void setLeft(node * left)
*		#setRight -- sets rightChild
*			void setRight(node * right)
*		#incCount -- adds one to # of times this function
*			has been called.
*			void incCount(void)
*		#incSubrs -- adds one to counter of subrs this func has called
*			void incSubrs(void)
*		#getCount -- returns count of calls made to this func
*			int getCount(void)
*		#getSubrs -- return count of subrs called
*			int getSubrs(void)
*		#getExclusive -- return exclusiveTime total
*			double getExclusive(void)
*		#getInclusive -- return inclusiveTime total
*			double getInclusive(void)
*		#addExclusive -- add exclusive to the exclusiveTime total
*			void addExclusive(double exclusive)
*		#addInclusive -- add inclusive to the inclusiveTime total
*			void addInclusive(double inclusive)
 
*		#incSamp -- updates sample node data
*			void incSamp(int n_sampleValue)
*		#addIncSamples -- add exclusive metric data to the total
*			void addIncSamples(unsigned long * a_incsamps, int n_sampdim)
*		#addExcSamples -- add exclusive metric data to the total
*			void addIncSamples(unsigned long * a_excsamps, int n_sampdim)
	getIncSamples
		returns inclusive samples array for this function node
	getExcSamples
		returns exclusive samples array for this function node
	setSamplesdim
		Sets the size of the sample arrays
	node::setMid sets the mid value
	node::getMid -- returns 'mid'
	getSampMean -- returns 'the mean of all samples'
	//Etc for square, min, max, sum
	isSamp -- returns 'isSamp'  (0 means it is a function node)
	incSamp -- updates the total user defined sample values.
 
 
************************************************************************/
#ifndef __NODE_H
#define __NODE_H

#include <string>
using std::string;

class node
{
private:
	/* Position pointers. */
	node *parent;				// Define parent node pointer
	node *leftChild;			// Define leftChild node pointer
	node *rightChild;			// Define rightChild node pointer

	/* function info. */
	int groupID;				// Holds the activitytoken
	int functionID;				// Holds the statetoken
	string functionName;		// Holds the statename
	string activityName;		// Holds the activityname
	double  exclusiveTime;
	double inclusiveTime;

	int callCounter;//Number of calls to the function
	double sampMax;//The max, min, sum of squares, mean and sum of a user defined metric
	double sampMin;
	double sampSquare;
	double sampMean;
	double sampSum;
	int mid;//The position in the current trio of user defined metrics (0,data,0).  Only the middle is used.
	int samp;//Confirms that this node is a sample holder.
	int subrs;					// Counts # of sub-rtns called

	unsigned long long * excsamples;//The exclusive sample data
	unsigned long long * incsamples;//The inclusive sample data
	int samplesdim;

public:

	/* 	Constructor for 'function' type node takes three parameters:
		int n_groupID = ID of the group function belongs to: pass by value,
		int n_functionID = unique ID of the function: pass by value,
		char n_functionname = char array holding unique name of the function:
		pass by value. */
	node(	const int n_groupID,
		  const int n_functionID,
		  string * n_functionName,
		  string * n_activityName)
	{
		groupID = n_groupID;
		functionID = n_functionID;
		functionName = *n_functionName;
		activityName = *n_activityName;
		//parent = n_node;
		parent = 0;			// Set to NULL
		leftChild = 0;		// Set to NULL
		rightChild = 0;		// Set to NULL
		exclusiveTime = 0;
		inclusiveTime = 0;

		excsamples = 0;
		incsamples = 0;

		callCounter = 0;
		sampMax = 0;
		sampMin = 0;
		sampSquare = 0;
		sampSum = 0;
		subrs = 0;
		mid = 0;
		samp = 0;
		samplesdim = 0;
	}//node(int, int, char[])

	~node()
	{
	  delete[] excsamples;
	  delete[] incsamples;
	}

	/* Now for function prototypes. */
	int getGroupID(void);
	int getFunctionID(void);
	string *getFunctionName(void);
	string * getActivityName(void);

	node* getParent(void);
	node* getLeft(void);
	node* getRight(void);

	void setParent(node *n_parent);
	void setLeft(node *n_left);
	void setRight(node *n_right);

	void incSamp(double n_sampleValue);
	void incCount(void);
	void incSubrs(void);
	int getCount(void);
	int getSubrs(void);
	double getSampMax(void);
	double getSampMin(void);
	double getSampSquare(void);
	double getSampMean(void);
	int isSamp(void);

	void setMid(int n_advance);
	int getMid(void);

	void setSamplesdim(int dim);

	double getExclusive(void);
	double getInclusive(void);

	unsigned long long *getExcSamples(void);
	unsigned long long *getIncSamples(void);

	void addExclusive(double n_exclusive);
	void addInclusive(double n_inclusive);

	void addExcSamples(unsigned long long * a_excsamps, int n_sampdim);
	void addIncSamples(unsigned long long * a_incsamps, int n_sampdim);

}
;//node

#endif
