/************************************************************************
************************************************************************/

/************************************************************************
*	binarytree.h							
*	Author: Ryan K. Harris & Wyatt Spear						
*									
*	Definitions for the binary tree (binarytree) class.		
*	A binarytree is a data structure which contains nodes,		
*	the nodes themselves will maintain data concerning thier	
*	position in relation to other nodes, the binary tree structure	
*	provides methods for manipulating the nodes and their		
*	internal data.
*
*	CURRENTLY -- (2) node types are available:
		'node' class nodes and,
		'nodeC' class nodes.
		
		node class -- These nodes represent distinct functions.
		nodeC class -- These nodes represent distinct cpu's.
		
		A single binarytree structure must only be used with one type,
		they ARE NOT mixable.
		
		I am overloading this binarytree class in order to implement
		functionality for two node types, the idea being that the
		method names are identical for trees of either type, though
		the node types cannot be mixed, node type checking will
		be implemented so that inappropriate types cannot be
		returned or set by functions; it is therefore up to the
		programmer to keep track of which type of tree they have created
		and what node type ('node' or 'nodeC') will be returned by
		'get()' methods.
		
		The alternative was to overload the original 'node' class
		forming one class that could be used for both function
		representation and cpu representation. The problem here is that
		this would likely cause headache later when I try to
		implement call tracing using some sort of stack structure,
		where each cpu would need it's own stack structure for tracing
		it's function calls.
		
		Technically, if a binarytree were created using one type,
		then completely cleared of all it's nodes including the root,
		a new root of the alternate type could be inserted, thus
		turning the tree into the alternate type of tree. This is
		confusing and is avoided since we never remove nodes from
		the structures (provide no removeNode() method).
*************************************************************************
*	class binarytree						
*									
*	Public Member functions:						
*		getRoot -- return the root of this tree			
*		insert -- inserts a node into the tree			
*		size -- returns number of nodes in tree
*		getFuncNode -- returns pointer to a node w/matching functionID
		getCPUNode -- returns pointer to a node w/matching CPUID
*		printTree -- prints out the contents of the tree
 
		#getRoot() --returns root of this tree.
		node * getRoot(void) -- return the root of this tree
		nodeC * getRootC(void)
 
		#insert() -- inserts node into this tree.
		void insert(groupID, funcID, funcName, actName) -- inserts a node into the tree
		void insert(cpuID, groupID, funcID, funcName, actName, startTime, endTime)
		void insert(cpuID, groupID, funcID, funcName, actName, threadID, startTime, endTime)	
 
		#size() --returns number of nodes in this tree.
		int size(void) -- returns number of nodes in tree
 
		#countSize() -- returns # of nodes in this func tree with count > 0.
		int countSize(void)
 
		#getFuncNode() -- returns pointer to a node w/matching functionID
		node * getFuncNode(int nodeID)
		node * getFuncNode(node * the_root, string * funcName)
		nodeC * getCPUNode(int nodeID)
 
		#printTree() -- prints out the contents of the tree
		void printTree(node * the_root)
		void printTree(nodeC * the_root)
 
		#writeTree() -- writes the contents of each 'function' node in this funcTree
					which resides inside of a 'cpu' node.
		int writeTree(node * the_root, ofstream * outFile)
		int writeTree(nodeC * the_root, string * n_destPath)
		int writeTree(nodeC * the_root,string * n_destPath, int cpuID)
 
		#countOutOrder() -- sums the 'totalOutOrder' variables from each 'cpu' node
		int countOutOrder(nodeC * the_root)
 
		#countRepeats() --sums the 'repeats' variables from each 'cpu' node
		int countRepeats(nodeC * the_root)
************************************************************************/
#ifndef __BINARYTREE_H
#define __BINARYTREE_H

using namespace std;

#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <stdio.h>
#include "node.h"
#include "nodeC.h"

class nodeC;

class binarytree
{
private:
	/* I provide two root pointers, only one will
		be utilized after the initial insert() into
		the binarytree object. */
	node* root;
	nodeC* rootC;

	int recsize(node* the_root);
	int recsize(nodeC* the_root);
	int recCountSize(node * the_root);

	int recCountUMet(node * the_root);


	void reckill(node* the_root);
	void reckill(nodeC* the_root);

public:
	/* constructor */
	binarytree()
	{
		root = 0;
		rootC = 0;
	}//binarytree()

	/* destructor */
	~binarytree()
	{
		/* kill (delete) all nodes in the tree,
			we do this in order to free up that
			memory on the heap; part of being responsible! */
		reckill(root);
		reckill(rootC);
	}//~binarytree()

	/* Function prototypes */
	node* getRoot(void);
	nodeC* getRootC(void);

	/* insert() for inserting 'function' nodes
		into a 'function' node tree. */
	void insert(int n_groupID,
				int n_functionID,
				string * functionname,
				string * activityname);

	/* insert() for inserting 'CPU' nodes
		into a 'CPU' node tree. */
	void insert(int n_cpuID,
				int n_groupID,
				int n_functionID,
				string * functionname,
				string * activityname,
				double n_startTime,
				double n_endTime);

	/* insert() for inserting 'CPU' nodes
		into a 'CPU' node tree. These 'CPU' nodes
		will each have a internal threadTree. */
	void insert(int n_cpuID,
				int n_groupID,
				int n_functionID,
				string * functionname,
				string * activityname,
				int n_threadID,
				double n_startTime,
				double n_endTime);

	int size(void);
	int countSize(void);
	int countUMet(void);

	node* getFuncNode(int n_nodeID);
	node * getFuncNode(node * the_root, string * funcName);
	nodeC* getCPUNode(int n_nodeID);

	void printTree(node* the_root);
	void printTree(nodeC* the_root);

	int writeTree(node * the_root, ofstream * outFile, int samprun, int globmet);
	int writeTree(nodeC * the_root, string * n_destPath, int numsamp, string * sampnames, int globmets,
				  int threadrun, int * threadA, int *cpuA);
	/* And the threaded version. */
	int writeTree(	nodeC * the_root,
				   string * n_destPath,
				   int  n_cpuID, int numsamp, string * sampnames, int globmets);//n_threadNumArrayDim?

	int countOutOrder(nodeC * the_root);
	int countRepeats(nodeC * the_root);
}
;//binarytree

#endif
