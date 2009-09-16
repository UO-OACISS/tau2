/************************************************************************
 ************************************************************************/

/************************************************************************
 *	binarytree.cc
 *	Author: Ryan K. Harris & Wyatt Spear
 *
 *	A class which creates a binary tree structure containing
 *	nodes defined by the 'node' class.
 ************************************************************************/

#include "binarytree.h"
#include <string.h>

/************************************************************************
 *	binarytree::getRoot -- return this tree's root node
 *
 *	Parameters: none
 ************************************************************************/
node* binarytree::getRoot(void)
{
  /* Verify root isn't NULL, otherwise return(0). */
  if( (root == 0) && (rootC == 0) )
    {
      return(0);
    }
  if( (root != 0) && (rootC == 0) )
    {
      return(root);
    }
  /* Else, we got a problem, this should never happen.
     This implies rootC != 0, BAD. */
  else
    {
      return(0);
    }
}//getRoot()

/************************************************************************
 *	binarytree::getRootC -- return this tree's root nodeC
 *
 *	Parameters: none
 ************************************************************************/
nodeC* binarytree::getRootC(void)
{
  if( (root == 0) && (rootC == 0))
    {
      return(0);
    }
  if( (root == 0) && (rootC != 0))
    {
      return(rootC);
    }
  /* Else, we've run into a situation where root != 0,
     this is bad and we simply return 0. */
  else
    {
      return(0);
    }
}//getRootC()

/************************************************************************
 *	binarytree::insert() -- insert a new 'function' node into this tree,
 *		but if the node is already in the tree, it returns silently,
 *		i.e. does nothing.
 *
 *	Parameters: takes functions groupID, functionID, and functionName
 ************************************************************************/
void binarytree::insert(int n_groupID,
			int n_functionID,
			string * functionname,
			string * activityname)
{
  /* first, verify the node isn't already in the tree! */
  if( (getFuncNode(n_functionID)) != 0)
    {
      return;
    }


  /* Create a temp node (n_node) with the passed info. */
  node* n_node = new node(	n_groupID,
				n_functionID,
				functionname,
				activityname);


  /* Check for NULL root */
  if( (root == 0) && (rootC == 0) )
    {
      root = n_node;
      /* Don't need to set the parent, left, or right
	 child since this is the only node currently
	 in the tree. */
      return;
    }//if
  else if( (root != 0) && (rootC == 0))
    {
      /* Start at root and descend, locate a place to
	 place the new node */
      node* temp_node = root;
      int set
	= 0;
      while(!set
	    )
	{
	  /* If new node (n_node).functionID < current node
	     (temp_node).functionID, move left */
	  if( ((*n_node).getFunctionID() ) < ((*temp_node).getFunctionID()) )
	    {
	      /* If child == 0, set temp_node.leftchild = n_node */
	      if( ((*temp_node).getLeft()) == 0 )
		{
		  (*temp_node).setLeft(n_node);
		  (*n_node).setParent(temp_node);
		  set
		    = 1;
		  continue;
		}//if

	      /* else, set temp_node = temp_node.leftchild and start again */
	      else
		{
		  temp_node = (*temp_node).getLeft();
		  continue;
		}//else
	    }//if

	  /* Else, move right! */
	  else
	    {
	      /* If child == 0, set temp_node.rightchild = n_node */
	      if( ((*temp_node).getRight()) == 0 )
		{
		  (*temp_node).setRight(n_node);
		  (*n_node).setParent(temp_node);
		  set
		    = 1;
		  continue;
		}//if

	      /*else, set temp_node = temp_node.rightchild and start agian */
	      else
		{
		  temp_node = (*temp_node).getRight();
		  continue;
		}//else
	    }//else
	}//while
      return;
    }//else

  /* Else, we've encountered an unacceptable condition where,
     either root != 0 or both root && rootC != 0. Just return. */
  else
    {
      return;
    }
}//insert()

/************************************************************************
 *	binarytree::insert -- insert a new 'CPU' node into this tree,
 *		but if the node is already in the tree, it returns silently,
 *		i.e. does nothing.
 *
 *	Parameters: takes CPUID, and initial function info.
 ************************************************************************/
void binarytree::insert(int n_cpuID,
			int n_groupID,
			int n_functionID,
			string * functionname,
			string * activityname,
			double n_startTime,
			double n_endTime)
{
  /* first, verify the node isn't already in the tree!
     If it is, we can't simply return, because we
     still have to try and insert the 'function' into
     this cpu's function_tree. */
  nodeC* temp_node = getCPUNode(n_cpuID);
  if( (temp_node) != 0)
    {
      temp_node->insert(	n_groupID,
				n_functionID,
				functionname,
				activityname);
      return;
    }//if


  /* Create a temp node (n_node) with the passed info. */
  nodeC* n_node = new nodeC(	n_cpuID,
				n_groupID,
				n_functionID,
				functionname,
				activityname,
				n_startTime,
				n_endTime);


  /* Check for NULL root */
  if( (root == 0) && (rootC == 0) )
    {
      rootC = n_node;
      /* Don't need to set the parent, left, or right
	 child since this is the only node currently
	 in the tree. */
      return;
    }//if
  else if( (root == 0) && (rootC != 0))
    {
      /* Start at root and descend, locate a place to
	 place the new node */
      temp_node = rootC;
      int set
	= 0;
      while(!set
	    )
	{
	  /* If new node (n_node).functionID < current node
	     (temp_node).functionID, move left */
	  if( ((*n_node).getCPUID() ) < ((*temp_node).getCPUID()) )
	    {
	      /* If child == 0, set temp_node.leftchild = n_node */
	      if( ((*temp_node).getLeft()) == 0 )
		{
		  (*temp_node).setLeft(n_node);
		  (*n_node).setParent(temp_node);
		  set
		    = 1;
		  continue;
		}//if

	      /* else, set temp_node = temp_node.leftchild and start again */
	      else
		{
		  temp_node = (*temp_node).getLeft();
		  continue;
		}//else
	    }//if

	  /* Else, move right! */
	  else
	    {
	      /* If child == 0, set temp_node.rightchild = n_node */
	      if( ((*temp_node).getRight()) == 0 )
		{
		  (*temp_node).setRight(n_node);
		  (*n_node).setParent(temp_node);
		  set
		    = 1;
		  continue;
		}//if

	      /*else, set temp_node = temp_node.rightchild and start agian */
	      else
		{
		  temp_node = (*temp_node).getRight();
		  continue;
		}//else
	    }//else
	}//while
      return;
    }//else

  /* Else, we've encountered an unacceptable condition where,
     either root != 0 or both root && rootC != 0. Just return. */
  else
    {
      return;
    }
}//insert()

/************************************************************************
 *	binarytree::insert -- insert a new 'CPU' node into this tree,
 *		but if the node is already in the tree, it returns silently,
 *		i.e. does nothing.
			------These 'CPU' nodes have internal threadTree's. --------
			*
			*	Parameters: takes CPUID, and initial function info.
			************************************************************************/
void binarytree::insert(int n_cpuID,
			int n_groupID,
			int n_functionID,
			string * functionname,
			string * activityname,
			int n_threadID,
			double n_startTime,
			double n_endTime)
{
  /* first, verify the node isn't already in the tree!
     If it is, we can't simply return, because we
     still have to try and insert the 'thread' and
     'function' into this cpu's threadTree. */
  nodeC* temp_node = getCPUNode(n_cpuID);
  if( (temp_node) != 0)
    {
      temp_node->threadTree->insert(	n_threadID,
					n_groupID,
					n_functionID,
					functionname,
					activityname,
					n_startTime,
					n_endTime);
      return;
    }//if


  /* Create a temp node (n_node) with the passed info. */
  nodeC* n_node = new nodeC(	n_cpuID,
				n_groupID,
				n_functionID,
				functionname,
				activityname,
				n_threadID,
				n_startTime,
				n_endTime);


  /* Check for NULL root */
  if( (root == 0) && (rootC == 0) )
    {
      rootC = n_node;
      /* Don't need to set the parent, left, or right
	 child since this is the only node currently
	 in the tree. */
      return;
    }//if
  else if( (root == 0) && (rootC != 0))
    {
      /* Start at root and descend, locate a place to
	 place the new node */
      temp_node = rootC;
      int set
	= 0;
      while(!set
	    )
	{
	  /* If new node (n_node).functionID < current node
	     (temp_node).functionID, move left */
	  if( ((*n_node).getCPUID() ) < ((*temp_node).getCPUID()) )
	    {
	      /* If child == 0, set temp_node.leftchild = n_node */
	      if( ((*temp_node).getLeft()) == 0 )
		{
		  (*temp_node).setLeft(n_node);
		  (*n_node).setParent(temp_node);
		  set
		    = 1;
		  continue;
		}//if

	      /* else, set temp_node = temp_node.leftchild and start again */
	      else
		{
		  temp_node = (*temp_node).getLeft();
		  continue;
		}//else
	    }//if

	  /* Else, move right! */
	  else
	    {
	      /* If child == 0, set temp_node.rightchild = n_node */
	      if( ((*temp_node).getRight()) == 0 )
		{
		  (*temp_node).setRight(n_node);
		  (*n_node).setParent(temp_node);
		  set
		    = 1;
		  continue;
		}//if

	      /*else, set temp_node = temp_node.rightchild and start agian */
	      else
		{
		  temp_node = (*temp_node).getRight();
		  continue;
		}//else
	    }//else
	}//while
      return;
    }//else

  /* Else, we've encountered an unacceptable condition where,
     either root != 0 or both root && rootC != 0. Just return. */
  else
    {
      return;
    }
}//insert() ---threads

/************************************************************************
 *	binarytree::size -- returns # of nodes in this tree
 *
 *	Parameters: none
 ************************************************************************/
int binarytree::size(void)
{
  //node* temp_root = root;

  /* First check to see if the tree is completely empty. */
  if( (root == 0) && (rootC == 0) )
    {
      return(0);
    }

  /* Else, proceed according to what type of node this tree uses. */
  if( (root != 0) && (rootC == 0) )
    {
      /* This is a binarytree using 'function' nodes. */
      node* temp_root = root;
      return(recsize(temp_root));
    }//if

  if( (root == 0) && (rootC != 0) )
    {
      /* this is a binarytree using 'cpu' nodes. */
      nodeC* temp_root = rootC;
      return(recsize(temp_root));
    }//if

  else
    {
      /* Else, we got a problem, this should never be a
	 possibility. So I'll return (-1) to notify. */
      return(-1);
    }//else
}//size()

/************************************************************************
 *	binarytree::recsize(node*) -- returns sum of 1 + all left and right
 *		children.
 *
 *	Parameters: takes a (node) pointer
 ************************************************************************/
int binarytree::recsize(node* the_root)
{
  /* if the left child is null, we don't need to worry about it any longer */
  if( (*the_root).getLeft() == 0)
    {
      /* if the right child also NULL, return just (1) for the current node */
      if( (*the_root).getRight() == 0)
	{
	  return(1);
	}
      else
	{
	  return(1 + recsize( (*the_root).getRight() ) );
	}
    }//if

  /* Else we have a left child, check for rightchild */
  else
    {
      if( (*the_root).getRight() == 0)
	{
	  return(1 + recsize( (*the_root).getLeft() ) );
	}//if

      /* else call recsize on both existing children */
      else
	{
	  return(1 + recsize( (*the_root).getLeft() ) + recsize( (*the_root).getRight() ));
	}//else
    }//else
  return(1 + recsize((*the_root).getLeft()) + recsize((*the_root).getRight()));
}//recsize(node*)

/************************************************************************
 *	binarytree::recsize(nodeC*) -- returns sum of 1 + all left and right
 *		children.
 *
 *	Parameters: takes a (nodeC) pointer
 ************************************************************************/
int binarytree::recsize(nodeC* the_root)
{
  /* if the left child is null, we don't need to worry about it any longer */
  if( (*the_root).getLeft() == 0)
    {
      /* if the right child also NULL, return just (1) for the current node */
      if( (*the_root).getRight() == 0)
	{
	  return(1);
	}
      else
	{
	  return(1 + recsize( (*the_root).getRight() ) );
	}
    }//if

  /* Else we have a left child, check for rightchild */
  else
    {
      if( (*the_root).getRight() == 0)
	{
	  return(1 + recsize( (*the_root).getLeft() ) );
	}//if

      /* else call recsize on both existing children */
      else
	{
	  return(1 + recsize( (*the_root).getLeft() ) + recsize( (*the_root).getRight() ));
	}//else
    }//else
  return(1 + recsize((*the_root).getLeft()) + recsize((*the_root).getRight()));
}//recsize(nodeC*)

/************************************************************************
 *	binarytree::countSize -- returns # of nodes in this function tree
		with it's 'count' > 0
		*
		*	Parameters: none
		************************************************************************/
int binarytree::countSize(void)
{
  //node* temp_root = root;

  /* First check to see if the tree is completely empty. */
  if( (root == 0) && (rootC == 0) )
    {
      return(0);
    }

  /* Else, proceed according to what type of node this tree uses. */
  if( (root != 0) && (rootC == 0) )
    {
      /* This is a binarytree using 'function' nodes. */
      node* temp_root = root;
      return(recCountSize(temp_root));
    }//if

  else
    {
      /* Else, we got a problem, this should never be a
	 possibility. So I'll return (-1) to notify. */
      return(-1);
    }//else
}//countSize()

/************************************************************************
 *	binarytree::recCountSize(node*) -- returns sum of 1 + all left and right
 *		children.
 *
 *	Parameters: takes a (node) pointer
 ************************************************************************/
int binarytree::recCountSize(node* the_root)
{
  int countFlag = 0;
  if((the_root->getCount() > 0) && (the_root->getExclusive() > 0))
    {
      countFlag = 1;
    }

  /* if the left child is null, we don't need to worry about it any longer */
  if( (*the_root).getLeft() == 0)
    {
      /* if the right child also NULL, return just countFlag for the current node */
      if( (*the_root).getRight() == 0)
	{
	  return(countFlag);
	}
      else
	{
	  return(countFlag + recCountSize( (*the_root).getRight() ) );
	}
    }//if

  /* Else we have a left child, check for rightchild */
  else
    {
      if( (*the_root).getRight() == 0)
	{
	  return(countFlag + recCountSize( (*the_root).getLeft() ) );
	}//if

      /* else call recCountSize on both existing children */
      else
	{
	  return(countFlag + recCountSize( (*the_root).getLeft() ) + recCountSize( (*the_root).getRight() ));
	}//else
    }//else
}//recCountSize(node*)







/************************************************************************
 *	binarytree::countSize -- returns # of nodes in this function tree
		with it's 'count' > 0
		*
		*	Parameters: none
		************************************************************************/
int binarytree::countUMet(void)
{
  //node* temp_root = root;

  /* First check to see if the tree is completely empty. */
  if( (root == 0) && (rootC == 0) )
    {
      return(0);
    }

  /* Else, proceed according to what type of node this tree uses. */
  if( (root != 0) && (rootC == 0) )
    {
      /* This is a binarytree using 'function' nodes. */
      node* temp_root = root;
      return(recCountUMet(temp_root));
    }//if

  else
    {
      /* Else, we got a problem, this should never be a
	 possibility. So I'll return (-1) to notify. */
      return(-1);
    }//else
}//countSize()

/************************************************************************
 *	binarytree::recCountSize(node*) -- returns sum of 1 + all left and right
 *		children.
 *
 *	Parameters: takes a (node) pointer
 ************************************************************************/
int binarytree::recCountUMet(node* the_root)
{
  int countFlag = 0;
  if(the_root->isSamp() >= 1)
    {
      countFlag = 1;
    }

  /* if the left child is null, we don't need to worry about it any longer */
  if( (*the_root).getLeft() == 0)
    {
      /* if the right child also NULL, return just countFlag for the current node */
      if( (*the_root).getRight() == 0)
	{
	  return(countFlag);
	}
      else
	{
	  return(countFlag + recCountUMet( (*the_root).getRight() ) );
	}
    }//if

  /* Else we have a left child, check for rightchild */
  else
    {
      if( (*the_root).getRight() == 0)
	{
	  return(countFlag + recCountUMet( (*the_root).getLeft() ) );
	}//if

      /* else call recCountSize on both existing children */
      else
	{
	  return(countFlag + recCountUMet( (*the_root).getLeft() ) + recCountUMet( (*the_root).getRight() ));
	}//else
    }//else
}//recCountSize(node*)






/************************************************************************
	binarytree::reckill(node*) -- recursive traverses the tree, using
		the 'delete' method on all (node) pointers in order to deallocate
		the memory they are using.
 
	Parameters: takes a (node) pointer to the root of the tree
************************************************************************/
void binarytree::reckill(node* the_root)
{
  node* left;
  node* right;

  /* check for NULL root, if found, we're done */
  if(the_root == 0)
    {
      return;
    }

  left = the_root->getLeft();
  right = the_root->getRight();

  delete the_root;

  reckill(left);
  reckill(right);

}//reckill(node*)

/************************************************************************
	binarytree::reckill(nodeC*) -- recursive traverses the tree, using
		the 'delete' method on all (nodeC) pointers in order to deallocate
		the memory they are using.
 
	Parameters: takes a (nodeC) pointer to the root of the tree
************************************************************************/
void binarytree::reckill(nodeC* the_root)
{
  nodeC* left;
  nodeC* right;

  /* check for NULL root, if found, we're done */
  if(the_root == 0)
    {
      return;
    }

  left = the_root->getLeft();
  right = the_root->getRight();

  delete the_root;

  reckill(left);
  reckill(right);

}//reckill(nodeC*)

/************************************************************************
	binarytree::getFuncNode -- given a unique functionID, will return a
		pointer to the associated node if it is in the tree, otherwise
		returns (0).
 
	Parameters: takes an int value (passed by value).
************************************************************************/
node* binarytree::getFuncNode(int n_nodeID)
{
  node* temp = root;
  int tempID = -1;

  while(temp != 0)
    {
      tempID = temp->getFunctionID();

      /* check to see if currentID(tempID) == nodeID */
      if(n_nodeID == tempID)
	{
	  return(temp);
	}

      /* check to see if we move left */
      if(n_nodeID < tempID)
	{
	  temp = temp->getLeft();
	  /* then, move back to the start */
	  continue;
	}//if

      /* else, we must need to move to the right */
      else
	{
	  temp = temp->getRight();
	  /* move back to the start of the loop */
	  continue;
	}//else
    }//while

  /* if we exit the while loop, it's because temp == 0, thus
     we've found a leaf and we assume the desired node isn't
     contained within our tree. */
  return(0);

}//getFuncNode()

/************************************************************************
	binarytree::getFuncNode -- given a unique functionname, will return a
		pointer to the associated node if it is in the tree, otherwise
		returns (0).
 
	Parameters: takes a const char * passed by reference.
************************************************************************/
node * binarytree::getFuncNode(node * the_root, string * funcName)
{
  if(the_root == 0)
    {
      return(0);
    }

  string tempName;
  string the_funcName;
  the_funcName = *funcName;
  tempName = *(the_root->getFunctionName());

  /*
    int test = strcmp(the_funcName, tempName);
    if(test == 0){return(the_root);}
  */
  /* If s1 == s2, compare returns 0. */
  if(!(the_funcName.compare(tempName)))
    {
      return(the_root);
    }

  node * childNode = the_root->getLeft();
  if(getFuncNode(childNode, &the_funcName) != 0)
    {
      return(childNode);
    }

  else
    {
      childNode = the_root->getRight();
      return( getFuncNode(childNode, &the_funcName) );
    }//else


}//getFuncNode(const char *)

/************************************************************************
	binarytree::getCPUNode -- given a unique CPUID, will return a
		pointer to the associated nodeC if it is in the tree, otherwise
		returns (0).
 
	Parameters: takes an int value (passed by value).
************************************************************************/
nodeC* binarytree::getCPUNode(int n_nodeID)
{
  nodeC* temp = rootC;
  int tempID = -1;

  while(temp != 0)
    {
      tempID = temp->getCPUID();

      /* check to see if currentID(tempID) == nodeID */
      if(n_nodeID == tempID)
	{
	  return(temp);
	}

      /* check to see if we move left */
      if(n_nodeID < tempID)
	{
	  temp = temp->getLeft();
	  /* then, move back to the start */
	  continue;
	}//if

      /* else, we must need to move to the right */
      else
	{
	  temp = temp->getRight();
	  /* move back to the start of the loop */
	  continue;
	}//else
    }//while

  /* if we exit the while loop, it's because temp == 0, thus
     we've found a leaf and we assume the desired node isn't
     contained within our tree. */
  return(0);

}//getCPUNode()

/************************************************************************
	binarytree::printTree(node*) -- prints a basic description of the trees
		contents to std out. Accomplishes its task recursively.
 
	Parameters: takes a node* pointer to the root.
************************************************************************/
void binarytree::printTree(node* the_root)
{
  if(the_root == 0)
    {
      return;
    }

  else
    {
      cout << "*****" << *(the_root->getFunctionName()) << "*****\n";
      cout << "functionID: " << the_root->getFunctionID() << "\n";
      cout << "groupID: " << the_root->getGroupID() << "\n";
      cout << "call_counter: " << the_root->getCount() << "\n";
      cout << "exclusive Time: " << the_root->getExclusive() << "\n";
      cout << "inclusive Time: " << the_root->getInclusive() << "\n";

      printTree(the_root->getLeft());
      printTree(the_root->getRight());
    }//else

}//printTree(node*)

/************************************************************************
	binarytree::printTree(nodeC*) -- prints a basic description of the trees
		contents to std out. Accomplishes its task recursively.
 
	Parameters: takes a nodeC* pointer to the root.
************************************************************************/
void binarytree::printTree(nodeC* the_root)
{
  if(the_root == 0)
    {
      return;
    }

  else
    {
      cout << "********************************\n";
      cout << "*****CPU: " << the_root->getCPUID() << "*****\n";
      cout << "********************************\n";
      cout << "**subtree**\n";

      the_root->funcTree->printTree( the_root->funcTree->getRoot() );

      printTree(the_root->getLeft());
      printTree(the_root->getRight());
      return;
    }//else

}//printTree(nodeC*)

/************************************************************************
	binarytree::writeTree(node*, ofstream*) -- writes the contents of
		each 'function' node in this funcTree which resides inside
		of a 'cpu' node to--> the ofstream which is passed in by address.
 
	Parameters: takes a node* pointer to the node to act upon,
				and a pointer to an open ofstream (outFile).
************************************************************************/
int binarytree::writeTree(node * the_root, ofstream * outFile, int samprun, int globmets)
{
  if(the_root == 0)
    {
      return(0);
    }

  if(outFile == 0)
    {
      cerr << "function " << the_root->getFunctionID()
	   << " didn't receive a good handle to outFile\n";
      return(-1);
    }

  /* Need to include a check to verify that the function was
     actually called within the desired interval. */


  //(the_root->getCount() == 0) !(the_root->getIsIn)


  //If this is a sample run, rather than a time run, print only the contents of sample nodes
  //Otherwise print only the contents of function nodes
  if(samprun == 1 && the_root->isSamp() >= 1)
    {
      //numsamps ++;
      string theFuncName;
      theFuncName = *(the_root->getFunctionName());
      //strcat(sampout,);

      (*outFile) << "\"" << theFuncName			<< "\" "
		 << the_root->getCount()				<< " "
		 << the_root->getSampMax()				<< " "
		 << the_root->getSampMin()				<< " "
		 << setprecision(20) << the_root->getSampMean()			<< " "
		 << setprecision(20) << the_root->getSampSquare()			<< "\n";

      int left = writeTree(the_root->getLeft(), outFile, samprun, globmets);
      int right = writeTree(the_root->getRight(), outFile, samprun, globmets);
      return(left + right);
    }
  else
    if((!(the_root->getExclusive() > 0) && (samprun == 0)) || samprun == 1)
      {
	int left = writeTree(the_root->getLeft(), outFile,samprun, globmets);
	int right = writeTree(the_root->getRight(), outFile,samprun, globmets);
	return(left + right);
      }//if

    else
      if((samprun == 0))
	{

	  if(the_root->getCount() == 0)
	    {
	      the_root->incCount();
	    }


	  string theFuncName;
	  string theGroupName;
	  double inc = -1;
	  double exc = -1;
	  if(globmets == -1)
	    {//This run is getting the time only.  Get the node's time data.
	      inc = the_root->getExclusive();
	      exc = the_root->getInclusive();
	    }
	  else
	    {//This run is getting sample data associated with a function.
	      // Get the data associated with the sample array's index (globmets)

	      unsigned long long * test = the_root->getExcSamples();
	      inc = test[globmets];
	      test = the_root->getIncSamples();
	      exc = test[globmets];

	    }

	  theFuncName = *(the_root->getFunctionName());
	  theGroupName = *(the_root->getActivityName());


	  // remove leading and trailing quotes
	  while (theFuncName[0] == '"') {
	    theFuncName = theFuncName.substr(1,theFuncName.size());
	  }
	  while (theFuncName[theFuncName.size()-1] == '"') {
	    theFuncName = theFuncName.substr(0,theFuncName.size()-1);
	  }

	  // remove leading and trailing quotes
	  while (theGroupName[0] == '"') {
	    theGroupName = theGroupName.substr(1,theGroupName.size());
	  }
	  while (theGroupName[theGroupName.size()-1] == '"') {
	    theGroupName = theGroupName.substr(0,theGroupName.size()-1);
	  }


	  (*outFile) << "\"" << theFuncName			<< "\" "
		     << the_root->getCount()				<< " "
		     << the_root->getSubrs()				<< " "
		     << setprecision(20) << inc			<< " "
		     << setprecision(20) << exc			<< " "
		     << "0"								<< " "
		     << "GROUP=\"" << theGroupName		<< "\"\n";

	  int left = writeTree(the_root->getLeft(), outFile, samprun, globmets);
	  int right = writeTree(the_root->getRight(), outFile, samprun, globmets);
	  return(left + right + 1);

	}//else
      else
	{
	  cout << "Unknown Printing error (no clause valid)\n";
	  return 0;
	}


}//writeTree(node*)

/************************************************************************
	binarytree::writeTree(nodeC*) -- writes the contents of each 'cpu'
		node in the cpu_tree to a profile file w/appropriate name.
 
	Parameters: takes a nodeC* pointer to the root.
************************************************************************/
int binarytree::writeTree(nodeC* the_root, string * n_destPath, int numsamp,
			  string * sampnames, int globmets,
			  int threadrun, int * threadA, int * cpuA)
{
  if(the_root == 0)
    {
      return(0);
    }

  /* Check for threads. */
  if(the_root->threadTree != 0)
    {
      string destPath;
      int cpuID = the_root->getCPUID();
      destPath = *n_destPath;
      the_root->threadTree->writeTree(the_root->threadTree->getRootC(),
				      &destPath, cpuID,numsamp, sampnames, globmets);

      int left = writeTree(the_root->getLeft(), &destPath,cpuID,numsamp, sampnames, globmets);
      int right = writeTree(the_root->getRight(), &destPath,cpuID,numsamp, sampnames, globmets);
      return(left + right);
    }//Threaded

  else
    {
      /* Must create the file/overwrite if it already exists.
	 TAU uses the n,c,t (node, context, thread) name scheme.
	 Notice that I am using a C-style FILE pointer below to
	 detect a bad directory path. */

      ofstream outFile;
      FILE * fp;
      char filename[100];
      string destPath;
      int cpuID = the_root->getCPUID();

      if(the_root->stackError == 1)
	{
	  cerr << "\nErrors occurred in the stack on node: "
	       << cpuID
	       << "\nThis is likely due to an incorrectly defined record in "
	       << "the trace file.\n"
	       << "Profiles will still be written as is!\n";
	}//if

      destPath = *n_destPath;

      /* Check for a bad dir path.
	 We try to open a read only file at destPath,
	 if the 'destPath' is bad, it won't work and
	 the C-style method returns null, making the
	 pointer NULL. */
      fp = fopen(destPath.c_str(), "r");
      if(fp == 0)
	{
	  cerr << "Received bad destination path for the profile(s)\n";
	  cerr << "No profile(s) generated!\n";
	  return(1);
	}//if
      fclose(fp);


      if(threadrun == 0)
	{
	  sprintf(filename, "%s/profile.%d.", destPath.c_str(), cpuID);

	  /*cat the context onto filename. */
	  strcat(filename, "0");
	  strcat(filename, ".");

	  /* cat the threadID onto filename. */
	  strcat(filename, "0");
	}
      else
	{
	  sprintf(filename, "%s/profile.%d.0.%d", destPath.c_str(), cpuA[cpuID], threadA[cpuID]);
	  //cout << threadA[cpuID] << " " << cpuID << endl;
	}

      outFile.open(filename, ios::out|ios::trunc);

      /* Check for successfull open. */
      if(outFile.bad())
	{
	  cerr << "outFile failed to open.\n";
	  return(1);
	}

      /* Now start writing to the file. */
      int tempedFuncs = the_root->funcTree->countSize();

      if(tempedFuncs == 0)
	{
	  ++tempedFuncs;
	}
      outFile << (tempedFuncs)
	      << " templated_functions"<< *sampnames <<"\n";

      outFile << "# Name Calls Subrs Excl Incl ProfileCalls\n";

      /* pass reference to outFile to our sub-tree. */
      int funcWrite = the_root->funcTree->writeTree( the_root->funcTree->getRoot(),
						     &outFile, 0, globmets );

      //Check for a blank profile, and fill w/an 'IDLE' function.
      if(funcWrite == 0)
	{

	  outFile << "\"" << "IDLE()"			<< "\" "
		  << 1	<< " "
		  << 0	<< " "
		  << setprecision(20) << (1E-20) << " "
		  << setprecision(20) << (1E-20) << " "
		  << "0"								<< " "
		  << "GROUP=\"" << "IDLE" << "\"\n";

	}//if

      int umets = the_root->funcTree->countUMet();
      outFile << "0 aggregates\n";
      //cout << umets << endl;

      if(umets > 0)
	{
	  outFile << umets << " userevents\n";
	  outFile << "# eventname numevents max min mean sumsqr\n";
	  the_root->funcTree->writeTree( the_root->funcTree->getRoot(),
					 &outFile, 1, globmets );
	}

      outFile.close();
      int left = writeTree(the_root->getLeft(), &destPath,numsamp, sampnames,
			   globmets, threadrun, threadA, cpuA);
      int right = writeTree(the_root->getRight(), &destPath,numsamp, sampnames,
			    globmets, threadrun,threadA, cpuA);
      return(left + right);
    }//else

}//writeTree(nodeC*)

/************************************************************************
	binarytree::writeTree(nodeC*, cpuID) -- writes the
		contents of each 'thread' node in the threadTree to a profile
		file w/appropriate name (n,c,t).
 
	Parameters: takes a nodeC* pointer to the root, the destPath, and
		an int value which is a cpuID.
************************************************************************/
int binarytree::writeTree(	nodeC* the_root,
				string * n_destPath,
				int n_cpuID, int numsamp, string * sampnames, int globmets)
{
  if(the_root == 0)
    {
      return(0);
    }

  else
    {
      /* Must create the file/overwrite if it alread exists.
	 TAU uses the n,c,t (node, context, thread) name scheme. */

      ofstream outFile;
      FILE * fp;
      char filename[100];
      string destPath;
      int cpuID = n_cpuID;
      int threadID = the_root->getCPUID();

      if(the_root->stackError == 1)
	{
	  cerr << "\nErrors occurred in the stack on node: "
	       << cpuID
	       << "\nThread: "
	       << threadID
	       << "\nThis is likely due to an incorrectly defined record in "
	       << "the trace file.\n"
	       << "Profiles will still be written as is!\n";
	}//if

      destPath = *n_destPath;

      /* Check for a bad dir path.
	 We try to open a read only file at destPath,
	 if the 'destPath' is bad, it won't work and
	 the C-style method returns null, making the
	 pointer NULL. */
      fp = fopen(destPath.c_str(), "r");
      if(fp == 0)
	{
	  cerr << "Received bad destination path for the profile(s)\n";
	  cerr << "No profile(s) generated!\n";
	  return(1);
	}//if
      fclose(fp);

      sprintf(filename, "%s/profile.%d.0.%d", destPath.c_str(), cpuID, threadID);

      outFile.open(filename, ios::out|ios::trunc);

      /* Check for successfull open. */
      if(outFile.bad())
	{
	  cerr << "outFile failed to open.\n";
	  return(1);
	}

      /* Now start writing to the file. */
      int tempedFuncs = the_root->funcTree->countSize();

      if(tempedFuncs == 0)
	{
	  ++tempedFuncs;
	}

      outFile << (the_root->funcTree->countSize())
	      << " templated_functions"<< *sampnames <<"\n";

      outFile << "# Name Calls Subrs Excl Incl ProfileCalls\n";

      /* pass reference to outFile to our sub-tree. */
      int funcWrite = the_root->funcTree->writeTree( the_root->funcTree->getRoot(),
						     &outFile, 0, globmets);

      //Check for a blank profile, and fill w/an 'IDLE' function.
      if(funcWrite == 0)
	{
	  outFile << "\"" << "IDLE()"			<< "\" "
		  << 1	<< " "
		  << 0	<< " "
		  << setprecision(20) << (1E-20) << " "
		  << setprecision(20) << (1E-20) << " "
		  << "0"								<< " "
		  << "GROUP=\"" << "IDLE" << "\"\n";
	}

      outFile << "0 aggregates\n";
      outFile << numsamp << " userevents\n";

      if(numsamp > 0)
	{
	  outFile << "# eventname numevents max min mean sumsqr\n";
	  the_root->funcTree->writeTree( the_root->funcTree->getRoot(),
					 &outFile, 1, globmets );
	}
      outFile.close();
      int left = writeTree(the_root->getLeft(), &destPath, cpuID,numsamp, sampnames, globmets);
      int right = writeTree(the_root->getRight(), &destPath, cpuID,numsamp, sampnames, globmets);
      return(left + right);
    }//else

}//writeTree(nodeC*) ---Threaded


/************************************************************************
	binarytree::countOutOrder(nodeC*)
 
	Parameters: Gets passed the root of this subtree.
************************************************************************/
int binarytree::countOutOrder(nodeC * the_root)
{
  if(the_root == 0)
    {
      return(0);
    }

  /* Check for threads. */
  if(the_root->threadTree != 0)
    {

      int thisCount = the_root->threadTree->countOutOrder(the_root->threadTree->getRootC());

      int left = countOutOrder(the_root->getLeft());
      int right = countOutOrder(the_root->getRight());
      return(thisCount + left + right);
    }//Threaded

  else
    {
      int thisCount = the_root->totalOutOrder;
      int left = countOutOrder(the_root->getLeft());
      int right = countOutOrder(the_root->getRight());
      return(thisCount + left + right);
    }//else
}//countOutOrder

/************************************************************************
	binarytree::countRepeats(nodeC*)
 
	Parameters: Gets passed the root of this subtree.
************************************************************************/
int binarytree::countRepeats(nodeC * the_root)
{
  if(the_root == 0)
    {
      return(0);
    }

  /* Check for threads. */
  if(the_root->threadTree != 0)
    {

      int thisCount = the_root->threadTree->countRepeats(the_root->threadTree->getRootC());

      int left = countRepeats(the_root->getLeft());
      int right = countRepeats(the_root->getRight());
      return(thisCount + left + right);
    }//Threaded

  else
    {
      int thisCount = the_root->totalRepeats;
      int left = countRepeats(the_root->getLeft());
      int right = countRepeats(the_root->getRight());
      return(thisCount + left + right);
    }//else
}//countRepeats
