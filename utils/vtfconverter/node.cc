/************************************************************************
 ************************************************************************/

/************************************************************************
 *	node.cc
 *	Author: Ryan K. Harris & Wyatt Spear
 *
 *	The 'node' class contains the desired function information
 *	pulled from the trace files.
 ************************************************************************/

#include "node.h"
#include <iostream>
using namespace std;

/************************************************************************
 *	node class
 *
 *	Member functions:
 *		getGroupID -- return this nodes group
 *		getFunctionID -- return this nodes functionID
 *		getFunctionName -- return the name of the function
 *		getParent -- return parent node of this node
 *		getLeft -- return left child node of this node
 *		getRight -- return right child node of this node
 *		setParent -- sets parent
 *		setLeft -- sets leftChild
 *		setRight -- sets rightChild
 ************************************************************************/


/************************************************************************
 *	node::getGroupID -- return groupID for this node
 *
 *	Parameters: none
 ************************************************************************/
int node::getGroupID(void)
{
  return(groupID);
}//getGroupID()

/************************************************************************
 *	node::getFunctionID -- return functionID for this node
 *
 *	Parameters: none
 ************************************************************************/
int node::getFunctionID(void)
{
  return(functionID);
}//getFunctionID()

/************************************************************************
 *	node::getFunctionName -- return functionName for this node
 *
 *	Parameters: none
 ************************************************************************/
string * node::getFunctionName(void)
{
  return(&functionName);
}//getFunctionName()

/************************************************************************
 *	node::getActivityName -- return activityName for this node
 *
 *	Parameters: none
 ************************************************************************/
string * node::getActivityName(void)
{
  return(&activityName);
}//getActivityName()

/************************************************************************
 *	node::getParent -- return parent of this node
 *
 *	Parameters: none
 ************************************************************************/
node* node::getParent(void)
{
  return(parent);
}//getParent()

/************************************************************************
 *	node::getLeft -- return left child of this node
 *
 *	Parameters: none
 ************************************************************************/
node* node::getLeft(void)
{
  return(leftChild);
}//getLeft()

/************************************************************************
 *	node::getRight -- return right child of this node
 *
 *	Parameters: none
 ************************************************************************/
node* node::getRight(void)
{
  return(rightChild);
}//getRight()

/************************************************************************
 *	node::setParent -- set parent node
 *
 *	Parameters: pass by address
 ************************************************************************/
void node::setParent(node *n_parent)
{
  parent = n_parent;
}//setParent(node)

/************************************************************************
 *	node::setLeft -- set leftChild node
 *
 *	Parameters: pass by address
 ************************************************************************/
void node::setLeft(node *n_left)
{
  leftChild = n_left;
}//setLeft(node)

/************************************************************************
 *	node::setRight -- set rightChild node
 *
 *	Parameters: pass by address
 ************************************************************************/
void node::setRight(node *n_right)
{
  rightChild = n_right;
}//setRight(node)

/************************************************************************
 *	node::incCount -- add one to counter, which is tracking the #
 *		of times this function has been called.
 *
 *	Parameters: none
 ************************************************************************/
void node::incCount(void)
{
  callCounter += 1;
}//incCount


/************************************************************************
 *	node::incSamp -- uupdates the total user defined sample values.
 *		
 *
 *	Parameters: int n_sampleValue
 ************************************************************************/
void node::incSamp(double n_sampleValue)
{

  if((mid == 0 || mid  == 2 || mid == 4) && n_sampleValue != 0)
    {
      mid = 4;
    }
  else
    if(mid == 0)
      {
	mid = 1;
	return;
      }
    else
      if(mid == 2)
	{
	  mid = 0;
	  return;
	}
      else
	if(mid == 1)
	  {
	    mid = 2;//This sample is in the middle so don't break
	  }


  if(callCounter == 0)
    {
      sampMax = n_sampleValue;
      sampMin = n_sampleValue;
    }
  else
    {
      if(sampMax < n_sampleValue)
	{
	  sampMax = n_sampleValue;
	}
      else
	if(sampMin > n_sampleValue)
	  {
	    sampMin = n_sampleValue;
	  }
    }
  callCounter += 1;
  //cout << callCounter << " " << sampSum << endl;
  sampSum += n_sampleValue;

  sampSquare += n_sampleValue*n_sampleValue;

  samp++;

}//incSamp


/************************************************************************
 *	node::isSamp -- returns 'isSamp'
 *
 *	Parameters: none
 ************************************************************************/
int node::isSamp(void)
{
  return(samp);
}//isSamp()


/************************************************************************
 *	node::getSampMax -- returns 'SampMax'
 *
 *	Parameters: none
 ************************************************************************/
double node::getSampMax(void)
{
  return(sampMax);
}//getSampMax()


/************************************************************************
 *	node::getSampMin -- returns 'SampMin'
 *
 *	Parameters: none
 ************************************************************************/
double node::getSampMin(void)
{
  return(sampMin);
}//getSampMin()


/************************************************************************
 *	node::getSampSquare -- returns 'SampSquare'
 *
 *	Parameters: none
 ************************************************************************/
double node::getSampSquare(void)
{
  return(sampSquare);
}//getSampSquare()






/************************************************************************
 *	node::getSampMean -- returns 'the mean of all samples'
 *
 *	Parameters: none
 ************************************************************************/
double node::getSampMean(void)
{
  return(sampSum/callCounter);
}//getSampMean()


/************************************************************************
 *	node::getMid -- returns 'mid'
 *
 *	Parameters: none
 ************************************************************************/
int node::getMid(void)
{
  return(mid);
}//getMid()

/************************************************************************
 *	node::setMid sets the mid value
 *   
 *
 *	Parameters: int value
 ************************************************************************/
void node::setMid(int value)
{
  mid = value;
}//incSubrs




/************************************************************************
 *	node::getCount -- returns 'callCounter'
 *
 *	Parameters: none
 ************************************************************************/
int node::getCount(void)
{
  return(callCounter);
}//getCount()

/************************************************************************
 *	node::incSubrs -- add one to Subrs, which is tracking the #
 *		of times this function calls a subroutine.
 *
 *	Parameters: none
 ************************************************************************/
void node::incSubrs(void)
{
  subrs += 1;
}//incSubrs

/************************************************************************
 *	node::getSubrs -- returns 'Subrs'
 *
 *	Parameters: none
 ************************************************************************/
int node::getSubrs(void)
{
  return(subrs);
}//getSubrs()

/************************************************************************
	node::getExclusive
		returns exclusive time for this function node
 
	Parameters: none
************************************************************************/
double node::getExclusive(void)
{
  return(exclusiveTime);
}//getExclusive

/************************************************************************
	node::getInclusive
		returns inclusive time for this function node
 
	Parameters: none
************************************************************************/
double node::getInclusive(void)
{
  return(inclusiveTime);
}//getInclusive

/************************************************************************
	node::addExclusive
		adds the passed amount of time to the total amount
 
	Parameters: int: time to add (pass by value)
************************************************************************/
void node::addExclusive(double n_exclusive)
{
  exclusiveTime += n_exclusive;
  return;
}//addExclusive

/************************************************************************
	node::addInclusive
		addds the passed amount of time to the total amount
 
	Parameters: int: time to add (pass by value)
************************************************************************/
void node::addInclusive(double n_inclusive)
{
  inclusiveTime += n_inclusive;
  return;
}//addInclusive




/************************************************************************
	node::setSamplesdim
		Sets the size of the sample arrays
 
	Parameters: int: samples array size
************************************************************************/
void node::setSamplesdim(int dim)
{
  delete []excsamples;
  delete []incsamples;
  
  samplesdim = dim;
  excsamples = new unsigned long long[dim];
  incsamples = new unsigned long long[dim];
  for(int i = 0; i<dim;i++)
    {
      excsamples[i] = 0;
      incsamples[i] = 0;
    }
  //cout << "Hey setdim! " << endl;
  return;
}//setSamplesdim


/************************************************************************
	node::addExecSamples
		adds the passed increases in sample data, entry by entry
 
	Parameters: int*: samples to add (pass by value)
************************************************************************/
void node::addExcSamples(unsigned long long * a_excsamps, int n_sampdim)
{
  //unsigned long * temp = a_excsamps;
  if(n_sampdim >= 0 && samplesdim == 0)
    { 

      delete []excsamples;
      delete []incsamples;

      samplesdim = n_sampdim;
      excsamples = new unsigned long long[samplesdim];
      incsamples = new unsigned long long[samplesdim];

      for(int i = 0; i<samplesdim; i++)
	{
	  //int a = temp[i];
	  excsamples[i] = a_excsamps[i];//temp[i];
	  incsamples[i] = 0;
	}

    }
  else
    for(int i = 0; i<samplesdim; i++)
      {//cout << "Hey exc! " << excsamples[i] << endl;

	/*if(excsamples[i] >= 0 && a_excsamps[i] >=0 && (excsamples[i] + a_excsamps[i]) <= 0)
	  cout << "EXC Overflow! " << excsamples[i] << " + " 
	  << a_excsamps[i] << " = " << (excsamples[i] + a_excsamps[i]) << endl;*/


	excsamples[i] += a_excsamps[i];//temp[i];

	//if(excsamples[i] <=0)
	//if(a_excsamps[i] <=0)
	//cout << "Hey exc! " << a_excsamps[i] << endl;
      }
  delete[] a_excsamps;
  return;

}//addExecSamples


/************************************************************************
	node::addIncSamples
		adds the passed increases in sample data, entry by entry
 
	Parameters: int*: samples to add (pass by value)
************************************************************************/
void node::addIncSamples(unsigned long long * a_incsamps, int n_sampdim)
{

  if(n_sampdim >= 0 && samplesdim == 0)
    {
      samplesdim = n_sampdim;

      delete []excsamples;
      delete []incsamples;

      incsamples = new unsigned long long[samplesdim];
      excsamples = new unsigned long long[samplesdim];
      for(int i = 0; i<samplesdim; i++)
	{
	  incsamples[i] = a_incsamps[i];
	  excsamples[i] = 0;
	}

    }
  else
    for(int i = 0; i<samplesdim; i++)
      {
	/*if(incsamples[i] >= 0 && a_incsamps[i] >=0 && (incsamples[i] + a_incsamps[i]) <= 0)
	  cout << "INC Overflow! " << incsamples[i] << " + " 
	  << a_incsamps[i] << " = " << (incsamples[i] + a_incsamps[i]) << endl;*/

	incsamples[i] += a_incsamps[i];

	//if(incsamples[i] <=0 && incsamples[i]-a_incsamps[i] >=0)
	//if(a_incsamps[i] <=0)
	//cout << "Hey inc! " << a_incsamps[i] << endl;

      }
  delete[] a_incsamps;
  return;
}//addExecSamples


/************************************************************************
	node::getIncSamples
		returns inclusive samples array for this function node
 
	Parameters: none
************************************************************************/
unsigned long long * node::getIncSamples(void)
{//cout << "Hey ginc! " << endl;
  return(incsamples);
}//getIncSamples

/************************************************************************
	node::getExcSamples
		returns exclusive samples array for this function node
 
	Parameters: none
************************************************************************/
unsigned long long * node::getExcSamples(void)
{//cout << "Hey gexc! " << endl;
  return(excsamples);
}//getIncSamples



