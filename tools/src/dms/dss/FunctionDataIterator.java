package dms.dss;

import java.util.*;
import dms.dss.*;

/**
 * Description: Implements the standard list interface.
 *
 * We maintain the concept of the ListIterator.  That is, there is no
 * current element. The index member variable is treated as lying between elements.
 * Rather than actually doing this, the index below represents the next
 * element. Thus when we call next(), we return the element at index
 * BEFORE we increment. A call to previous returns the element at index
 * AFTER decrementing index. Thus, alternating calls to next and previous
 * return the same element. As required by the ListIterator specification.
 *
 * <P>CVS $Id: FunctionDataIterator.java,v 1.1 2004/03/01 03:46:20 khuck Exp $</P>
 * @author	Kevin Huck, Robert Bell
 * @version	0.1
 */
public class FunctionDataIterator extends DataSessionIterator {

/**
 * Standard Constructor for the DataSessionIterator class.
 *
 * @param	inVector Vector object to be converted to an Iterator
 */
	public FunctionDataIterator(int size, int metricCount, int[] node, int[] context, int[] thread, int[] function, int[] numCalls, int[] numSubroutines, double[] data){

		//Check to make sure that the Vector is not null.
		if(size <= 0 || metricCount <= 0)
			throw new IllegalArgumentException();

		//Safe to continue.				
		this.size = size;
		this.metricCount = metricCount;
		this.node = node;
		this.context = context;
		this.thread = thread;
		this.function = function;
		this.numCalls = numCalls;
		this.numSubroutines = numSubroutines;
		this.data = data;
	}

	//List Iterator implemetation. Additional methods follow afterwards.

	public void add(Object o){
		throw new UnsupportedOperationException();
	}

	public boolean hasNext(){
		if(index < size)
			return true;
		else
			return false;
	}

	public boolean hasPrevious(){
		if(index == 0)
			return false;
		else
			return true;
	}

	public Object next(){
		FunctionDataObject nextObject = new FunctionDataObject();

		//Check to make sure we are not over the end.
		if(index >= size){
			throw new NoSuchElementException();
		}
		else{
		//Since, by the specification, alternating calls to next and previous
		//return the same element. We get the element first, and then increment
		//the index.  The reverse of previous(). See instance notes below.
			nextObject.setNode(node[index]);
			nextObject.setContext(context[index]);
			nextObject.setThread(thread[index]);
			nextObject.setFunctionIndexID(function[index]);
			nextObject.setNumCalls(numCalls[index]);
			nextObject.setNumSubroutines(numSubroutines[index]);
			int current = 5 * metricCount * index;
			for (int i = 0 ; i < metricCount ; i++) {
				nextObject.setInclusivePercentage(i, data[current++]);
				nextObject.setInclusive(i, data[current++]);
				nextObject.setInclusivePercentage(i, data[current++]);
				nextObject.setExclusive(i, data[current++]);
				nextObject.setInclusivePerCall(i, data[current++]);
			}

			//Increment the index.
			index++;
		}

		return nextObject;
	}

	public int nextIndex(){
		return index++;
	}

	public Object previous(){
		FunctionDataObject previousObject = new FunctionDataObject();

		//Check to make sure we are not over the end.
		if(index == 0){
			throw new NoSuchElementException();
		}
		else{
		//Since, by the specification, alternating calls to next and previous
		//return the same element. We decrement, then get the element.
		//The reverse of next(). See instance notes below.
			index--;

			previousObject.setNode(node[index]);
			previousObject.setContext(context[index]);
			previousObject.setThread(thread[index]);
			previousObject.setFunctionIndexID(function[index]);
			previousObject.setNumCalls(numCalls[index]);
			previousObject.setNumSubroutines(numSubroutines[index]);
			int current = 5 * metricCount * index;
			for (int i = 0 ; i < metricCount ; i++) {
				previousObject.setInclusivePercentage(i, data[current++]);
				previousObject.setInclusive(i, data[current++]);
				previousObject.setInclusivePercentage(i, data[current++]);
				previousObject.setExclusive(i, data[current++]);
				previousObject.setInclusivePerCall(i, data[current++]);
			}
		}

		return previousObject;
	}

	public int previousIndex(){
		return index--;
	}

	public void remove(){
		throw new UnsupportedOperationException();
	}

	public void set(Object o){
		throw new UnsupportedOperationException();
	}


		//Methods in addition to the standard ListIterator.

		//Resets the Iterator to its initial state.
/**
 * Resets the Iterator to its initial state.
 *
 */
	public void reset(){
		index = 0;
	}

/**
 * Gives the number of elements this iterator maintains.
 *
 * @return int number of elements in the iterator.
 */
	public int size(){
		return size;
	}

	private int[] node = null;
	private int[] context = null;
	private int[] thread = null;
	private int[] function = null;
	private int[] numCalls = null;
	private int[] numSubroutines = null;
	private double[] data = null;
	private int metricCount = 1;

}
