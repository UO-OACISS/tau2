package edu.uoregon.tau.perfexplorer.server;

import java.util.LinkedList;

import edu.uoregon.tau.perfexplorer.common.RMIPerfExplorerModel;

/**
 * Implementation of a simple queue.
 *
 * <P>CVS $Id: Queue.java,v 1.3 2009/02/24 00:53:45 khuck Exp $</P>
 * @author  Kevin Huck
 * @version 0.1
 * @since   0.1
 */
public class Queue extends LinkedList
{
	/**
	 * Adds an element to the end of the queue.
	 * 
	 * @param element
	 * @return
	 */
	public RMIPerfExplorerModel enqueue(RMIPerfExplorerModel element) {
		add(element);
		return element;
	}

	/**
	 * Take the next element off the queue.
	 * 
	 * @return
	 */
	public RMIPerfExplorerModel dequeue() {
		if (size() == 0)
			return null;
		else
			return (RMIPerfExplorerModel)removeFirst();
	}

	/**
	 * Peek at the front of the line, without removing the element from the 
	 * queue.
	 * 
	 * @return
	 */
	public RMIPerfExplorerModel peekNext() {
		if (size() == 0)
			return null;
		else
			return (RMIPerfExplorerModel)getFirst();
	}
}
