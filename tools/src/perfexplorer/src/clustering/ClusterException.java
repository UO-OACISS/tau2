/*
 * Created on Mar 16, 2005
 *
 * TODO To change the template for this generated file go to
 * Window - Preferences - Java - Code Style - Code Templates
 */
package edu.uoregon.tau.perfexplorer.clustering;


/**
 * This class is used as an extention of the Exception class, for handling
 * clustering exceptions.  There is no special code here - just a renaming
 * of the Exception class.

 * @author khuck
 * <P>CVS $Id: ClusterException.java,v 1.3 2009/02/24 00:53:35 khuck Exp $</P>
 * @version 0.1
 * @since   0.1
 *
 */
public class ClusterException extends Exception {

	/**
	 * 
	 */
	public ClusterException() {
		super();
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param arg0
	 */
	public ClusterException(String arg0) {
		super(arg0);
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param arg0
	 * @param arg1
	 */
	public ClusterException(String arg0, Throwable arg1) {
		super(arg0, arg1);
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param arg0
	 */
	public ClusterException(Throwable arg0) {
		super(arg0);
		// TODO Auto-generated constructor stub
	}

}
