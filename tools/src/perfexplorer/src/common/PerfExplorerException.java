package common;

import java.lang.Exception;

/**
 * Base exception class for the PerfExplorer application.
 *
 * <P>CVS $Id: PerfExplorerException.java,v 1.1 2005/07/05 22:29:52 amorris Exp $</P>
 * @author khuck
 * @version 0.1
 * @since   0.1
 *
 */
public class PerfExplorerException extends Exception {
	/**
	 * Constructor.
	 * 
	 * @param message
	 * @param cause
	 */
	public PerfExplorerException (String message, Throwable cause) {
		super (message, cause);
	}
	/**
	 * Constructor.
	 * 
	 * @param message
	 */
	public PerfExplorerException (String message) {
		super (message);
	}
}

