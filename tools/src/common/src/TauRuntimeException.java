package edu.uoregon.tau.common;

/**
 * TauRuntimeException.java
 * Wraps another kind of exception in a RuntimeException with an optional message
 *       
 * <P>CVS $Id: TauRuntimeException.java,v 1.2 2008/05/15 22:22:10 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.2 $
 */
public class TauRuntimeException extends RuntimeException {

    /**
	 * 
	 */
	private static final long serialVersionUID = -230488640608256159L;
	private String message;
    private Exception exception;
    
    public TauRuntimeException(Exception e) {
        exception = e;
    }

    public TauRuntimeException(String message, Exception e) {
        this.message = message;
        exception = e;
    }

    public TauRuntimeException(String message) {
        this.message = message;
        exception = null;
    }

    public Exception getActualException() {
        return exception;
    }
    
    public String getMessage() {
        return message;
    }
    
}
