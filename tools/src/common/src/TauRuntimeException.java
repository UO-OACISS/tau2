package edu.uoregon.tau.common;

/**
 * TauRuntimeException.java
 * Wraps another kind of exception in a RuntimeException with an optional message
 *       
 * <P>CVS $Id: TauRuntimeException.java,v 1.1 2006/03/03 02:47:03 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.1 $
 */
public class TauRuntimeException extends RuntimeException {

    private String message;
    private Exception exception;
    
    public TauRuntimeException(Exception e) {
        exception = e;
    }

    public TauRuntimeException(String message, Exception e) {
        this.message = message;
        exception = e;
    }

    public Exception getActualException() {
        return exception;
    }
    
    public String getMessage() {
        return message;
    }
    
}
