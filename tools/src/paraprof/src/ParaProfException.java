
package edu.uoregon.tau.paraprof;

/**
 * This exception represents programming errors on our part.  Hopefully the user
 * will send us the message given by the error window.
 * 
 * <P>CVS $Id: ParaProfException.java,v 1.2 2006/03/15 22:32:27 amorris Exp $</P>
 * @author	Alan Morris
 * @version	$Revision: 1.2 $
 */
public class ParaProfException extends RuntimeException {

    String message;
    public ParaProfException(String message) {
        this.message = message;
    }
    public String getMessage() {
        return message;
    }
    
//    public ParaProfException(Exception e) {
//        this.exception = e;
//    }
//    
//    public Exception getException() {
//        return exception;
//    }
//    
//    Exception exception; 
//    
}
