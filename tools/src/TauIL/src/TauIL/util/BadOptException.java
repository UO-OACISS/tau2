/************************************************************
 *
 *           File : BadOptException.java
 *         Author : Tyrel Datwyler
 *
 *    Description : Thrown in the event of a command line
 *                  switch parsing error.
 *
 ************************************************************/

package TauIL.util;

public class BadOptException extends Exception {
    private String errmsg;

    public BadOptException(String error) {
	super(error);
    }

    public String getMessage() {
	return errmsg;
    }
}
