
package edu.uoregon.tau.dms.dss;


public class DataSourceException extends Exception {

    public DataSourceException(String message) {
        this.message = message;
    }
    
    public DataSourceException(Exception e) {
        this.exception = e;
    }
    
    public String getMessage() {
        return message;
    }

    public Exception getException() {
        return exception;
    }
    
    Exception exception; 
    
    String message;
}
