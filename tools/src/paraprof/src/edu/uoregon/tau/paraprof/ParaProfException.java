
package edu.uoregon.tau.paraprof;


public class ParaProfException extends Exception {

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
