package edu.uoregon.tau.vis;

/**
 * Defines a class that will handle exceptions
 * 
 * @author amorris
 *
 * @see VisTools#setSwingExceptionHandler(ExceptionHandler)
 */
public interface ExceptionHandler {
    void handleException(Exception e);
}
