/*
 * ExceptionHandler.java
 *
 * Copyright 2005                                                 
 * Department of Computer and Information Science, University of Oregon
 */
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
