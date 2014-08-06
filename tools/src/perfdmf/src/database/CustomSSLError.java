package edu.uoregon.tau.perfdmf.database;

/**
 * The JSSE APIs are full of checked exceptions that our simple
 * little dumb custom trust manager doesn't want to bother differentiating
 * between. They all mean "something broke".
 *
 * This runtime exception class wraps them, so they can be bubbled up
 * to terminate the app.
 */
public class CustomSSLError extends RuntimeException {
    public CustomSSLError() { super(); }
    public CustomSSLError(String msg) { super(msg); }
    public CustomSSLError(Throwable ex) { super(ex); }
    public CustomSSLError(String msg, Throwable ex) { super(msg, ex); }
}
