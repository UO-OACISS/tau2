package edu.uoregon.tau.perfdmf;


import java.sql.*;


public class DatabaseException extends Exception {

    DatabaseException(String s, SQLException sqlException) {
        this.message = s;
        this.exception = sqlException;
    }

    public String getMessage() {
        return message;
    }

    public Exception getException() {
        return exception;
    }
    
    private String message;
    private SQLException exception;

}


