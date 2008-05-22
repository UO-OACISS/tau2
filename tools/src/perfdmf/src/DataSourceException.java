package edu.uoregon.tau.perfdmf;

public class DataSourceException extends RuntimeException {

    private Exception exception;
    private String message;

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
}
