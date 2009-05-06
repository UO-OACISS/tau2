package edu.uoregon.tau.perfdmf;

public class DataSourceException extends RuntimeException {

    private Exception exception;
    private String message;
    private String filename;

    public DataSourceException(String message) {
        this.message = message;
    }

    public DataSourceException(Exception e) {
        this.exception = e;
    }

    public DataSourceException(Exception e, String filename) {
        this.exception = e;
        this.filename = filename;
    }

    public String getMessage() {
        return message;
    }

    public Exception getException() {
        return exception;
    }
    
    public String getFilename() {
        return filename;
    }
}
