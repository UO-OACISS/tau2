package edu.uoregon.tau.perfdmf.loader;

public class DatabaseConfigurationException extends RuntimeException {
    String message;

    public DatabaseConfigurationException(String string) {
        message = string;
    }

    public String getMessage() {
        return message;
    }
}
