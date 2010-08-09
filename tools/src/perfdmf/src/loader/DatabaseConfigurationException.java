package edu.uoregon.tau.perfdmf.loader;

public class DatabaseConfigurationException extends RuntimeException {
    /**
	 * 
	 */
	private static final long serialVersionUID = 3653733287837669448L;
	String message;

    public DatabaseConfigurationException(String string) {
        message = string;
    }

    public String getMessage() {
        return message;
    }
}
