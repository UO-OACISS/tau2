package edu.uoregon.tau.perfdmf.loader;

public class DatabaseConfigurationException extends Exception {
	String message;
	public DatabaseConfigurationException(String string) {
		message = string;
	}
	public String getMessage()
	{
		return message;
	}
}
