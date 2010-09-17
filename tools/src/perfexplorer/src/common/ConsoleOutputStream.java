package edu.uoregon.tau.perfexplorer.common;

import java.io.ByteArrayOutputStream;
import java.io.IOException;

/**
 * An OutputStream that writes contents to a Logger upon each call to flush()
 */
class ConsoleOutputStream extends ByteArrayOutputStream {
    
    //private String lineSeparator;
    
	private boolean error = false;
	private Console console = null;
    
    /**
     * Constructor
     * @param logger Logger to write to
     * @param level Level at which to write the log message
     */
    public ConsoleOutputStream(Console console, boolean error) {
        super();
        this.console = console;
        this.error = error;
        //lineSeparator = System.getProperty("line.separator");
    }
    
    /**
     * upon flush() write the existing contents of the OutputStream to the logger as 
     * a log record.
     * @throws java.io.IOException in case of error
     */
    public void flush() throws IOException {

        String record;
        synchronized(this) {
            super.flush();
            record = this.toString();
            super.reset();
        }
        
		/*
        if (record.length() == 0 || record.equals(lineSeparator)) {
            // avoid empty records
            return;
        }
		*/

        console.print(this.error, record);
    }
}