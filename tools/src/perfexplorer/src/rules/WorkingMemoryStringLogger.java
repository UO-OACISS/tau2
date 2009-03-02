/**
 * 
 */
package edu.uoregon.tau.perfexplorer.rules;

import java.util.ArrayList;
import java.util.List;

import org.drools.WorkingMemory;
import org.drools.audit.WorkingMemoryLogger;
import org.drools.audit.event.ActivationLogEvent;
import org.drools.audit.event.LogEvent;

/**
 * @author khuck
 *
 */
public class WorkingMemoryStringLogger extends WorkingMemoryLogger {

    private final List<String> events = new ArrayList<String>();

	/**
	 * @param arg0
	 */
	public WorkingMemoryStringLogger(WorkingMemory arg0) {
		super(arg0);
		// TODO Auto-generated constructor stub
	}

	/* (non-Javadoc)
	 * @see org.drools.audit.WorkingMemoryLogger#logEventCreated(org.drools.audit.event.LogEvent)
	 */
	@Override
	public void logEventCreated(LogEvent logEvent) {
		StringBuilder buf = new StringBuilder();
		switch (logEvent.getType()) {
		case LogEvent.ACTIVATION_CANCELLED:
	        buf.append("ACTIVATION CANCELLED\t" );
			buf.append(" ID: " + ((ActivationLogEvent)(logEvent)).getActivationId());
			break;
		case LogEvent.ACTIVATION_CREATED:
	        buf.append("ACTIVATION CREATED\t" );
			buf.append(" ID: " + ((ActivationLogEvent)(logEvent)).getActivationId());
			break;
		case LogEvent.AFTER_ACTIVATION_FIRE:
	        buf.append("AFTER ACTIVATION FIRE\t" );
			buf.append(" ID: " + ((ActivationLogEvent)(logEvent)).getActivationId());
			break;
		case LogEvent.BEFORE_ACTIVATION_FIRE:
	        buf.append("BEFORE ACTIVATION FIRE\t" );
			buf.append(" ID: " + ((ActivationLogEvent)(logEvent)).getActivationId());
			break;
		default:
			buf.append(logEvent.toString());
			break;
		}
		this.events.add(buf.toString());
    }

    /**
     * All events in the log are written to a string.
     * The log is automatically cleared afterwards.
     */
    public String toString() {
    	StringBuilder buf = new StringBuilder();
    	for (String event : this.events) {
    		buf.append(event + "\n");
    	}
    	return buf.toString();
    }
}
