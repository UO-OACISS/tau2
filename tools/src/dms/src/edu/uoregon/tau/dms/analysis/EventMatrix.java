package edu.uoregon.tau.dms.analysis;

public class EventMatrix extends DistanceMatrix {

	public EventMatrix (int threadCount, int eventCount) {
		super (eventCount, threadCount);
		this.eventName = new String[eventCount];
	}

}
