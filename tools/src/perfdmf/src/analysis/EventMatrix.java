package edu.uoregon.tau.perfdmf.analysis;

public class EventMatrix extends DistanceMatrix {

	public EventMatrix (int threadCount, int eventCount) {
		super (eventCount, threadCount);
		this.eventName = new String[eventCount];
	}

}
