package edu.uoregon.tau.dms.analysis;


public class ThreadMatrix extends DistanceMatrix {

	public ThreadMatrix (int threadCount, int eventCount) {
		super (threadCount, eventCount);
		this.eventName = new String[eventCount];
	}

}
