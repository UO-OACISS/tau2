package edu.uoregon.tau.dms.analysis;

import java.lang.StrictMath;

public class EventMatrix extends DistanceMatrix {

	public EventMatrix (int threadCount, int eventCount) {
		super (eventCount, threadCount);
		this.eventName = new String[eventCount];
	}

}
