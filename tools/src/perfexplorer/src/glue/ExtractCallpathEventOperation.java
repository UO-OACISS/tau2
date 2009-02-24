/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import edu.uoregon.tau.perfdmf.Trial;

/**
 * @author khuck
 *
 */
public class ExtractCallpathEventOperation extends ExtractEventOperation {

	public ExtractCallpathEventOperation(PerformanceResult input) {
		super(input, buildListOfEvents(input));
	}

	public ExtractCallpathEventOperation(Trial trial) {
		super(trial, buildListOfEvents(new TrialMeanResult(trial)));
	}

	public ExtractCallpathEventOperation(List<PerformanceResult> inputs) {
		super(inputs, buildListOfEvents(inputs));
	}

	private static List<String> buildListOfEvents(PerformanceResult input) {
		List<String> events = new ArrayList<String>();
		for (String event : input.getEvents()) {
			if (event.contains(" => ")) {
				events.add(event);
			}
		}
		return events;
	}

	private static List<String> buildListOfEvents(List<PerformanceResult> inputs) {
		Set<String> events = new HashSet<String>();
		for (PerformanceResult input : inputs) {
			for (String event : input.getEvents()) {
				if (event.contains(" => ")) {
					events.add(event);
				}
			}
		}
		return new ArrayList<String>(events);
	}
}
