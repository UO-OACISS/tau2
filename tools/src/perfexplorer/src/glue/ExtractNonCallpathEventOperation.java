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
public class ExtractNonCallpathEventOperation extends ExtractEventOperation {

	/**
	 * 
	 */
	private static final long serialVersionUID = 737893750945834332L;

	public ExtractNonCallpathEventOperation(PerformanceResult input) {
		super(input, buildListOfEvents(input));
	}

	public ExtractNonCallpathEventOperation(Trial trial) {
		super(trial, buildListOfEvents(new TrialMeanResult(trial)));
	}

	public ExtractNonCallpathEventOperation(List<PerformanceResult> inputs) {
		super(inputs, buildListOfEvents(inputs));
	}

	private static List<String> buildListOfEvents(PerformanceResult input) {
		List<String> events = new ArrayList<String>();
		for (String event : input.getEvents()) {
			if (!event.contains(" => ")) {
				events.add(event);
			}
		}
		return events;
	}

	private static List<String> buildListOfEvents(List<PerformanceResult> inputs) {
		Set<String> events = new HashSet<String>();
		for (PerformanceResult input : inputs) {
			for (String event : input.getEvents()) {
				if (!event.contains(" => ")) {
					events.add(event);
				}
			}
		}
		return new ArrayList<String>(events);
	}
}
