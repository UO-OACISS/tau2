/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

import edu.uoregon.tau.common.AlphanumComparator;
import edu.uoregon.tau.perfdmf.Trial;

/**
 * @author khuck
 *
 */
public class SplitTrialPhasesOperation extends AbstractPerformanceOperation {
	/**
	 * 
	 */
	private static final long serialVersionUID = -3994182597522257286L;
	final String phasePrefix;

	/**
	 * @param input
	 */
	public SplitTrialPhasesOperation(PerformanceResult input, String phasePrefix) {
		super(input);
		this.phasePrefix = phasePrefix;
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param trial
	 */
	public SplitTrialPhasesOperation(Trial trial, String phasePrefix) {
		super(trial);
		this.phasePrefix = phasePrefix;
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param inputs
	 */
	public SplitTrialPhasesOperation(List<PerformanceResult> inputs, String phasePrefix) {
		super(inputs);
		this.phasePrefix = phasePrefix;
		// TODO Auto-generated constructor stub
	}

	/* (non-Javadoc)
	 * @see glue.PerformanceAnalysisOperation#processData()
	 */
	public List<PerformanceResult> processData() {
		
		// iterate through the inputs, and for each one, split each phase into a separate trial.
		for (PerformanceResult input : this.inputs) {
			//List<String> phases = new ArrayList<String>();
			Set<String> phases = new TreeSet<String>(new AlphanumComparator());
			boolean nestedPhases = false;
			String regEx = "^" + phasePrefix + " \\d+_\\d+$";
			// iterate through the events, and find the phase events
			for (String event : input.getEvents()) {
				// find the events which start with the phase prefix, but are not TAU_PHASE events.
				//if (event.startsWith(phasePrefix) && !event.contains(" => ")) {
				if (event.contains(phasePrefix) && !event.contains(" => ")) {
					phases.add(event);
					if (event.matches(regEx)) {
						nestedPhases = true;
					}
				}
			}

			if (nestedPhases) {
				//List<String> tmpPhases = new ArrayList<String>();
				Set<String> tmpPhases = new TreeSet<String>(new AlphanumComparator());

				// remove the "base" iteration events.  I really don't like this.  I need to convince Boyana and Van to do something different.  But what?
				for (String currentPhase : phases) {
//					if (currentPhase.matches(regEx))  // changed by Kevin, Oct. 27, 3:18 PM - to fix accuracy of classifier
					if (!currentPhase.matches(regEx))  // changed by Kevin, Oct. 27, 3:18 PM - to fix accuracy of classifier
						tmpPhases.add(currentPhase);
				}
				phases = tmpPhases;
			}
			
			// now, iterate through the phase events
			for (String currentPhase : phases) {
//				String currentPhasePrefix = "";
/*				if (nestedPhases) {  // changed by Kevin, Oct. 27, 3:18 PM - to fix accuracy of classifier
					int start = currentPhase.indexOf("_", phasePrefix.length());
					currentPhasePrefix = currentPhase.substring(0, start);
				}
*/				List<String> phaseEvents = new ArrayList<String>();				
				// iterate through the events, and find the events in JUST THIS PHASE
				for (String event : input.getEvents()) {
					//String tmpRelation = currentPhasePrefix + "  => " + currentPhase;
					// find the events which start with the phase prefix
					//if (event.equals(currentPhase) || event.startsWith(currentPhase + " => ")) {
					if (event.equals(currentPhase) || event.contains(currentPhase + " => ")) {
					//if (event.equals(currentPhase) || 
					    //(event.contains(currentPhase + " ") && event.contains(" => ") && !event.endsWith(currentPhase))) {
						phaseEvents.add(event);
/*					} else if (nestedPhases && (event.equals(currentPhasePrefix) || event.equals(tmpRelation))) {
						phaseEvents.add(event);						  // changed by Kevin, Oct. 27, 3:18 PM - to fix accuracy of classifier
*/					}
				}
			
				// now, call ExtractEventOperation on this puppy
				PerformanceAnalysisOperation extractor = new ExtractEventOperation(input, phaseEvents);
				PerformanceResult extracted = extractor.processData().get(0);
				outputs.add(extracted);
			}
		}
		
		return this.outputs;
	}

}
