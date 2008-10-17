/**
 * 
 */
package glue;

import java.util.ArrayList;
import java.util.List;

import edu.uoregon.tau.perfdmf.Trial;

/**
 * @author khuck
 *
 */
public class SplitTrialPhasesOperation extends AbstractPerformanceOperation {
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
			List<String> phases = new ArrayList<String>();
			boolean nestedPhases = false;
			String regEx = "^" + phasePrefix + " \\d+_\\d+$";
			// iterate through the events, and find the phase events
			for (String event : input.getEvents()) {
				// find the events which start with the phase prefix, but are not TAU_PHASE events.
				if (event.startsWith(phasePrefix) && !event.contains(" => ")) {
					phases.add(event);
					if (event.matches(regEx)) {
						nestedPhases = true;
					}
				}
			}

			if (nestedPhases) {
				List<String> tmpPhases = new ArrayList<String>();
				// remove the "base" iteration events.  I really don't like this.  I need to convince Boyana and Van to do something different.  But what?
				for (String currentPhase : phases) {
					if (currentPhase.matches(regEx))
						tmpPhases.add(currentPhase);
				}
				phases = tmpPhases;
			}
			
			// now, iterate through the phase events
			for (String currentPhase : phases) {
				String currentPhasePrefix = "";
				if (nestedPhases) {
					int start = currentPhase.indexOf("_", phasePrefix.length());
					currentPhasePrefix = currentPhase.substring(0, start);
				}
				List<String> phaseEvents = new ArrayList<String>();				
				// iterate through the events, and find the events in JUST THIS PHASE
				for (String event : input.getEvents()) {
					String tmpRelation = currentPhasePrefix + "  => " + currentPhase;
					// find the events which start with the phase prefix
					if (event.equals(currentPhase) || event.startsWith(currentPhase + "  => ")) {
						phaseEvents.add(event);
					} else if (nestedPhases && (event.equals(currentPhasePrefix) || event.equals(tmpRelation))) {
						phaseEvents.add(event);						
					}
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
