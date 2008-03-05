/**
 * 
 */
package glue;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import edu.uoregon.tau.perfdmf.Trial;

/**
 * @author khuck
 *
 */
public class ExtractPhasesOperation extends AbstractPerformanceOperation {

	private String phaseNamePrefix = null;
	
	/**
	 * @param input
	 */
	public ExtractPhasesOperation(PerformanceResult input, String phaseNamePrefix) {
		super(input);
		this.phaseNamePrefix = phaseNamePrefix;
	}

	/**
	 * @param trial
	 */
	public ExtractPhasesOperation(Trial trial, String phaseNamePrefix) {
		super(trial);
		this.phaseNamePrefix = phaseNamePrefix;
	}

	/**
	 * @param inputs
	 */
	public ExtractPhasesOperation(List<PerformanceResult> inputs, String phaseNamePrefix) {
		super(inputs);
		this.phaseNamePrefix = phaseNamePrefix;
	}

	/* (non-Javadoc)
	 * @see glue.PerformanceAnalysisOperation#processData()
	 */
	public List<PerformanceResult> processData() {
		this.outputs = new ArrayList<PerformanceResult>();
		for (PerformanceResult input : inputs) {
			PerformanceResult output = new DefaultResult();
			outputs.add(output);
			Set<String> foundIterations = new HashSet<String>();
			for (String event : input.getEvents()) {
				if (event.startsWith(phaseNamePrefix)) {
					int phaseID = Integer.parseInt(event.replaceAll(phaseNamePrefix, "").trim());
					for (String metric : input.getMetrics()) {
						for (Integer threadIndex : input.getThreads()) {
							output.putExclusive(foundIterations.size(), phaseNamePrefix + " measurement", metric,
									output.getExclusive(foundIterations.size(), phaseNamePrefix + " measurement", metric) + 
									input.getExclusive(threadIndex, event, metric));
							output.putExclusive(foundIterations.size(), phaseNamePrefix + " ID", metric, phaseID);
							output.putInclusive(foundIterations.size(), phaseNamePrefix + " measurement", metric,
									output.getInclusive(foundIterations.size(), phaseNamePrefix + " measurement", metric) + 
									input.getInclusive(threadIndex, event, metric));
							output.putInclusive(foundIterations.size(), phaseNamePrefix + " ID", metric, phaseID);
							output.putCalls(foundIterations.size(), phaseNamePrefix + " measurement",
									output.getCalls(foundIterations.size(), phaseNamePrefix + " measurement") + 
									input.getCalls(threadIndex, event));
							output.putCalls(foundIterations.size(), phaseNamePrefix + " ID", phaseID);
							output.putSubroutines(foundIterations.size(), phaseNamePrefix + " measurement",
									output.getSubroutines(foundIterations.size(), phaseNamePrefix + " measurement") + 
									input.getSubroutines(threadIndex, event));
							output.putSubroutines(foundIterations.size(), phaseNamePrefix + " ID", phaseID);
						}
						output.putExclusive(foundIterations.size(), phaseNamePrefix + " measurement", metric,
								output.getExclusive(foundIterations.size(), phaseNamePrefix + " measurement", metric) /
								input.getThreads().size());
						output.putInclusive(foundIterations.size(), phaseNamePrefix + " measurement", metric,
								output.getInclusive(foundIterations.size(), phaseNamePrefix + " measurement", metric) / 
								input.getThreads().size());
						output.putCalls(foundIterations.size(), phaseNamePrefix + " measurement",
								output.getCalls(foundIterations.size(), phaseNamePrefix + " measurement") / 
								input.getThreads().size());
						output.putSubroutines(foundIterations.size(), phaseNamePrefix + " measurement",
								output.getSubroutines(foundIterations.size(), phaseNamePrefix + " measurement") / 
								input.getThreads().size());
					}
					foundIterations.add(event);
				}
			}
		}
		return outputs;
	}

}
