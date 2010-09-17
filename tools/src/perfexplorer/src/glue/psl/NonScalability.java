/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue.psl;

import java.util.List;
/**
 * @author khuck
 *
 */
public class NonScalability extends SimpleProperty {

	/**
	 * 
	 */
	public NonScalability(Experiment baseExp, List<Experiment> parallelExps, CodeRegion r) {
		CodeRegionFilter f = new CodeRegionFilter(r);
		//double baseTime = baseExp.summaryIterator(f).next().getExecutionTime();
		Statistics baseStats = new ExecutionTimeStatistics(baseExp.summaryIterator(f));
		double baseTime = baseStats.getMax();
		double baseCount = baseStats.getGroupSize();
		double minEfficiency = 1.0;
		double sumEfficiencies = 0.0;
		double avgEfficiency = 0.0;
		
		for (Experiment parallelExp : parallelExps) {
			Statistics stats = new ExecutionTimeStatistics(parallelExp.summaryIterator(f));
			double parTime = stats.getMax();
			double parCount = stats.getGroupSize();
			double efficiency = (baseTime * baseCount) / (parTime * parCount);
			if (efficiency > 1.0) {
				efficiency = 1.0;
			}
			minEfficiency = Math.min(minEfficiency, efficiency);
			sumEfficiencies += efficiency;
		}
		avgEfficiency = sumEfficiencies / parallelExps.size();
		severity = avgEfficiency - minEfficiency;
	}

}
