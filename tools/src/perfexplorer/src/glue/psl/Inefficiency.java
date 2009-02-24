/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue.psl;

import edu.uoregon.tau.perfexplorer.glue.PerformanceResult;

/**
 * @author khuck
 *
 */
public class Inefficiency extends SimpleProperty {

	public enum Scaling {WEAK, STRONG};
	
	/**
	 * 
	 */
	public Inefficiency(Experiment baseExp, Experiment parExp, CodeRegion r, Scaling scaling) {
		
		CodeRegionFilter f = new CodeRegionFilter(r);
		Statistics baseStats = new ExecutionTimeStatistics(baseExp.summaryIterator(f));
		double baseTime = baseStats.getMax();
		double baseCount = baseStats.getGroupSize();
		Statistics parStats = new ExecutionTimeStatistics(parExp.summaryIterator(f));
		double parTime = parStats.getMax();
		double parCount = parStats.getGroupSize();
		
		double ratio = parCount / baseCount;
		double speedup = 1;
		if (scaling == Scaling.WEAK) {
			speedup = (baseTime * ratio) / parTime;
		} else {
			speedup = baseTime / parTime;
		}
		double efficiency = speedup / ratio;
		severity = efficiency > 1.0 ? 0.0 : 1.0 - efficiency;
	}

}
