/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue.psl;

/**
 * @author khuck
 *
 */
public abstract class OverheadForAnyExecution extends Metaproperty {

	/**
	 * 
	 */
	protected OverheadForAnyExecution(Class<Inefficiency> property, Experiment parallelExp, RegionSummary rankBasis, CodeRegion r) {
		RegionSummaryIterator it = parallelExp.summaryIterator(new CodeRegionFilter(r));
		while (it.hasNext()) {
			RegionSummary parSummary = it.next();
			Object[] arguments = new Object[] {parSummary, rankBasis};
			add (property, arguments);
		}
	}
	
	public boolean holds() {
		return anyHolds();
	}
	
	public double getConfidence() {
		return getMinConfidence();
	}
	
	public double getSeverity() {
		return getMaxSeverity();
	}

}
