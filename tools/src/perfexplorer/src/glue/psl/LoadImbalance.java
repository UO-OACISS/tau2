/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue.psl;

/**
 * @author khuck
 *
 */
public abstract class LoadImbalance extends SimpleProperty {

	/**
	 * 
	 */
	public LoadImbalance(Experiment e, CodeRegion r) {
		CodeRegionFilter filter = new CodeRegionFilter(r);
		Statistics st = new Statistics (e.summaryIterator(filter)) {
			protected double getValue(RegionSummary rs) {return getValueOfInterest(rs); }
		};
		double loadImbalance = st.getAvg() / st.getMax();
		severity = (1 - loadImbalance) / (1 - 1/st.getGroupSize());
	}
	
	protected abstract double getValueOfInterest(RegionSummary rs);

}
