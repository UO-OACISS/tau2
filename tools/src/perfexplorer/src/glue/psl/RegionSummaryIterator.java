/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue.psl;

import java.util.Iterator;
import java.util.Set;

/**
 * @author khuck
 *
 */
public class RegionSummaryIterator implements Iterator {

	private RegionSummaryFilter filter = null;
	private Iterator<RegionSummary> summaries = null;
	private Set<RegionSummary> summarySet = null;
	private RegionSummary currentSummary = null;
	
	/**
	 * 
	 */
	public RegionSummaryIterator(RegionSummaryFilter filter, Experiment experiment) {
		this.filter = filter;
		this.summarySet = experiment.getRegionSummaries();
		reset();
	}

	/**
	 * 
	 */
	public RegionSummaryIterator(Experiment experiment) {
		this.summarySet = experiment.getRegionSummaries();
		reset();
	}

	public boolean hasNext() {
		return (this.currentSummary != null);
	}

	public RegionSummary next() {
		RegionSummary region = this.currentSummary;
		if (region != null) {
			getNext();
		}
		return region;
	}
	
	private void getNext() {
		// null the current summary
		this.currentSummary = null;
		
		while (this.summaries.hasNext()) {
			// get the next region
			RegionSummary tmp = this.summaries.next();
			
			// if there is a filter, check if this filter accepts this region summary
			if (this.filter != null) {
				if (this.filter.accept(tmp)) {
					this.currentSummary = tmp;
					break;
				}
			} else {
				this.currentSummary = tmp;
				break;
			}
		}
	}

	public void remove() {
		// TODO Auto-generated method stub
		
	}

	/**
	 * @return the filter
	 */
	public RegionSummaryFilter getFilter() {
		return filter;
	}

	/**
	 * @param filter the filter to set
	 */
	public void setFilter(RegionSummaryFilter filter) {
		this.filter = filter;
	}

	public void reset() {
		// create a new iterator
		this.summaries = summarySet.iterator();
		getNext();
	}
}
