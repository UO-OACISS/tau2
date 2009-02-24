/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue.psl;

/**
 * @author khuck
 *
 */
public class CodeRegionFilter implements RegionSummaryFilter {

	private CodeRegion region = null;
	
	/**
	 * 
	 */
	public CodeRegionFilter(CodeRegion region) {
		this.region = region;
	}

	/* (non-Javadoc)
	 * @see glue.psl.RegionSummaryFilter#accept(glue.psl.RegionSummary)
	 */
	public boolean accept(RegionSummary regionSummary) {
		if (regionSummary.getCodeRegion().getLongName().equals(region.getLongName()))
			return true;
		return false;
	}

}
