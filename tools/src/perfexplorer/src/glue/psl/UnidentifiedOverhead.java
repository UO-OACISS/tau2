/**
 * 
 */
package glue.psl;

/**
 * @author khuck
 *
 */
public class UnidentifiedOverhead extends SimpleProperty {

	/**
	 * 
	 */
	public UnidentifiedOverhead(RegionSummary baseSummary, RegionSummary parSummary, RegionSummary rankBasis) {
		double baseTime = baseSummary.getExecutionTime();
		double parTime = parSummary.getExecutionTime();
		int q = parSummary.getExperiment().getNumberOfProcessingUnits() / baseSummary.getExperiment().getNumberOfProcessingUnits();
		double unidentifiedOverhead = (parTime - baseTime / q) - parSummary.getIdentifiedOverhead();
		severity = unidentifiedOverhead / rankBasis.getExecutionTime();
	}

}
