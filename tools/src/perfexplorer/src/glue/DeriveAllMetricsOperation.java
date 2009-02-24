/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue;

import java.util.List;


import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfexplorer.rules.RuleHarness;

/**
 * @author khuck
 *
 */
public class DeriveAllMetricsOperation extends AbstractPerformanceOperation {

	private int type = AbstractResult.EXCLUSIVE;
	
	/**
	 * @param input
	 */
	public DeriveAllMetricsOperation(PerformanceResult input) {
		super(input);
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param trial
	 */
	public DeriveAllMetricsOperation(Trial trial) {
		super(trial);
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param inputs
	 */
	public DeriveAllMetricsOperation(List<PerformanceResult> inputs) {
		super(inputs);
		// TODO Auto-generated constructor stub
	}

	/* (non-Javadoc)
	 * @see glue.PerformanceAnalysisOperation#processData()
	 */
	public List<PerformanceResult> processData() {
		for (PerformanceResult input : inputs) {
			PerformanceResult output = new DefaultResult(input, false);
			for (String event : input.getEvents()) {
				String timeMetric = input.getTimeMetric();
				String FPMetric = input.getFPMetric();
				String L1AccessMetric = input.getL1AccessMetric();
				String L1MissMetric = input.getL1MissMetric();
				String L2AccessMetric = input.getL2AccessMetric();
				String L2MissMetric = input.getL2MissMetric();
				String totalInstMetric = input.getTotalInstructionMetric();
				
			    // check for our needed metrics
			    if (timeMetric == null) {
			        RuleHarness.assertObject(new DataNeeded(DataNeeded.DataType.TIME));
			    }
			    if (FPMetric == null) {
			    	RuleHarness.assertObject(new DataNeeded(DataNeeded.DataType.FLOATING_POINT));
			    }
			    if (L1AccessMetric == null) {
			    	RuleHarness.assertObject(new DataNeeded(DataNeeded.DataType.CACHE_ACCESS_L1));
			    }
			    if (L1MissMetric == null) {
			    	RuleHarness.assertObject(new DataNeeded(DataNeeded.DataType.CACHE_MISS_L1));
			    }
			    if (L2AccessMetric == null) {
			    	RuleHarness.assertObject(new DataNeeded(DataNeeded.DataType.CACHE_ACCESS_L2));
			    }
			    if (L2MissMetric == null) {
			    	RuleHarness.assertObject(new DataNeeded(DataNeeded.DataType.CACHE_MISS_L2));
			    }
			    if (totalInstMetric == null) {
			    	RuleHarness.assertObject(new DataNeeded(DataNeeded.DataType.TOTAL_INSTRUCTIONS));
			    }
	
	
			    // get the computation density for this routine
			    PerformanceAnalysisOperation derivor = new DeriveMetricOperation(input, FPMetric, timeMetric, DeriveMetricOperation.DIVIDE);
			    PerformanceResult derived = derivor.processData().get(0);
			    PerformanceAnalysisOperation merger = new MergeTrialsOperation(input);
			    merger.addInput(derived);
			    output = merger.processData().get(0);

			    // get the L1 access behavior for this routine
			    derivor = new DeriveMetricOperation(output, L1AccessMetric, timeMetric, DeriveMetricOperation.DIVIDE);
			    derived = derivor.processData().get(0);
			    merger = new MergeTrialsOperation(output);
			    merger.addInput(derived);
			    output = merger.processData().get(0);
	
			    // get the L1 hit ratio
			    derivor = new DeriveMetricOperation(output, L1AccessMetric, L1MissMetric, DeriveMetricOperation.SUBTRACT);
			    derived = derivor.processData().get(0);
			    merger = new MergeTrialsOperation(output);
			    merger.addInput(derived);
			    output = merger.processData().get(0);
			    derivor = new DeriveMetricOperation(output, DerivedMetrics.L1_CACHE_HITS, L1AccessMetric, DeriveMetricOperation.DIVIDE);
			    derived = derivor.processData().get(0);
			    merger = new MergeTrialsOperation(output);
			    merger.addInput(derived);
			    output = merger.processData().get(0);
		
			    // get the L2 access behavior for this routine
			    derivor = new DeriveMetricOperation(output, L2AccessMetric, timeMetric, DeriveMetricOperation.DIVIDE);
			    derived = derivor.processData().get(0);
			    merger = new MergeTrialsOperation(output);
			    merger.addInput(derived);
			    output = merger.processData().get(0);
	
			    // get the L2 hit ratio
			    derivor = new DeriveMetricOperation(output, L2AccessMetric, L2MissMetric, DeriveMetricOperation.SUBTRACT);
			    derived = derivor.processData().get(0);
			    merger = new MergeTrialsOperation(output);
			    merger.addInput(derived);
			    output = merger.processData().get(0);
			    derivor = new DeriveMetricOperation(output, DerivedMetrics.L2_CACHE_HITS, L2AccessMetric, DeriveMetricOperation.DIVIDE);
			    derived = derivor.processData().get(0);
			    merger = new MergeTrialsOperation(output);
			    merger.addInput(derived);
			    output = merger.processData().get(0);
		    
			    // get the total instruction rate for this routine
			    derivor = new DeriveMetricOperation(output, totalInstMetric, timeMetric, DeriveMetricOperation.DIVIDE);
			    derived = derivor.processData().get(0);
			    merger = new MergeTrialsOperation(output);
			    merger.addInput(derived);
			    output = merger.processData().get(0);
			}
			outputs.add(output);
		}
		return outputs;
	}

	/**
	 * @return the type
	 */
	public int getType() {
		return type;
	}

	/**
	 * @param type the type to set
	 */
	public void setType(int type) {
		this.type = type;
	}

}
