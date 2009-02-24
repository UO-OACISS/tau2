package edu.uoregon.tau.perfexplorer.glue.test;

import edu.uoregon.tau.perfexplorer.glue.AbstractResult;
import edu.uoregon.tau.perfexplorer.glue.DeriveAllMetricsOperation;
import edu.uoregon.tau.perfexplorer.glue.ExtractEventOperation;
import edu.uoregon.tau.perfexplorer.glue.PerformanceAnalysisOperation;
import edu.uoregon.tau.perfexplorer.glue.PerformanceResult;
import edu.uoregon.tau.perfexplorer.glue.TrialMeanResult;
import edu.uoregon.tau.perfexplorer.glue.Utilities;
import edu.uoregon.tau.perfexplorer.rules.RuleHarness;

import java.util.List;


import junit.framework.TestCase;

public class DeriveAllMetricsOperationTest extends TestCase {

	public final void testProcessData() {
		
	    RuleHarness ruleHarness = RuleHarness.useGlobalRules("rules/GeneralRules.drl");
	    ruleHarness.addRules("rules/ApplicationRules.drl");
	    ruleHarness.addRules("rules/MachineRules.drl");

	    Utilities.setSession("spaceghost_apart");
	    PerformanceResult trial = new TrialMeanResult(Utilities.getTrial("gtc", "jaguar", "64"));

/*	    List<PerformanceResult> outputs = new ArrayList<PerformanceResult>();
	    outputs.add(trial);
*/	    String event = trial.getMainEvent();
	    PerformanceAnalysisOperation extractor = new ExtractEventOperation(trial, event);
	    List<PerformanceResult> outputs = extractor.processData();
	    PerformanceResult extracted = outputs.get(0);
	    DeriveAllMetricsOperation derivor = new DeriveAllMetricsOperation(extracted);
	    derivor.setType(AbstractResult.INCLUSIVE);
	    outputs = derivor.processData();
	    
	    for (PerformanceResult output : outputs) {
	    	for (Integer thread : output.getThreads()) {
	    		for (String e : output.getEvents()) {
	    			for (String metric : output.getMetrics()) {
	    				System.out.println(e + " " + metric + ": " + output.getInclusive(thread, e, metric));
	    			}
	    		}
	    	}
	    }
	    
	    ruleHarness.processRules();

	}

}
