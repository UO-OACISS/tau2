package glue.test;

import glue.AbstractResult;
import glue.DeriveAllMetricsOperation;
import glue.ExtractEventOperation;
import glue.PerformanceAnalysisOperation;
import glue.PerformanceResult;
import glue.TrialMeanResult;
import glue.Utilities;
import java.util.List;

import rules.RuleHarness;

import junit.framework.TestCase;

public class DeriveAllMetricsOperationTest extends TestCase {

	public final void testProcessData() {
		
	    RuleHarness ruleHarness = RuleHarness.useGlobalRules("rules/GeneralRules.drl");
	    ruleHarness.addRules("rules/ApplicationRules.drl");
	    ruleHarness.addRules("rules/MachineRules.drl");

	    Utilities.setSession("PERI_DB_production");
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
