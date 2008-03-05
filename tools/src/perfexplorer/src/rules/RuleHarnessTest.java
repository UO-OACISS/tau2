package rules;

import junit.framework.TestCase;
import client.ScriptFacade;
import edu.uoregon.tau.perfdmf.Trial;
import glue.DifferenceMetadataOperation;
import glue.Utilities;
import glue.TrialResult;
import glue.DifferenceOperation;

public class RuleHarnessTest extends TestCase {

	public final void testProcessRules() {
	    Utilities.setSession("perfdmf_test");
		Trial baseline = Utilities.getTrial("gtc_bench", "superscaling.jaguar", "64");
	    Trial comparison = Utilities.getTrial("gtc_bench", "superscaling.jaguar", "128");
	    DifferenceOperation diff = new DifferenceOperation(baseline);
	    diff.addInput(comparison);
	    diff.processData();
	    DifferenceMetadataOperation metaDiff = new DifferenceMetadataOperation(baseline, comparison);
/*	    String samplerules =  "/rules/PerfExplorer.drl" ;
	    System.out.println("****** Processing rules for performance data ******");
	    RuleHarness.processRules(diff, samplerules);
	    System.out.println("****** Processing rules for metadata ******");
	    RuleHarness.processRules(metaDiff, samplerules);
	    System.out.println("****** Processing old rules ******");
	    RuleHarnessOld.processRules(baseline, comparison);
*/
	    System.out.println(metaDiff.differencesAsString());
	    String samplerules =  "rules/GeneralRules.drl" ;
	    System.out.println("****** Processing Super Duper Rules! ******");
	    RuleHarness ruleHarness = new RuleHarness(samplerules);
	    ruleHarness.addRules("rules/ApplicationRules.drl");
	    ruleHarness.addRules("rules/MachineRules.drl");
	    ruleHarness.assertObject(metaDiff);
	    ruleHarness.assertObject(diff);
	    ruleHarness.processRules();
//	    System.out.println(ruleHarness.getLog());
}

}
