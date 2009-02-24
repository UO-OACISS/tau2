package edu.uoregon.tau.perfexplorer.glue.test;

import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfexplorer.glue.AbstractResult;
import edu.uoregon.tau.perfexplorer.glue.PerformanceAnalysisOperation;
import edu.uoregon.tau.perfexplorer.glue.PerformanceResult;
import edu.uoregon.tau.perfexplorer.glue.Provenance;
import edu.uoregon.tau.perfexplorer.glue.TopXEvents;
import edu.uoregon.tau.perfexplorer.glue.TrialMeanResult;
import edu.uoregon.tau.perfexplorer.glue.Utilities;

import junit.framework.TestCase;

public class ProvenanceTest extends TestCase {

	public ProvenanceTest(String arg0) {
		super(arg0);
	}
	
	public final void testSave() {
		Utilities.setSession("perigtc");
		Trial trial = Utilities.getTrial("GTC", "ocracoke-O2", "64");
		PerformanceResult result = new TrialMeanResult(trial);
		String metric = "Time";
		PerformanceAnalysisOperation top10 = new TopXEvents(result, metric, AbstractResult.EXCLUSIVE, 10);
		top10.processData();
		Provenance.save();
		Provenance.listAll();
	}
}
