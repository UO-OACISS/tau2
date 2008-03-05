package glue;

import edu.uoregon.tau.perfdmf.Trial;

import java.util.List;

import junit.framework.TestCase;

public class ProvenanceTest extends TestCase {

	public ProvenanceTest(String arg0) {
		super(arg0);
	}
	
	public final void testSave() {
		Utilities.setSession("peri_gtc");
		Trial trial = Utilities.getTrial("GTC", "ocracoke-O2", "64");
		PerformanceResult result = new TrialMeanResult(trial);
		String metric = "Time";
		PerformanceAnalysisOperation top10 = new TopXEvents(result, metric, AbstractResult.EXCLUSIVE, 10);
		List<PerformanceResult> outputs = top10.processData();
		Provenance.save();
		Provenance.listAll();
	}
}
