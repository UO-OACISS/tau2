package glue.psl;

import java.util.ArrayList;
import java.util.List;

import edu.uoregon.tau.perfdmf.Trial;
import glue.TrialResult;
import glue.Utilities;
import junit.framework.TestCase;

public class NonScalabilityTest extends TestCase {

	public final void testNonScalability() {
	    Utilities.setSession("PERI_DB_production");
	    TrialResult baseResult = new TrialResult(Utilities.getTrial("gtc", "jaguar", "64"));

		Application app = new Application("test");
		Version version = new Version(app, "test");
		Experiment exp = new Experiment(version, baseResult);
		List<Experiment> exps = new ArrayList<Experiment>();
		exps.add(new Experiment(version, new TrialResult(Utilities.getTrial("gtc", "jaguar", "128"))));
		
		NonScalability ns = new NonScalability(exp, exps, exp.getTopCodeRegion());
		System.out.print("\n" + ns.getClass().getName() + ": ");
		System.out.println("Holds: " + ns.holds() + ", Severity " + ns.getSeverity() + ", Confidence: " + ns.getConfidence());
	}

}
