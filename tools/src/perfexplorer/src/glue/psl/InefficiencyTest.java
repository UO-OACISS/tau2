/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue.psl;

import junit.framework.TestCase;
import edu.uoregon.tau.perfexplorer.glue.TrialResult;
import edu.uoregon.tau.perfexplorer.glue.Utilities;

/**
 * @author khuck
 *
 */
public class InefficiencyTest extends TestCase {

	/**
	 * @param arg0
	 */
	public InefficiencyTest(String arg0) {
		super(arg0);
	}

	/**
	 * Test method for {@link edu.uoregon.tau.perfexplorer.glue.psl.Inefficiency#Inefficiency(edu.uoregon.tau.perfexplorer.glue.PerformanceResult, edu.uoregon.tau.perfexplorer.glue.PerformanceResult, java.lang.Integer, java.lang.String, java.lang.String, int)}.
	 */
	public final void testInefficiency() {

		Application app = new Application("test");
		Version version = new Version(app, "test");
		Experiment exp = null;
		Experiment exp2 = null;
		Inefficiency ns = null;
		
/*		
	    Utilities.setSession("PERI_DB_production");
		exp = new Experiment(version, new TrialResult(Utilities.getTrial("gtc", "jaguar", "64")));
		exp2 = new Experiment(version, new TrialResult(Utilities.getTrial("gtc", "jaguar", "128")));

		ns = new Inefficiency(exp, exp2, exp.getTopCodeRegion(), Inefficiency.Scaling.WEAK);
		System.out.print("\n" + ns.getClass().getName() + ", " + exp2.getNumberOfProcessingUnits() + ": ");
		System.out.println("Holds: " + ns.holds() + ", Severity " + ns.getSeverity() + ", Confidence: " + ns.getConfidence());

		exp2 = new Experiment(version, new TrialResult(Utilities.getTrial("gtc", "jaguar", "256")));
		ns = new Inefficiency(exp, exp2, exp.getTopCodeRegion(), Inefficiency.Scaling.WEAK);
		System.out.print("\n" + ns.getClass().getName() + ", " + exp2.getNumberOfProcessingUnits() + ": ");
		System.out.println("Holds: " + ns.holds() + ", Severity " + ns.getSeverity() + ", Confidence: " + ns.getConfidence());

		exp2 = new Experiment(version, new TrialResult(Utilities.getTrial("gtc", "jaguar", "512")));
		ns = new Inefficiency(exp, exp2, exp.getTopCodeRegion(), Inefficiency.Scaling.WEAK);
		System.out.print("\n" + ns.getClass().getName() + ", " + exp2.getNumberOfProcessingUnits() + ": ");
		System.out.println("Holds: " + ns.holds() + ", Severity " + ns.getSeverity() + ", Confidence: " + ns.getConfidence());
*/	
		Utilities.setSession("perfdmf_uploaded");

		exp = new Experiment(version, new TrialResult(Utilities.getTrial("WRF", "MCR scalability callpath", "wrf-callpath-0008.ppk")));
		exp2 = new Experiment(version, new TrialResult(Utilities.getTrial("WRF", "MCR scalability callpath", "wrf-callpath-0016.ppk")));
		ns = new Inefficiency(exp, exp2, exp.getTopCodeRegion(), Inefficiency.Scaling.STRONG);
		System.out.print("\n" + ns.getClass().getName() + ", " + exp2.getNumberOfProcessingUnits() + ": ");
		System.out.println("Holds: " + ns.holds() + ", Severity " + ns.getSeverity() + ", Confidence: " + ns.getConfidence());
		
		exp2 = new Experiment(version, new TrialResult(Utilities.getTrial("WRF", "MCR scalability callpath", "wrf-callpath-0032.ppk")));
		ns = new Inefficiency(exp, exp2, exp.getTopCodeRegion(), Inefficiency.Scaling.STRONG);
		System.out.print("\n" + ns.getClass().getName() + ", " + exp2.getNumberOfProcessingUnits() + ": ");
		System.out.println("Holds: " + ns.holds() + ", Severity " + ns.getSeverity() + ", Confidence: " + ns.getConfidence());
		
		exp2 = new Experiment(version, new TrialResult(Utilities.getTrial("WRF", "MCR scalability callpath", "wrf-callpath-0064.ppk")));
		ns = new Inefficiency(exp, exp2, exp.getTopCodeRegion(), Inefficiency.Scaling.STRONG);
		System.out.print("\n" + ns.getClass().getName() + ", " + exp2.getNumberOfProcessingUnits() +  ": ");
		System.out.println("Holds: " + ns.holds() + ", Severity " + ns.getSeverity() + ", Confidence: " + ns.getConfidence());
		
		exp2 = new Experiment(version, new TrialResult(Utilities.getTrial("WRF", "MCR scalability callpath", "wrf-callpath-0128.ppk")));
		ns = new Inefficiency(exp, exp2, exp.getTopCodeRegion(), Inefficiency.Scaling.STRONG);
		System.out.print("\n" + ns.getClass().getName() + ", " + exp2.getNumberOfProcessingUnits() +  ": ");
		System.out.println("Holds: " + ns.holds() + ", Severity " + ns.getSeverity() + ", Confidence: " + ns.getConfidence());
		
		exp2 = new Experiment(version, new TrialResult(Utilities.getTrial("WRF", "MCR scalability callpath", "wrf-callpath-0256.ppk")));
		ns = new Inefficiency(exp, exp2, exp.getTopCodeRegion(), Inefficiency.Scaling.STRONG);
		System.out.print("\n" + ns.getClass().getName() + ", " + exp2.getNumberOfProcessingUnits() +  ": ");
		System.out.println("Holds: " + ns.holds() + ", Severity " + ns.getSeverity() + ", Confidence: " + ns.getConfidence());
		
		exp2 = new Experiment(version, new TrialResult(Utilities.getTrial("WRF", "MCR scalability callpath", "wrf-callpath-0512.ppk")));
		ns = new Inefficiency(exp, exp2, exp.getTopCodeRegion(), Inefficiency.Scaling.STRONG);
		System.out.print("\n" + ns.getClass().getName() + ", " + exp2.getNumberOfProcessingUnits() +  ": ");
		System.out.println("Holds: " + ns.holds() + ", Severity " + ns.getSeverity() + ", Confidence: " + ns.getConfidence());
		
	}

}
