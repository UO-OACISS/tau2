/**
 * 
 */
package glue;

import java.util.List;

import server.PerfExplorerServer;

import common.EngineType;

import edu.uoregon.tau.perfdmf.Application;
import edu.uoregon.tau.perfdmf.Experiment;
import edu.uoregon.tau.perfdmf.Trial;
import junit.framework.TestCase;

/**
 * This class is a JUnit test case for the TrialResult class.
 * 
 * <P>CVS $Id: TrialResultTest.java,v 1.2 2008/03/05 00:25:55 khuck Exp $</P>
 * @author  Kevin Huck
 * @version 2.0
 * @since   2.0 
 */
public class TrialResultTest extends TestCase {

	/**
	 * @param arg0
	 */
	public TrialResultTest(String arg0) {
		super(arg0);
	}

	/**
	 * Test method for {@link glue.TrialResult#TrialResult(edu.uoregon.tau.perfdmf.Trial)}.
	 */
	public final void testTrialResultTrial() {
		PerformanceResult result = null;
		PerformanceResult means = null;
		PerformanceResult totals = null;
		Utilities.setSession("peri_gtc");
		Trial trial = Utilities.getTrial("GTC", "ocracoke-O2", "64");
    	System.out.println("Getting full trial");
		result = new TrialResult(trial);
		System.out.println("Getting means");
		means = new TrialMeanResult(trial, null, null);
		System.out.println("Getting totals");
		totals = new TrialTotalResult(trial, null, null);

		assertNotNull(result);
        assertEquals(result.getMainEvent(), "GTC [{main.F90} {10,9}]");
        assertEquals(means.getMainEvent(), "GTC [{main.F90} {10,9}]");
        assertEquals(totals.getMainEvent(), "GTC [{main.F90} {10,9}]");
//        for (Integer thread : result.getThreads()) {
//            for (String event : result.getEvents()) {
//            	for (String metric : result.getMetrics()) {
        assertEquals(result.getThreads().size(), 64);
        assertEquals(result.getEvents().size(), 42);
        assertEquals(result.getMetrics().size(), 1);
        Integer thread = 1;
        String event = result.getMainEvent();
        String metric = "Time";
	            	System.out.print(thread + " : " + event + " : " + metric + " : inclusive : ");
	            	System.out.println(result.getInclusive(thread, event, metric));
	            	System.out.print(thread + " : " + event + " : " + metric + " : exclusive : ");
	            	System.out.println(result.getExclusive(thread, event, metric));
//            	}
            	System.out.print(thread + " : " + event + " : " + AbstractResult.CALLS + " : ");
            	System.out.println(result.getCalls(thread, event));
            	System.out.print(thread + " : " + event + " : " + AbstractResult.SUBROUTINES + " : ");
            	System.out.println(result.getSubroutines(thread, event));
//            }
//        }
        assertEquals(means.getThreads().size(), 1);
        assertEquals(means.getEvents().size(), 42);
        assertEquals(means.getMetrics().size(), 1);
//        for (String event : means.getEvents()) {
//        	for (String metric : means.getMetrics()) {
            	System.out.print(event + " : " + metric + " : inclusive : ");
            	System.out.println(means.getInclusive(0, event, metric));
            	System.out.print(event + " : " + metric + " : exclusive : ");
            	System.out.println(means.getExclusive(0, event, metric));
//        	}
        	System.out.print(event + " : " + AbstractResult.CALLS + " : ");
        	System.out.println(means.getCalls(0, event));
        	System.out.print(event + " : " + AbstractResult.SUBROUTINES + " : ");
        	System.out.println(means.getSubroutines(0, event));
//        }
        assertEquals(totals.getThreads().size(), 1);
        assertEquals(totals.getEvents().size(), 42);
        assertEquals(totals.getMetrics().size(), 1);
//        for (String event : totals.getEvents()) {
//        	for (String metric : totals.getMetrics()) {
            	System.out.print(event + " : " + metric + " : inclusive : ");
            	System.out.println(totals.getInclusive(0, event, metric));
            	System.out.print(event + " : " + metric + " : exclusive : ");
            	System.out.println(totals.getExclusive(0, event, metric));
//        	}
        	System.out.print(event + " : " + AbstractResult.CALLS + " : ");
        	System.out.println(totals.getCalls(0, event));
        	System.out.print(event + " : " + AbstractResult.SUBROUTINES + " : ");
        	System.out.println(totals.getSubroutines(0, event));
//        }
	}

	public static void main (String[] args) {
		junit.textui.TestRunner.run(TrialResultTest.class);
	}


}
