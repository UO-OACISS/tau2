/**
 * 
 */
package glue.test;

import edu.uoregon.tau.perfdmf.Trial;
import glue.PerformanceResult;
import glue.ScalabilityOperation;
import glue.ScalabilityResult;
import glue.TrialMeanResult;
import glue.Utilities;
import java.util.List;

import junit.framework.TestCase;

/**
 * @author khuck
 *
 */
public class ScalabilityOperationTest extends TestCase {

	public final void testProcessData() {
		Utilities.setSession("perfdmf_test");
		Trial trial = Utilities.getTrial("Meng-Shiou", "luciferin", "luciferin-1.1");
		PerformanceResult result = new TrialMeanResult(trial);
		ScalabilityOperation scalability = new ScalabilityOperation(result);
		scalability.addInput(new TrialMeanResult(Utilities.getTrial("Meng-Shiou", "luciferin", "luciferin-2.1")));
		scalability.addInput(new TrialMeanResult(Utilities.getTrial("Meng-Shiou", "luciferin", "luciferin-4.1")));
		for (ScalabilityResult.Measure measure : ScalabilityResult.Measure.values()) {
			for (ScalabilityResult.Scaling scaling : ScalabilityResult.Scaling.values()) {
				scalability.reset();
				scalability.setMeasure(measure);
				scalability.setScaling(scaling);
				System.out.println("\nscaling study: " + measure + " " + scaling);
				List<PerformanceResult> outputs = scalability.processData();
				for (PerformanceResult output : outputs) {
					assertNotNull(output);
					for (Integer thread : output.getThreads()) {
						String mainEvent = output.getMainEvent();
						String timeMetric = output.getTimeMetric();
						System.out.print(thread + " " + mainEvent + " " + timeMetric + " " + output.getInclusive(thread, mainEvent, timeMetric) + ", ");
						System.out.print(thread + " " + mainEvent + " " + output.getCalls(thread, mainEvent) + ", ");
						System.out.print(thread + " " + mainEvent + " " + output.getSubroutines(thread, mainEvent) + "\n\n");
		
						for (String event : output.getEvents()) {
							for (String metric : output.getMetrics()) {
								System.out.print(thread + " " + event + " " + metric + " " + output.getInclusive(thread, event, metric) + ", ");
								System.out.print(thread + " " + event + " " + metric + " " + output.getExclusive(thread, event, metric) + ", ");
							}
							System.out.print(thread + " " + event + " " + output.getCalls(thread, event) + ", ");
							System.out.print(thread + " " + event + " " + output.getSubroutines(thread, event) + "\n");
						}
					}
				}
			}
		}

	}
}
