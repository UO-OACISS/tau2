package glue;

import java.util.Hashtable;

import edu.uoregon.tau.perfdmf.Trial;
import junit.framework.TestCase;

public class TrialThreadMetadataTest extends TestCase {

	public void testTrialThreadMetadataTrial() {
		int sessionid = Utilities.setSession("localhost:5432/perfdmf");
		Trial trial = Utilities.getTrial("sweep3d", "jaguar", "16");
		TrialThreadMetadata trialMetadata = new TrialThreadMetadata(trial);
		for (Integer thread : trialMetadata.getThreads()) {
			for (String event : trialMetadata.getEvents()) {
				for (String metric : trialMetadata.getMetrics()) {
					System.out.println (metric + ": " + trialMetadata.getExclusive(thread, event, metric));
				}
			}
		}

	}

}
