package edu.uoregon.tau.perfexplorer.glue.test;

import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfexplorer.glue.TrialThreadMetadata;
import edu.uoregon.tau.perfexplorer.glue.Utilities;
import junit.framework.TestCase;

public class TrialThreadMetadataTest extends TestCase {

	public void testTrialThreadMetadataTrial() {
		Utilities.setSession("localhost:5432/perfdmf");
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
