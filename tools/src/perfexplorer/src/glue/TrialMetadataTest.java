package glue;

import java.util.Hashtable;

import edu.uoregon.tau.perfdmf.Trial;
import junit.framework.TestCase;

public class TrialMetadataTest extends TestCase {

	public final void testTrialMetadata() {
//		int sessionid = Utilities.setSession("perfdmf_test");
//		Trial trial = Utilities.getTrial("GTC_s_PAPI", "VN XT3", "004");
		int sessionid = Utilities.setSession("localhost:5432/perfdmf");
		Trial trial = Utilities.getTrial("sweep3d", "jaguar", "16");
		TrialMetadata trialMetadata = new TrialMetadata(trial.getID());
		Hashtable<String,String> metadata = trialMetadata.getCommonAttributes();
//		assertEquals(123, metadata.keySet().size());
		assertEquals(307, metadata.keySet().size());
		for (String key : metadata.keySet()) {
			System.out.println(key + ": " + metadata.get(key));
		}
	}

}
