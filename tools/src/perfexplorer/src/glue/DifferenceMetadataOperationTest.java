package glue;

import java.util.Hashtable;
import java.util.Set;

import edu.uoregon.tau.perfdmf.Trial;
import junit.framework.TestCase;

public class DifferenceMetadataOperationTest extends TestCase {

	public final void testDifferenceMetadataOperation() {
		int sessionid = Utilities.setSession("peri_gtc");
		Trial trial1 = Utilities.getTrial("GTC", "jaguar", "0064");
		TrialMetadata trialMetadata1 = new TrialMetadata(trial1.getID());
		Trial trial2 = Utilities.getTrial("GTC", "jaguar", "0128");
		TrialMetadata trialMetadata2 = new TrialMetadata(trial2.getID());
		DifferenceMetadataOperation oper = new DifferenceMetadataOperation(trialMetadata1, trialMetadata2);
		System.out.println(oper.differencesAsString());
	}
}
