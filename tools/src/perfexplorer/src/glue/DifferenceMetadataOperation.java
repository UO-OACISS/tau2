/**
 * 
 */
package glue;

import java.util.Hashtable;
import java.util.Iterator;
import java.util.Set;

import org.drools.FactHandle;

import rules.FactWrapper;
import rules.RuleHarness;
import rules.SelfAsserting;

import edu.uoregon.tau.perfdmf.Trial;

/**
 * @author khuck
 *
 */
public class DifferenceMetadataOperation implements SelfAsserting {
	TrialMetadata baseline = null;
	TrialMetadata comparison = null;
	Hashtable<String,String[]> differences = new Hashtable<String,String[]>();
	double expectedRatio = 1.0;
	Hashtable<String,FactHandle> assertedFacts = new Hashtable<String,FactHandle>();
	public final static String NAME = glue.DifferenceMetadataOperation.class.getName();
	private RuleHarness ruleHarness = null;
	
	public double getExpectedRatio() {
		return expectedRatio;
	}

	public void setExpectedRatio(double expectedRatio) {
		this.expectedRatio = expectedRatio;
	}

	public DifferenceMetadataOperation (TrialMetadata baseline, TrialMetadata comparison) {
		this.baseline = baseline;
		this.comparison = comparison;
		doComparison();
	}
	
	public DifferenceMetadataOperation (Trial baseline, Trial comparison) {
		this.baseline = new TrialMetadata(baseline);
		this.comparison = new TrialMetadata(comparison);
		doComparison();
	}
	
	private void doComparison() {
		Hashtable<String,String> base = baseline.getCommonAttributes();
		Hashtable<String,String> comp = comparison.getCommonAttributes();
		Set<String> basekeys = base.keySet();
		Set<String> compkeys = comp.keySet();
		
		for (String key : basekeys) {
			String baseValue = base.get(key);
			String compValue = comp.get(key);
			if (compValue == null || !baseValue.equals(compValue)) {
				String[] values = new String[2];
				values[0] = baseValue;
				values[1] = compValue;
				differences.put(key, values);
			}
		}
		
		compkeys.removeAll(basekeys);
		for (String key : compkeys) {
			String compValue = comp.get(key);
			String[] values = new String[2];
			values[0] = "";
			values[1] = compValue;
			differences.put(key, values);
		}
	}

	/**
	 * @return the baseline
	 */
	public TrialMetadata getBaseline() {
		return baseline;
	}

	/**
	 * @param baseline the baseline to set
	 */
	public void setBaseline(TrialMetadata baseline) {
		this.baseline = baseline;
	}

	/**
	 * @return the comparison
	 */
	public TrialMetadata getComparison() {
		return comparison;
	}

	/**
	 * @param comparison the comparison to set
	 */
	public void setComparison(TrialMetadata comparison) {
		this.comparison = comparison;
	}

	/**
	 * @return the differences
	 */
	public Hashtable<String, String[]> getDifferences() {
		return differences;
	}

	/**
	 * @param differences the differences to set
	 */
	public void setDifferences(Hashtable<String, String[]> differences) {
		this.differences = differences;
	}

	public String differencesAsString() {
		StringBuffer buf = new StringBuffer();
		Hashtable diff = getDifferences();
		
		Set keys = diff.keySet();
		
		for (Iterator iter = keys.iterator() ; iter.hasNext() ; ) {
			String key = (String)iter.next();
			if (!key.startsWith("buildenv:") && !key.startsWith("runenv:") && !key.startsWith("build:")) {
			String[] values = (String[])diff.get(key);
			buf.append(key + " " + values[0] + " " + values[1] + "\n");
			}
		}
		return buf.toString();
	}

	public void assertFacts() {
		if (RuleHarness.getInstance() != null);
			assertFacts(RuleHarness.getInstance());
	}
	
	public void assertFacts(RuleHarness ruleHarness) {
		this.ruleHarness = ruleHarness;
		Set keys = differences.keySet();
		for (Iterator iter = keys.iterator() ; iter.hasNext() ; ) {
			String key = (String)iter.next();
			String[] values = (String[])differences.get(key);
			FactHandle handle = RuleHarness.assertObject(new FactWrapper(key, NAME, values));
			assertedFacts.put(key, handle);
		}
	}

	public void removeFact(String factName) {
		FactHandle handle = assertedFacts.get(factName);
		if (handle == null) {
			System.err.println("HANDLE NOT FOUND for " + factName + ", " + NAME);
		} else {
			ruleHarness.retractObject(handle);
		}
	}
	
}
