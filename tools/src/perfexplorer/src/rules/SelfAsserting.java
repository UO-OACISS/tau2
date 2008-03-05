/**
 * 
 */
package rules;

/**
 * @author khuck
 *
 */
public interface SelfAsserting {
	/**
	 * Method for asserting facts in a working memory maintained by the rule harness.
	 * 
	 */
	public void assertFacts();
	
	/**
	 * Method for asserting facts in a working memory maintained by the rule harness.
	 * 
	 * @param ruleHarness
	 */
	public void assertFacts(RuleHarness ruleHarness);
	
	/**
	 * Method for removing facts once they have been processed
	 * @param factName
	 */
	public void removeFact(String factName);
}
