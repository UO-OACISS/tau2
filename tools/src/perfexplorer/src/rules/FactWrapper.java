/**
 * This class is used as a simple wrapper to create simple facts.
 * For more complex fact objects, implement the {@link SelfAsserting} interface.
 */
package edu.uoregon.tau.perfexplorer.rules;

/**
 * This class is used as a simple wrapper to create simple facts.
 * For more complex fact objects, implement the {@link SelfAsserting} interface.
 *
 * @author khuck
 * @since 2.0
 * @see SelfAsserting
 */
public class FactWrapper {

	private String factName = null;
	private String factType = null;
	private Object factData = null;
	
	/**
	 * Constructor method.
	 *
	 * @param factName The name of the fact
	 * @param factType The type of the fact
	 * @param factData The object that this fact wrapper will contain
	 */
	public FactWrapper(String factName, String factType, Object factData) {
		this.factName = factName;
		this.factType = factType;
		this.factData = factData;
	}

	/**
	 * Return the object stored in the FactWrapper object.
	 *
	 * @return the factData
	 * @see #setFactData
	 */
	public Object getFactData() {
		return factData;
	}

	/**
	 * Set the object in the FactWrapper.
	 *
	 * @param factData the factData to set
	 * @see #getFactData
	 */
	public void setFactData(Object factData) {
		this.factData = factData;
	}

	/**
	 * Return the name of the fact.
	 *
	 * @return the factName
	 * @see #setFactName
	 */
	public String getFactName() {
		return factName;
	}

	/**
	 * Set the name of the Fact.
	 *
	 * @param factName the factName to set
	 * @see #getFactName
	 */
	public void setFactName(String factName) {
		this.factName = factName;
	}

	/**
	 * Return the type of the fact.
	 *
	 * @return the factType
	 * @see #setFactType
	 */
	public String getFactType() {
		return factType;
	}

	/**
	 * Set the type of the fact.
	 *
	 * @param factType the factType to set
	 * @see #getFactType
	 */
	public void setFactType(String factType) {
		this.factType = factType;
	}

}
