/**
 * 
 */
package edu.uoregon.tau.perfexplorer.rules;

/**
 * @author khuck
 *
 */
public class FactWrapper {

	private String factName = null;
	private String factType = null;
	private Object factData = null;
	
	/**
	 * 
	 */
	public FactWrapper(String factName, String factType, Object factData) {
		this.factName = factName;
		this.factType = factType;
		this.factData = factData;
	}

	/**
	 * @return the factData
	 */
	public Object getFactData() {
		return factData;
	}

	/**
	 * @param factData the factData to set
	 */
	public void setFactData(Object factData) {
		this.factData = factData;
	}

	/**
	 * @return the factName
	 */
	public String getFactName() {
		return factName;
	}

	/**
	 * @param factName the factName to set
	 */
	public void setFactName(String factName) {
		this.factName = factName;
	}

	/**
	 * @return the factType
	 */
	public String getFactType() {
		return factType;
	}

	/**
	 * @param factType the factType to set
	 */
	public void setFactType(String factType) {
		this.factType = factType;
	}

}
