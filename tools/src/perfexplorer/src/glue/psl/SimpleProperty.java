/**
 * 
 */
package glue.psl;


/**
 * @author khuck
 *
 */
public abstract class SimpleProperty implements Property {

	protected double severity = 0.0;
	protected double confidence = 1.0;
	
	/**
	 * 
	 */
	public SimpleProperty() {
		// TODO Auto-generated constructor stub
	}

	/* (non-Javadoc)
	 * @see glue.Property#getConfidence()
	 */
	public double getConfidence() {
		// TODO Auto-generated method stub
		return confidence;
	}

	/* (non-Javadoc)
	 * @see glue.Property#getSeverity()
	 */
	public double getSeverity() {
		// TODO Auto-generated method stub
		return severity;
	}

	/* (non-Javadoc)
	 * @see glue.Property#holds()
	 */
	public boolean holds() {
		// TODO Auto-generated method stub
		return severity > 0.0;
	}

}
