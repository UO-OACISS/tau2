/**
 * 
 */
package clustering;

/**
 * @author khuck
 *
 */
public interface ClassifierInterface {
	
	/**
	 * Build the classifier from the input data
	 * 
	 */
	void buildClassifier();
	
	/**
	 * Using the constructed classifier, classify the new instance.
	 * 
	 * @param inputData
	 * @return
	 */
	Object classifyInstance(RawDataInterface inputData);

	/**
	 * method for outputting the results of the classifier
	 * @return
	 */
	String toString();
	
	/**
	 * Evaluate the classifier
	 * 
	 * @param testData
	 * @return
	 */
	String evaluate(RawDataInterface testData);
}
