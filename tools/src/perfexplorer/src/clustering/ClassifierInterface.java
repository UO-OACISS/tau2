/**
 * 
 */
package clustering;

import java.util.List;

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
	List<String> classifyInstances(RawDataInterface inputData);

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
