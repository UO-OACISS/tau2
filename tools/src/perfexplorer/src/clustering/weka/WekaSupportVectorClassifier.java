/**
 * 
 */
package edu.uoregon.tau.perfexplorer.clustering.weka;

import weka.classifiers.Classifier;
import edu.uoregon.tau.perfexplorer.clustering.RawDataInterface;

/**
 * @author khuck
 *
 */
public class WekaSupportVectorClassifier extends WekaNaiveBayesClassifier {

	/**
	 * @param inputData
	 */
	public WekaSupportVectorClassifier(RawDataInterface inputData) {
		super(inputData);
		// TODO Auto-generated constructor stub
	}

	public void buildClassifier() {
		try {
			this.classifier = Classifier.forName("weka.classifiers.functions.SMO", null);
			this.classifier.buildClassifier(trainingData);
		} catch (Exception e) {
			System.err.println("Error building classifier");
			System.err.println(e.getMessage());
			e.printStackTrace(System.err);
			System.exit(0);			
		}
	}


}
