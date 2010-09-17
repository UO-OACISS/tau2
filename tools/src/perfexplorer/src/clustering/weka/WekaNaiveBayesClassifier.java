/**
 * 
 */
package edu.uoregon.tau.perfexplorer.clustering.weka;

import java.util.ArrayList;
import java.util.List;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import edu.uoregon.tau.perfexplorer.clustering.ClassifierInterface;
import edu.uoregon.tau.perfexplorer.clustering.RawDataInterface;

/**
 * This class is an implementation of the ClassifierInterface, and is also 
 * a base implementation for all classifiers.  Any classifer for the clustering.weka
 * package should extend this class.
 * 
 * @author khuck
 *
 */
public class WekaNaiveBayesClassifier implements ClassifierInterface {
	
	protected Instances trainingData = null;
	protected Classifier classifier = null;
	protected List<double[]> distributions = new ArrayList<double[]>();
	
	/**
	 * 
	 */
	protected WekaNaiveBayesClassifier(RawDataInterface inputData) {
		this.trainingData = (Instances) inputData.getData();
		// the class attribute is the last attribute.
		this.trainingData.setClassIndex(this.trainingData.numAttributes() - 1);
		// this is equivalent?
		this.trainingData.setClass(trainingData.attribute(this.trainingData.numAttributes() - 1));
	}

	/* (non-Javadoc)
	 * @see clustering.ClassifierInterface#classifyInstance(clustering.RawDataInterface)
	 */
	public List<String> classifyInstances(RawDataInterface inputData) {
		List<String> result = new ArrayList<String>();
		Instances tmp = (Instances)inputData.getData();
		tmp.setClassIndex(this.trainingData.numAttributes() - 1);
		try {
			// for each instance passed in...
			for (int i = 0 ; i < tmp.numInstances(); i++) {
				Instance current = tmp.instance(i);
				// ...classify the instance...
				//output = classifier.distributionForInstance(current);
				double output = classifier.classifyInstance(current);
				double[] distribution = classifier.distributionForInstance(current);
				this.distributions.add(distribution);
				// ...the class is the last attribute.
				result.add(Double.toString(output));
			}
		} catch (Exception e) {
			System.err.println("Error performing classification");
			System.err.println(e.getMessage());
			e.printStackTrace(System.err);
			System.exit(0);			
		}
		return result;
	}

	public void buildClassifier() {
		try {
			this.classifier = Classifier.forName("weka.classifiers.bayes.NaiveBayes", null);
			this.classifier.buildClassifier(trainingData);
		} catch (Exception e) {
			System.err.println("Error building classifier");
			System.err.println(e.getMessage());
			e.printStackTrace(System.err);
			System.exit(0);			
		}
	}

	public String evaluate(RawDataInterface testData) {
		Instances tmp = (Instances)testData.getData();
		// evaluate classifier and print some statistics
		Evaluation eval = null;
		try {
			eval = new Evaluation(trainingData);
			eval.evaluateModel(classifier, tmp);
		} catch (Exception e) {
			System.err.println("Error evaluating classifier");
			System.err.println(e.getMessage());
			e.printStackTrace(System.err);
			System.exit(0);			
		}
		return (eval.toSummaryString("\nResults\n=======\n", false));
	}

	public List<double[]> getDistributions() {
		return this.distributions;
	}

	
}
