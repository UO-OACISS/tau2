/**
 * 
 */
package clustering.weka;

import java.util.ArrayList;
import java.util.List;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instance;
import weka.core.Instances;
import clustering.ClassifierInterface;
import clustering.RawDataInterface;

/**
 * @author khuck
 *
 */
public class WekaNaiveBayesClassifier implements ClassifierInterface {
	
	private Instances trainingData = null;
	private NaiveBayes classifier = null;
	private double[] output = null;
	
	/**
	 * 
	 */
	protected WekaNaiveBayesClassifier(RawDataInterface inputData) {
		this.classifier = new NaiveBayes();
		this.trainingData = (Instances) inputData.getData();
		// the class attribute is the last attribute.
		this.trainingData.setClassIndex(this.trainingData.numAttributes() - 1);

	}

	/* (non-Javadoc)
	 * @see clustering.ClassifierInterface#classifyInstance(clustering.RawDataInterface)
	 */
	public Object classifyInstance(RawDataInterface inputData) {
		Object result = null;
		Instances tmp = (Instances)inputData.getData();
		try {
			output = classifier.distributionForInstance(tmp.firstInstance());
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
			classifier.buildClassifier(trainingData);
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

	
}
