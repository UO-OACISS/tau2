/**
 * 
 */
package edu.uoregon.tau.perfexplorer.clustering.weka;

import java.util.List;
import java.util.ArrayList;

import edu.uoregon.tau.perfexplorer.clustering.LinearRegressionInterface;
import edu.uoregon.tau.perfexplorer.clustering.RawDataInterface;

import weka.classifiers.functions.LinearRegression;
import weka.core.Instances;
import weka.core.SelectedTag;

/**
 * @author khuck
 *
 */
public class WekaLinearRegression implements LinearRegressionInterface {

	private LinearRegression regress;
	private Instances instances;
	/**
	 * 
	 */
	public WekaLinearRegression() {
		regress = new LinearRegression();
		regress.setEliminateColinearAttributes(false);
		regress.setAttributeSelectionMethod(new SelectedTag(LinearRegression.SELECTION_NONE, LinearRegression.TAGS_SELECTION));
	}

	/* (non-Javadoc)
	 * @see clustering.LinearRegressionInterface#findCoefficients()
	 */
	public void findCoefficients() {
		try {
			regress.buildClassifier(instances);
//			System.out.println(regress.toString());
		} catch (Exception e) {
			System.err.println("Error performing linear regression");
			System.err.println(e.getMessage());
			e.printStackTrace(System.err);
			System.exit(0);
		}
	}

	/* (non-Javadoc)
	 * @see clustering.LinearRegressionInterface#getCoefficients()
	 */
	public List<Double> getCoefficients() {
		List<Double> coefficients = new ArrayList<Double>();
		double[] params = regress.coefficients();
		for (int i = 0 ; i < params.length ; i++) {
			coefficients.add(new Double(params[i]));
		}
		return coefficients;
	}

	/* (non-Javadoc)
	 * @see clustering.LinearRegressionInterface#setInputData(clustering.RawDataInterface)
	 */
	public void setInputData(RawDataInterface inputData) {
		this.instances = (Instances) inputData.getData();
//		System.out.println(this.instances.toString());
		this.instances.setClassIndex(instances.numAttributes() - 1); 
	}

}
