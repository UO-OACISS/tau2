package server;

import common.AnalysisType;
import common.TransformationType;

/**
 * The facade interface to the application for scripting purpose. This facade
 * allows a limited and easy access from user scripts With this facade user do
 * not have to traverse the object containment hierarchy. Also because the
 * subsystems are not exposed, scripts are limited in what they can do.
 */
public interface ScriptFacade {
	/**
	 * Test method for the facade class.
	 * 
	 */
	public void doSomething();

	/**
	 * Set the focus on the application specified.
	 * 
	 * @param name
	 */
	public void setApplication(String name);

	/**
	 * Set the focus on the experiment specified.
	 * 
	 * @param name
	 */
	public void setExperiment(String name);

	/**
	 * Set the focus on the trial specified.
	 * 
	 * @param name
	 */
	public void setTrial(String name);

	/**
	 * Set the focus on the metric specified.
	 * 
	 * @param name
	 */
	public void setMetric(String name);

	/**
	 * Choose the dimension reduction method.
	 * 
	 * @param type
	 * @param parameter
	 */
	public void setDimensionReduction(TransformationType type, String parameter);

	/**
	 * Set the analysis method.
	 * 
	 * @param type
	 */
	public void setAnalysisType(AnalysisType type);

	/**
	 * Request the analysis configured.
	 * 
	 * @return
	 */
	public String requestAnalysis();

	/**
	 * Request the ANOVA results.
	 * 
	 */
	public void DoANOVA();

	/**
	 * Set the maximum number of clusters for cluster analysis
	 * 
	 * @param max
	 */
	public void SetMaximumNumberOfClusters(int max);

	/**
	 * Request a 3D view of correlation data
	 *
	 */
	public void Do3DCorrelationCube();
}