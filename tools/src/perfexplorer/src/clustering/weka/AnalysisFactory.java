/**
 * Created on Mar 18, 2005
 *
 */

package edu.uoregon.tau.perfexplorer.clustering.weka;


import java.util.List;

import edu.uoregon.tau.perfexplorer.clustering.ClassifierInterface;
import edu.uoregon.tau.perfexplorer.clustering.ClusterInterface;
import edu.uoregon.tau.perfexplorer.clustering.DBScanClusterInterface;
import edu.uoregon.tau.perfexplorer.clustering.DataNormalizer;
import edu.uoregon.tau.perfexplorer.clustering.HierarchicalCluster;
import edu.uoregon.tau.perfexplorer.clustering.KMeansClusterInterface;
import edu.uoregon.tau.perfexplorer.clustering.LinearRegressionInterface;
import edu.uoregon.tau.perfexplorer.clustering.PrincipalComponentsAnalysisInterface;
import edu.uoregon.tau.perfexplorer.clustering.RawDataInterface;
import edu.uoregon.tau.perfexplorer.clustering.Utilities;
import edu.uoregon.tau.perfexplorer.common.RMICubeData;

/**
  * The top-level analysis factory class.  This class is used to construct
  * a specific type of factory, and to define the interface for the factories.
  * To use this class properly, you need only call the static buildFactory
  * method to construct the desired factory.  All other methods are implemented
  * by the specific factory classes.
  *
  * <P>CVS $Id: AnalysisFactory.java,v 1.3 2009/11/18 17:45:21 khuck Exp $</P>
  * @author  Kevin Huck
  * @version 0.1
  * @since   0.1
  */
public class AnalysisFactory {
    
//    /**
//     * Method to construct the specified factory.
//     * This method is used to construct the specified factory.
//     * 
//     * @param factoryType the type of factory to create
//     * @return an AnalysisFactory 
//     * 
//     */
//    public static AnalysisFactory buildFactory(){
//    		return new AnalysisFactory();
//    };

    /**
     * Method to create a RawDataInterface object.
     * Any extention of the AnalysisFactory class has to implement this method.
     * 
     * @param name The description of the data
     * @param attributes The list of column names
     * @param vectors The number of rows in the data (for initialization)
     * @param dimensions The number of dimensions to be stored 
     *          (for initialization)
     * @param classAttributes TODO
     * @return 
     */
    public static RawDataInterface createRawData(String name, 
        List<String> attributes, int vectors, int dimensions, List<String> classAttributes){
    	return new WekaRawData(name, attributes, vectors, dimensions, classAttributes);
    }
    
    /**
     * Method to create the KMeansClusterInterface.
     * Any extention of the AnalysisFactory class has to implement this method.
     *  
     * @return
     */
    public static KMeansClusterInterface createKMeansEngine(){
    	return new WekaKMeansCluster();
    }
    
    /**
     * Method to create the KMeansClusterInterface.
     * Any extention of the AnalysisFactory class has to implement this method.
     *  
     * @return
     */
    public static LinearRegressionInterface createLinearRegressionEngine(){
    	return new WekaLinearRegression();
    }
    
    /**
     * Method to create a component to perform PCA analysis on the data.
     * 
     * @param cubeData The data which specifies the ordering of the dimensions
     * @return
     */
    public static PrincipalComponentsAnalysisInterface 
        createPCAEngine(RMICubeData cubeData){
    	return new WekaPrincipalComponents(cubeData);
    }
        
    /**
     * Method to create a component to perform PCA analysis on the data.
     * 
     * @param cubeData The data which specifies the ordering of the dimensions
     * @return
     */
    public static PrincipalComponentsAnalysisInterface 
        createPCAEngine(RawDataInterface rawData){
    	return new WekaPrincipalComponents(rawData);
    }
        
    /**
     * Method to create a component to normalize the data.
     * 
     * @param inputData
     * @return
     */
    public static DataNormalizer createDataNormalizer
        (RawDataInterface inputData){
    	return new WekaDataNormalizer(inputData);
    }
    
//    /**
//     * Method for shutting down analysis engines, if necessary.
//     */
//    public void closeFactory(){
//		// do nothing
//		return;
//    }
    
    /**
     * Method for building Naive Bayes classifier
     * 
     * @param inputData
     * @return
     */
    public static ClassifierInterface createNaiveBayesClassifier
    	(RawDataInterface inputData){
    	return new WekaNaiveBayesClassifier(inputData);
    }

    /**
     * Method for building Support Vector Machine classifier
     * 
     * @param inputData
     * @return
     */
    public static ClassifierInterface createSupportVectorClassifier
    	(RawDataInterface inputData){
    	return new WekaSupportVectorClassifier(inputData);
    }


	public static Utilities getUtilities(){
		return new WekaUtilities();
	}

	public static HierarchicalCluster createHierarchicalClusteringEngine() {
		return new JavaHierarchicalCluster();
	}

	public static DBScanClusterInterface createDBScanEngine() {
		return new WekaDBScanCluster();
	}
}
