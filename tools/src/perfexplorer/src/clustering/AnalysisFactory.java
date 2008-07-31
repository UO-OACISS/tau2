/**
 * Created on Mar 18, 2005
 *
 */

package clustering;

import common.EngineType;
import common.RMICubeData;

import java.util.List;

/**
  * The top-level analysis factory class.  This class is used to construct
  * a specific type of factory, and to define the interface for the factories.
  * To use this class properly, you need only call the static buildFactory
  * method to construct the desired factory.  All other methods are implemented
  * by the specific factory classes.
  *
  * <P>CVS $Id: AnalysisFactory.java,v 1.7 2008/07/31 05:34:55 khuck Exp $</P>
  * @author  Kevin Huck
  * @version 0.1
  * @since   0.1
  */
public abstract class AnalysisFactory {

    private static AnalysisFactory theFactory = null;

    /**
     * Protected constructor to prevent instantiating this class
     */
    protected AnalysisFactory() {};
    
    /**
     * Method to construct the specified factory.
     * This method is used to construct the specified factory.
     * 
     * @param factoryType the type of factory to create
     * @return an AnalysisFactory 
     * 
     */
    public static AnalysisFactory buildFactory(EngineType factoryType) 
        throws ClusterException {

        try {
            if (factoryType == EngineType.RPROJECT) {
                return (AnalysisFactory)(Class.forName("clustering.r.RAnalysisFactory").newInstance());
            } else if (factoryType == EngineType.WEKA) {
                return (AnalysisFactory)(Class.forName("clustering.weka.WekaAnalysisFactory").newInstance());
            } else if (factoryType == EngineType.OCTAVE) {
                throw new ClusterException("Unsupported factory type");
            } else {
                throw new ClusterException("Unknown factory type");
            }
        } catch (ClassNotFoundException e) {
            throw new ClusterException("Unknown factory type", e);
        } catch (IllegalAccessException e) {
            throw new ClusterException("Unknown factory type", e);
        } catch (InstantiationException e) {
            throw new ClusterException("Unknown factory type", e);
        }
    };

    /**
     * Method to create a RawDataInterface object.
     * Any extention of the AnalysisFactory class has to implement this method.
     * 
     * @param name The description of the data
     * @param attributes The list of column names
     * @param vectors The number of rows in the data (for initialization)
     * @param dimensions The number of dimensions to be stored 
     *          (for initialization)
     * @return 
     */
    public abstract RawDataInterface createRawData(String name, 
        List attributes, int vectors, int dimensions);
    
    /**
     * Method to create the KMeansClusterInterface.
     * Any extention of the AnalysisFactory class has to implement this method.
     *  
     * @return
     */
    public abstract KMeansClusterInterface createKMeansEngine();
    
    /**
     * Method to create the KMeansClusterInterface.
     * Any extention of the AnalysisFactory class has to implement this method.
     *  
     * @return
     */
    public abstract LinearRegressionInterface createLinearRegressionEngine();
    
    /**
     * Method to create a component to perform PCA analysis on the data.
     * 
     * @param cubeData The data which specifies the ordering of the dimensions
     * @return
     */
    public abstract PrincipalComponentsAnalysisInterface 
        createPCAEngine(RMICubeData cubeData);
        
    /**
     * Method to create a component to perform PCA analysis on the data.
     * 
     * @param cubeData The data which specifies the ordering of the dimensions
     * @return
     */
    public abstract PrincipalComponentsAnalysisInterface 
        createPCAEngine(RawDataInterface rawData);
        
    /**
     * Method to create a component to normalize the data.
     * 
     * @param inputData
     * @return
     */
    public abstract DataNormalizer createDataNormalizer
        (RawDataInterface inputData);
    
    /**
     * Method for shutting down analysis engines, if necessary.
     */
    public abstract void closeFactory();
    
    /**
     * Method for building Naive Bayes classifier
     * 
     * @param inputData
     * @return
     */
    public abstract ClassifierInterface createNaiveBayesClassifier
    	(RawDataInterface inputData);


	public abstract Utilities getUtilities();
}
