/**
 * Created on Mar 16, 2005
 *
 */
package edu.uoregon.tau.perfexplorer.clustering.r;



import org.omegahat.R.Java.REvaluator;

import edu.uoregon.tau.perfexplorer.clustering.ClusterException;
import edu.uoregon.tau.perfexplorer.clustering.DimensionReductionInterface;
import edu.uoregon.tau.perfexplorer.clustering.RawDataInterface;
import edu.uoregon.tau.perfexplorer.common.PerfExplorerOutput;
import edu.uoregon.tau.perfexplorer.common.RMIPerfExplorerModel;
import edu.uoregon.tau.perfexplorer.common.TransformationType;


/**
 * This class is the R implementation of the dimension reduction operation.
 * This class is package private - it should only be accessed from the
 * clustering class.  To access these methods, create an AnalysisFactory,
 * and the factory will be able to create a Dimension Reduciton object.
 *
 * <P>CVS $Id: RDimensionReduction.java,v 1.6 2009/02/24 00:53:35 khuck Exp $</P>
 * @author khuck
 * @version 0.1
 * @since   0.1
 */
public class RDimensionReduction implements DimensionReductionInterface {

    private RawDataInterface inputData = null;
    private REvaluator rEvaluator = null;
    private TransformationType method = TransformationType.NONE;
    private int newDimension = 0;

    /**
     * Private default constructor.
     */
    private RDimensionReduction() {}
    
    /**
     * The constructor - restricted to package private.
     * @param method
     * @param newDimension
     */
    RDimensionReduction(TransformationType method, int newDimension) {
        super();
        this.rEvaluator = RSingletons.getREvaluator();
        this.method = method;
        this.newDimension = newDimension;
    }
    
    public void reduce() throws ClusterException {
        if (method.equals(TransformationType.LINEAR_PROJECTION)) {
            PerfExplorerOutput.print("Reducing Dimensions...");
            int numReduced = inputData.numVectors() * newDimension;
            rEvaluator.voidEval("reducer <- matrix((runif(" + 
                numReduced + ",0,1)), nrow=" + inputData.numDimensions() + 
                ", ncol="+newDimension+")");
            rEvaluator.voidEval("raw <- crossprod(t(raw), reducer)");
            PerfExplorerOutput.println(" Done!");
        }
        return;
    }
    
    public void setInputData(RawDataInterface inputData) {
        this.inputData = inputData;
        return ;
    }
    
    public RawDataInterface getOutputData () {
        // do nothing - the data stays in R
        RawDataInterface data = null;
        return data;
    }
}
