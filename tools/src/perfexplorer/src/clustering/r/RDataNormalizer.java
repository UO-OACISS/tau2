/**
 * Created Feb. 13, 2006
 */
package edu.uoregon.tau.perfexplorer.clustering.r;


import edu.uoregon.tau.perfexplorer.clustering.DataNormalizer;
import edu.uoregon.tau.perfexplorer.clustering.RawDataInterface;

/**
 * Implementation of the DataNormalizer interface for R data.
 * TODO - make this class immutable?
 *
 * <P>CVS $Id: RDataNormalizer.java,v 1.3 2009/02/24 00:53:35 khuck Exp $</P>
 * @author khuck
 * @version 0.2
 * @since   0.2

 */
public class RDataNormalizer implements DataNormalizer {
    private RRawData _normalizedData = null;

    private RDataNormalizer() {}
    
    /**
     * Constructor restricted to package private.
     *
     * @param inputData
     */
    public RDataNormalizer(RawDataInterface inputData) {
        // get the
        int dimensions = inputData.numDimensions();
        int vectors = inputData.numVectors();
        String[] eventNames = (String[])inputData.getEventNames().toArray();

        // calcuate the ranges
        double[][] ranges = new double[dimensions][2];

        for (int i = 0; i < dimensions; i++) {
            ranges[i][0] = inputData.getValue(0, i);
            ranges[i][1] = inputData.getValue(0, i);
            for (int j = 0; j < vectors; j++) {
                // check against the min
                if (ranges[i][0] > inputData.getValue(j, i))
                    ranges[i][0] = inputData.getValue(j, i);
                // check against the max
                if (ranges[i][1] < inputData.getValue(j, i))
                    ranges[i][1] = inputData.getValue(j, i);
            }
            // subtract the min from the max
            ranges[i][1] = ranges[i][1] - ranges[i][0];
        }

        // create the new data
        _normalizedData = new RRawData(vectors, dimensions, eventNames);

        double tmp = 0;
        for (int v = 0; v < vectors; v++) {
            for (int d = 0; d < dimensions; d++) {
                tmp = inputData.getValue(v, d);
                // subtract the min
                tmp = tmp - ranges[d][0];
                // divide by the range
                tmp = tmp / ranges[d][1];
                _normalizedData.addValue(v, d, tmp);
            }
        }

    }

    public RawDataInterface getNormalizedData() {
        return _normalizedData;
    }
}
