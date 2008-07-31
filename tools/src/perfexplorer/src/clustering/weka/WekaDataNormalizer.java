/**
 * Created Feb. 13, 2006
 */
package clustering.weka;

import clustering.DataNormalizer;
import clustering.RawDataInterface;

import java.util.List;



/**
  * Implementation of the DataNormalizer interface for Weka data.
  * TODO - make this class immutable?
  *
  * <P>CVS $Id: WekaDataNormalizer.java,v 1.3 2008/07/31 18:43:48 khuck Exp $</P>
  * @author khuck
  * @version 0.2
  * @since   0.2
  */
public class WekaDataNormalizer implements DataNormalizer {
    private RawDataInterface _normalizedData = null;

    private WekaDataNormalizer() {}
    
    /**
     * Constructor restricted to package private.
     *
     * @param inputData
     */
    public WekaDataNormalizer(RawDataInterface inputData) {
        // get the
        int dimensions = inputData.numDimensions();
        int vectors = inputData.numVectors();
        List eventNames = inputData.getEventNames();
        String name = inputData.getName();

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
        _normalizedData =
                new WekaRawData(name, eventNames, vectors, dimensions, null);

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
        return this._normalizedData;
    }
}
