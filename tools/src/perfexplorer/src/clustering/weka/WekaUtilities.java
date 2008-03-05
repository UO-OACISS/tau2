
package clustering.weka;

import clustering.Utilities;
import weka.core.Utils;

public class WekaUtilities implements Utilities {

	public double doCorrelation (double[] y1, double[] y2, int arrayLength) {
		return (Utils.correlation(y1, y2, arrayLength));
	}
}