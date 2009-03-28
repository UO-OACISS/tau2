/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue.test;

import edu.uoregon.tau.perfdmf.DataSource;
import edu.uoregon.tau.perfexplorer.client.PerfExplorerClient;
import edu.uoregon.tau.perfexplorer.common.EngineType;
import edu.uoregon.tau.perfexplorer.glue.BuildMessageHeatMap;
import edu.uoregon.tau.perfexplorer.glue.DataSourceResult;
import edu.uoregon.tau.perfexplorer.glue.PerformanceAnalysisOperation;
import edu.uoregon.tau.perfexplorer.glue.PerformanceResult;
import junit.framework.TestCase;

/**
 * @author khuck
 *
 */
public class DataSourceResultTest extends TestCase {

	/**
	 * @param arg0
	 */
	public DataSourceResultTest(String arg0) {
		super(arg0);
	}

	/**
	 * Test method for {@link edu.uoregon.tau.perfexplorer.glue.DataSourceResult#DataSourceResult(int, java.lang.String[], boolean)}.
	 */
	public final void testDataSourceResultIntStringArrayBoolean() {
        String home = System.getProperty("user.home");
		String slash = System.getProperty("file.separator");

		String[] files = new String[1];
//		files[0] = home + slash + "PERI" + slash + "GTC_s" + slash + "jaguar" + slash + "test" + slash + "0016";
		files[0] = home + slash + "tau2" + slash + "examples" + slash + "NPB2.3" + slash + "bin";
		System.out.println(files[0]);
		PerformanceResult input= new DataSourceResult(DataSource.TAUPROFILE, files, false);
/*		assertEquals(input.getThreads().size(), 16);
		assertEquals(input.getEvents().size(), 31);
		assertEquals(input.getMetrics().size(), 2);
*/		assertEquals(input.getThreads().size(), 4);
		assertEquals(input.getEvents().size(), 32);
		assertEquals(input.getMetrics().size(), 1);
		assertEquals(input.getUserEvents().size(), 17);
		PerformanceAnalysisOperation messageHeatMap = new BuildMessageHeatMap(input);
		messageHeatMap.processData();
		try {
			Thread.sleep(5000);
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
