/**
 * 
 */
package glue;

import edu.uoregon.tau.perfdmf.DataSource;
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
	 * Test method for {@link glue.DataSourceResult#DataSourceResult(int, java.lang.String[], boolean)}.
	 */
	public final void testDataSourceResultIntStringArrayBoolean() {
		String home = System.getProperty("user.home");
		String slash = System.getProperty("file.separator");

		String[] files = new String[1];
		files[0] = home + slash + "PERI" + slash + "GTC_s" + slash + "jaguar" + slash + "test" + slash + "0016";
		System.out.println(files[0]);
		PerformanceResult input= new DataSourceResult(DataSource.TAUPROFILE, files, false);
		assertEquals(input.getThreads().size(), 16);
		assertEquals(input.getEvents().size(), 31);
		assertEquals(input.getMetrics().size(), 2);
	}

}
