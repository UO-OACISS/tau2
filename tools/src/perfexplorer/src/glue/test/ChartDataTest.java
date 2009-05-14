/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue.test;

import javax.swing.JFrame;

import edu.uoregon.tau.perfexplorer.glue.ChartData;
import junit.framework.TestCase;
import edu.uoregon.tau.perfexplorer.client.*;

/**
 * @author khuck
 *
 */
public class ChartDataTest extends TestCase {

	/**
	 * @param name
	 */
	public ChartDataTest(String name) {
		super(name);
	}

	/**
	 * Test method for {@link edu.uoregon.tau.perfexplorer.glue.ChartData#ChartData()}.
	 */
	public final void testChartData() {
		PerfExplorerClient client = new PerfExplorerClient(true, "/home/khuck/.ParaProf/perfdmf.cfg.peris3d",
			    false);
        client.pack();
        client.setVisible(true);
        client.toFront();

		ChartData data = new ChartData();
		data.addRow("Dilation");
		// series, x value, y value
		data.addColumn(0,0,0);
		data.addColumn(0,2,2);
		data.addColumn(0,4,4);
		data.addColumn(0,0,0);
		data.addColumn(0,2,2);
		data.addColumn(0,4,4);
		data.addColumn(0,0,0);
		data.addColumn(0,2,2);
		data.addColumn(0,4,4);
		data.addColumn(0,0,0);
		data.addColumn(0,2,2);
		data.addColumn(0,4,4);
		data.addColumn(0,4,4);
		data.addColumn(0,4,4);
		PerfExplorerHistogramChart.doHistogram(data,"title");
	}

}
