package edu.uoregon.tau.perfexplorer.glue;

import java.awt.Dimension;
import java.awt.Point;
import java.awt.Toolkit;
import java.awt.event.ActionListener;
import java.net.URL;
import java.util.List;
import java.util.StringTokenizer;

import javax.swing.JFrame;

import org.jfree.chart.ChartPanel;

import edu.uoregon.tau.common.Utility;
import edu.uoregon.tau.perfexplorer.client.HeatMap;
import edu.uoregon.tau.perfexplorer.client.PerfExplorerChartJMenuBar;
import edu.uoregon.tau.perfexplorer.client.PerfExplorerClient;

public class BuildMessageHeatMap extends AbstractPerformanceOperation {

	private JFrame window = null;
	public BuildMessageHeatMap(PerformanceResult input) {
		super(input);
	}

	public List<PerformanceResult> processData() {
		
		// iterate over the atomic counters, and get the messages sent to each neighbor
		for (PerformanceResult input : this.inputs) {
			int size = input.getThreads().size();
			double[][] map = new double[size][size];
			double max = 0; 
			outputs.add(new DefaultResult(input, false));
			for (Integer thread : input.getThreads()) {
				for (String event : input.getUserEvents()) {
					if (event.startsWith("Message size sent to node ")) {
						// split the string
						StringTokenizer st = new StringTokenizer(event, "Message size sent to node ");
						if (st.hasMoreTokens()) {
							String receiver = st.nextToken();
							//System.out.println(receiver + " from " + thread + " : " + input.getUsereventMean(thread, event));
							map[thread][Integer.parseInt(receiver)] = input.getUsereventMean(thread, event);
							max = max < input.getUsereventMean(thread, event) ? input.getUsereventMean(thread, event) : max;
						}
					}
				}
			}
			HeatMap hm = new HeatMap(map, size, max, "MessageSizes");
			System.out.println(hm.getImage());
			window = new JFrame("Message Size Heat Map");
	        window.getContentPane().add(hm);
/*			ActionListener listener = this;
			this.setJMenuBar(new PerfExplorerChartJMenuBar(listener));
*/	        URL url = Utility.getResource("tau32x32.gif");
			if (url != null) 
				window.setIconImage(Toolkit.getDefaultToolkit().getImage(url));
			try {
				centerFrame(window);
			} catch (NullPointerException e) {
				// we didn't have a client window open.
			}
			window.pack();
			window.setVisible(true);

		}
		return this.outputs;
	}

	public static void centerFrame(JFrame frame) {
        //Window Stuff.
        int windowWidth = 700;
        int windowHeight = 450;
        
        //Grab paraProfManager position and size.
        Point parentPosition = PerfExplorerClient.getMainFrame().getLocationOnScreen();
        Dimension parentSize = PerfExplorerClient.getMainFrame().getSize();
        int parentWidth = parentSize.width;
        int parentHeight = parentSize.height;
        
        //Set the window to come up in the center of the screen.
        int xPosition = (parentWidth - windowWidth) / 2;
        int yPosition = (parentHeight - windowHeight) / 2;

        xPosition = (int) parentPosition.getX() + xPosition;
        yPosition = (int) parentPosition.getY() + yPosition;

        frame.setLocation(xPosition, yPosition);
        frame.setSize(new java.awt.Dimension(windowWidth, windowHeight));
 	}

}
