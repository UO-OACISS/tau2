package edu.uoregon.tau.perfexplorer.glue;

import java.awt.Dimension;
import java.awt.Point;
import java.awt.Toolkit;
import java.awt.event.ActionListener;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import javax.swing.JPanel;

import java.net.URL;
import java.util.List;
import java.util.StringTokenizer;

import javax.swing.JFrame;
import javax.swing.JLabel;

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
			double[][][] map = new double[5][size][size];
			double[] max = {0,0,0,0,0}; 
			double numEvents, eventMax, eventMin, eventMean, eventSumSqr = 0;
			outputs.add(new DefaultResult(input, false));
			for (Integer thread : input.getThreads()) {
				for (String event : input.getUserEvents()) {
					if (event.startsWith("Message size sent to node ")) {
						// split the string
						StringTokenizer st = new StringTokenizer(event, "Message size sent to node ");
						if (st.hasMoreTokens()) {
							String receiver = st.nextToken();
							numEvents = input.getUsereventNumevents(thread, event);
							map[0][thread][Integer.parseInt(receiver)] = numEvents;
							max[0] = max[0] < numEvents ? numEvents : max[0];
							eventMax = input.getUsereventMax(thread, event);
							map[1][thread][Integer.parseInt(receiver)] = eventMax;
							max[1] = max[1] < eventMax ? eventMax : max[1];
							eventMin = input.getUsereventMin(thread, event);
							map[2][thread][Integer.parseInt(receiver)] = eventMin;
							max[2] = max[2] < eventMin ? eventMin : max[2];
							eventMean = input.getUsereventMean(thread, event);
							map[3][thread][Integer.parseInt(receiver)] = eventMean;
							max[3] = max[3] < eventMean ? eventMean : max[3];
							eventSumSqr = input.getUsereventSumsqr(thread, event);
							map[4][thread][Integer.parseInt(receiver)] = eventSumSqr;
							max[4] = max[4] < eventSumSqr ? eventSumSqr : max[4];
						}
					}
				}
			}
			window = new JFrame("Message Size Heat Maps");
			JPanel panel = new JPanel(new GridBagLayout());
			window.getContentPane().add(panel);
			GridBagConstraints c = new GridBagConstraints();
			c.fill = GridBagConstraints.BOTH;
			c.anchor = GridBagConstraints.CENTER;
			c.weightx = 0.5;
			c.insets = new Insets(2,2,2,2);

			c.gridx = 0;
			c.gridy = 0;
			panel.add(new JLabel("Number of Calls", JLabel.CENTER),c);
			c.gridx = 1;
			panel.add(new JLabel("Max Message Size", JLabel.CENTER),c);
			c.gridx = 2;
			panel.add(new JLabel("Min Message Size", JLabel.CENTER),c);

			c.gridx = 0;
			c.gridy = 1;
	        panel.add(new HeatMap(map[0], size, max[0], "NumEvents"), c);
			c.gridx = 1;
	        panel.add(new HeatMap(map[1], size, max[1], "MaxMessageSize"), c);
			c.gridx = 2;
	        panel.add(new HeatMap(map[2], size, max[2], "MinMessageSize"), c);

			c.gridx = 0;
			c.gridy = 2;
			panel.add(new JLabel("Mean Message Size", JLabel.CENTER),c);
			c.gridx = 1;
			panel.add(new JLabel("Message Size Sum Squared", JLabel.CENTER),c);
			c.gridx = 2;
			panel.add(new JLabel("Display Options", JLabel.CENTER),c);

			c.gridx = 0;
			c.gridy = 3;
	        panel.add(new HeatMap(map[3], size, max[3], "MeanMessageSize"), c);
			c.gridx = 1;
	        panel.add(new HeatMap(map[4], size, max[4], "MessageSizeSumSqr"), c);

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
        int windowWidth = 1200;
        int windowHeight = 800;
        
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
