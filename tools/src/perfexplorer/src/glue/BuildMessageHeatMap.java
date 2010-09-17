package edu.uoregon.tau.perfexplorer.glue;

import java.awt.Dimension;
import java.awt.Point;
import java.awt.Toolkit;
import java.net.URL;
import java.text.DecimalFormat;
import java.util.List;
import java.util.StringTokenizer;

import javax.swing.JFrame;
import javax.swing.JOptionPane;

import edu.uoregon.tau.common.Utility;
import edu.uoregon.tau.perfexplorer.client.PerfExplorerClient;
import edu.uoregon.tau.vis.HeatMapData;
import edu.uoregon.tau.vis.HeatMapWindow;

public class BuildMessageHeatMap extends AbstractPerformanceOperation {

	/**
	 * 
	 */
	private static final long serialVersionUID = -3737895345670546137L;
	private HeatMapData mapData = null;
	private int size = 0;
	private final static String allPaths = "All Paths";
	private static final int COUNT = 0;
	private static final int MAX = 1;
	private static final int MIN = 2;
	private static final int MEAN = 3;
	private static final int STDDEV = 4;
	private static final int VOLUME = 5;

	private JFrame window = null;
	public BuildMessageHeatMap(PerformanceResult input) {
		super(input);
	}

	public List<PerformanceResult> processData() {
		DecimalFormat f = new DecimalFormat("0.0");

		// iterate over the atomic counters, and get the messages sent to each neighbor
		for (PerformanceResult input : this.inputs) {
		    long start = System.currentTimeMillis();
			size = input.getThreads().size();
			outputs.add(new DefaultResult(input, false));
		    boolean foundData = false;
		    mapData = new HeatMapData(size);
			for (Integer thread : input.getThreads()) {
				for (String event : input.getUserEvents(thread)) {
					
					// don't process if this thread doesn't have this event
					if (input.getUsereventNumevents(thread, event) == 0) continue;
					
					if (event.startsWith("Message size sent to node ") && !event.contains("=>")) {
						foundData = true;
						// split the string
						extractData(input, thread, event, event, allPaths);
					} else if (event.startsWith("Message size sent to node ") && event.contains("=>")) {
						foundData = true;
						StringTokenizer st = new StringTokenizer(event, ":");
						String first = st.nextToken().trim();
						String path = st.nextToken().trim();
						// now, split up the path, and handle each node 
						StringTokenizer st2 = new StringTokenizer(path, "=>");
						String tmp = null;
						while (st2.hasMoreTokens()) {
							tmp = st2.nextToken().trim();
							extractData(input, thread, event, first, tmp);
						}
					}
				}
			}
			if (!foundData) {
				JOptionPane.showMessageDialog(PerfExplorerClient.getMainFrame(), "This trial does not have communication matrix data.\nTo collect communication matrix data, set the environment variable TAU_COMM_MATRIX=1 before executing your application.",
						"No Communication Matrix Data", JOptionPane.ERROR_MESSAGE);
				return null;
			}
		    long elapsedTimeMillis = System.currentTimeMillis()-start;
		    float elapsedTimeSec = elapsedTimeMillis/1000F;
		    System.out.println("Total time to extract data: " + f.format(elapsedTimeSec) + " seconds");
		    
		    start = System.currentTimeMillis();
			mapData.massageData();
		    elapsedTimeMillis = System.currentTimeMillis()-start;
		    elapsedTimeSec = elapsedTimeMillis/1000F;
		    System.out.println("Total time to process data: " + elapsedTimeSec + " seconds");

			window = new HeatMapWindow("Message Size Heat Maps", mapData);
	        URL url = Utility.getResource("tau32x32.gif");
			if (url != null) 
				window.setIconImage(Toolkit.getDefaultToolkit().getImage(url));
			window.setVisible(true);
		}
		return this.outputs;
	}

	private void extractData(PerformanceResult input, Integer thread, String event, String first, String path) {
		double numEvents, eventMax, eventMin, eventMean, eventSumSqr, volume = 0;//stdev, 
		double[] empty = {0,0,0,0,0,0};

		StringTokenizer st = new StringTokenizer(first, "Message size sent to node ");
		if (st.hasMoreTokens()) {
			int receiver = Integer.parseInt(st.nextToken());
			double[] pointData = mapData.get(thread, receiver, path);
			if (pointData == null) {
				pointData = empty;
			}

			numEvents = input.getUsereventNumevents(thread, event);
			pointData[COUNT] += numEvents;
			
			eventMax = input.getUsereventMax(thread, event);
			pointData[MAX] = Math.max(eventMax, pointData[MAX]);
			
			eventMin = input.getUsereventMin(thread, event);
			if (pointData[MIN] > 0) {
				pointData[MIN] = Math.min(pointData[MIN],eventMin);
			} else {
				pointData[MIN] = eventMin;
			}
			
			// we'll recompute this later.
			eventMean = input.getUsereventMean(thread, event);
			pointData[MEAN] += eventMean;
			
			eventSumSqr = input.getUsereventSumsqr(thread, event);
			pointData[STDDEV] += eventSumSqr;
			
			volume = numEvents * eventMean;
			pointData[VOLUME] += volume;
			mapData.put(thread, receiver, path, pointData);
		}
	}

	public static void centerFrame(JFrame frame) {
        //Window Stuff.
        int windowWidth = 700;
        int windowHeight = 500;
        
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
 	}

}
