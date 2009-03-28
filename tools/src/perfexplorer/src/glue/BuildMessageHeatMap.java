package edu.uoregon.tau.perfexplorer.glue;

import java.awt.*;

import javax.swing.*;

import java.lang.Math;

import java.net.URL;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.StringTokenizer;
import java.util.TreeMap;

import edu.uoregon.tau.common.Utility;
import edu.uoregon.tau.perfexplorer.client.PerfExplorerClient;
import edu.uoregon.tau.vis.HeatMapWindow;

public class BuildMessageHeatMap extends AbstractPerformanceOperation {

	private Map<String, double[][][]> maps = new TreeMap<String, double[][][]>();
	private Map<String, double[]> maxs = new TreeMap<String, double[]>(); 
	private Map<String, double[]> mins = new TreeMap<String, double[]>(); 
	private int size = 0;
	private final static String allPaths = "All Paths";

	private JFrame window = null;
	public BuildMessageHeatMap(PerformanceResult input) {
		super(input);
	}

	public List<PerformanceResult> processData() {
		
		// iterate over the atomic counters, and get the messages sent to each neighbor
		for (PerformanceResult input : this.inputs) {
			size = input.getThreads().size();
			outputs.add(new DefaultResult(input, false));
			for (Integer thread : input.getThreads()) {
				for (String event : input.getUserEvents()) {
					if (event.startsWith("Message size sent to node ") && !event.contains("=>")) {
						// split the string
						extractData(input, thread, event, event, allPaths);
					}
					else if (event.startsWith("Message size sent to node ") && event.contains("=>")) {
						StringTokenizer st = new StringTokenizer(event, ":");
						String first = st.nextToken().trim();
						String path = st.nextToken().trim();
						extractData(input, thread, event, first, path);
					}
				}
			}

			window = new HeatMapWindow("Message Size Heat Maps", maps, maxs, mins, size);
	        URL url = Utility.getResource("tau32x32.gif");
			if (url != null) 
				window.setIconImage(Toolkit.getDefaultToolkit().getImage(url));
/*			try {
				centerFrame(window);
			} catch (NullPointerException e) {
				// we didn't have a client window open.
			}
*/		}
		return this.outputs;
	}

	private void extractData(PerformanceResult input, Integer thread, String event, String first, String path) {
		double numEvents, eventMax, eventMin, eventMean, eventSumSqr, stdev, volume = 0;
		double[][][] map = new double[6][size][size];
		double[] max = {0,0,0,0,0,0}; 
//		double[] min = {Double.MAX_VALUE,Double.MAX_VALUE,Double.MAX_VALUE,Double.MAX_VALUE,Double.MAX_VALUE}; 
		double[] min = {0,0,0,0,0,0}; 
		
		if (maps.keySet().contains(path)) {
			map = maps.get(path);
			max = maxs.get(path);
			min = mins.get(path);
		} else {
			maps.put(path, map);
			maxs.put(path, max);
			mins.put(path, min);
		}

		StringTokenizer st = new StringTokenizer(first, "Message size sent to node ");
		if (st.hasMoreTokens()) {
			int receiver = Integer.parseInt(st.nextToken());
			
			numEvents = input.getUsereventNumevents(thread, event);
			map[0][thread][receiver] = numEvents;
			max[0] = max[0] < numEvents ? numEvents : max[0];
			min[0] = numEvents > 0 && min[0] > numEvents ? numEvents : min[0];
			
			eventMax = input.getUsereventMax(thread, event);
			map[1][thread][receiver] = eventMax;
			max[1] = max[1] < eventMax ? eventMax : max[1];
			min[1] = eventMax > 0 && min[1] > eventMax ? eventMax : min[1];
			
			eventMin = input.getUsereventMin(thread, event);
			map[2][thread][receiver] = eventMin;
			max[2] = max[2] < eventMin ? eventMin : max[2];
			min[2] = eventMin > 0 && min[2] > eventMin ? eventMin : min[2];
			
			eventMean = input.getUsereventMean(thread, event);
			map[3][thread][receiver] = eventMean;
			max[3] = max[3] < eventMean ? eventMean : max[3];
			min[3] = eventMean > 0 && min[3] > eventMean ? eventMean : min[3];
			
			eventSumSqr = input.getUsereventSumsqr(thread, event);
			if (numEvents > 0)
				stdev = Math.sqrt(Math.abs((eventSumSqr/numEvents)-(eventMean*eventMean)));
			else
				stdev = 0;
			map[4][thread][receiver] = stdev;
			max[4] = max[4] < stdev ? stdev : max[4];
			min[4] = stdev > 0 && min[4] > stdev ? stdev : min[4];
			
			volume = numEvents * eventMean;
			map[5][thread][receiver] = volume;
			max[5] = max[5] < volume ? volume : max[5];
			min[5] = volume > 0 && min[5] > volume ? volume : min[5];
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
//        frame.setSize(new java.awt.Dimension(windowWidth, windowHeight));
 	}

}
