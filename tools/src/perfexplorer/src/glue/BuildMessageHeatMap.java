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
						// now, split up the path, and handle each node 
						StringTokenizer st2 = new StringTokenizer(path, "=>");
						StringBuilder sb = new StringBuilder();
						while (st2.hasMoreTokens()) {
							if (sb.length() > 0) {
								sb.append(" => ");
							}
							sb.append(st2.nextToken());
							extractData(input, thread, event, first, sb.toString());
						}
					}
				}
			}
			
			massageData();

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

	private void massageData() {
		for (String key : maps.keySet()) {
			double[][][] map = maps.get(key);
			double[] max = {0,0,0,0,0,0}; 
			double[] min = {0,0,0,0,0,0};
			for (int sender = 0 ; sender < size ; sender++) {
				for (int receiver = 0 ; receiver < size ; receiver++) {
					
					// count and volume are fine... we need to re-compute the mean
					if (map[COUNT][sender][receiver] > 0) {
						map[MEAN][sender][receiver] = map[VOLUME][sender][receiver] / map[COUNT][sender][receiver];
					} else {
						map[MEAN][sender][receiver] = 0;
					}

					// compute stddev
					if (map[COUNT][sender][receiver] > 0)
						map[STDDEV][sender][receiver] = Math.sqrt(Math.abs((map[STDDEV][sender][receiver]/map[COUNT][sender][receiver])-(map[MEAN][sender][receiver]*map[MEAN][sender][receiver])));
					else
						map[STDDEV][sender][receiver] = 0;

					max[COUNT] = Math.max(max[COUNT], map[COUNT][sender][receiver]);
					max[MAX] = Math.max(max[MAX], map[MAX][sender][receiver]);
					max[MIN] = Math.max(max[MIN], map[MIN][sender][receiver]);
					max[MEAN] = Math.max(max[MEAN], map[MEAN][sender][receiver]);
					max[STDDEV] = Math.max(max[STDDEV], map[STDDEV][sender][receiver]);
					max[VOLUME] = Math.max(max[VOLUME], map[VOLUME][sender][receiver]);

					if (map[COUNT][sender][receiver] > 0.0) {
						min[COUNT] = (min[COUNT] == 0.0) ? map[COUNT][sender][receiver] : Math.min(min[COUNT], map[COUNT][sender][receiver]);
						min[MAX] = (min[MAX] == 0.0) ? map[MAX][sender][receiver] : Math.min(min[MAX], map[MAX][sender][receiver]);
						min[MIN] = (min[MIN] == 0.0) ? map[MIN][sender][receiver] : Math.min(min[MIN], map[MIN][sender][receiver]);
						min[MEAN] = (min[MEAN] == 0.0) ? map[MEAN][sender][receiver] : Math.min(min[MEAN], map[MEAN][sender][receiver]);
						min[STDDEV] = (min[STDDEV] == 0.0) ? map[STDDEV][sender][receiver] : Math.min(min[STDDEV], map[STDDEV][sender][receiver]);
						min[VOLUME] = (min[VOLUME] == 0.0) ? map[VOLUME][sender][receiver] : Math.min(min[VOLUME], map[VOLUME][sender][receiver]);
					}
				}
			}
			maps.put(key, map);
			maxs.put(key, max);
			mins.put(key, min);
		}
	}

	private void extractData(PerformanceResult input, Integer thread, String event, String first, String path) {
		double numEvents, eventMax, eventMin, eventMean, eventSumSqr, stdev, volume = 0;
		double[][][] map = new double[6][size][size];
		double[] max = {0,0,0,0,0,0}; 
		double[] min = {0,0,0,0,0,0}; 
		
		if (maps.keySet().contains(path)) {
			map = maps.get(path);
//			max = maxs.get(path);
//			min = mins.get(path);
		} else {
			maps.put(path, map);
//			maxs.put(path, max);
//			mins.put(path, min);
		}

		StringTokenizer st = new StringTokenizer(first, "Message size sent to node ");
		if (st.hasMoreTokens()) {
			int receiver = Integer.parseInt(st.nextToken());

			numEvents = input.getUsereventNumevents(thread, event);
			map[COUNT][thread][receiver] += numEvents;
//			max[0] = max[0] < numEvents ? numEvents : max[0];
//			min[0] = numEvents > 0 && min[0] > numEvents ? numEvents : min[0];
			
			eventMax = input.getUsereventMax(thread, event);
			map[MAX][thread][receiver] = Math.max(eventMax, map[1][thread][receiver]);
//			max[1] = max[1] < eventMax ? eventMax : max[1];
//			min[1] = eventMax > 0 && min[1] > eventMax ? eventMax : min[1];
			
			eventMin = input.getUsereventMin(thread, event);
			if (map[MIN][thread][receiver] > 0) {
				map[MIN][thread][receiver] = Math.min(map[MIN][thread][receiver],eventMin);
			} else {
				map[MIN][thread][receiver] = eventMin;
			}
//			max[2] = max[2] < eventMin ? eventMin : max[2];
//			min[2] = eventMin > 0 && min[2] > eventMin ? eventMin : min[2];
			
			// we'll recompute this later.
			eventMean = input.getUsereventMean(thread, event);
			map[MIN][thread][receiver] += eventMean;
//			max[3] = max[3] < eventMean ? eventMean : max[3];
//			min[3] = eventMean > 0 && min[3] > eventMean ? eventMean : min[3];
			
			eventSumSqr = input.getUsereventSumsqr(thread, event);
//			if (numEvents > 0)
//				stdev = Math.sqrt(Math.abs((eventSumSqr/numEvents)-(eventMean*eventMean)));
//			else
//				stdev = 0;
			map[STDDEV][thread][receiver] += eventSumSqr;
//			max[4] = max[4] < stdev ? stdev : max[4];
//			min[4] = stdev > 0 && min[4] > stdev ? stdev : min[4];
			
			volume = numEvents * eventMean;
			map[VOLUME][thread][receiver] += volume;
//			max[5] = max[5] < volume ? volume : max[5];
//			min[5] = volume > 0 && min[5] > volume ? volume : min[5];
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
