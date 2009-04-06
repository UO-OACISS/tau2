
package edu.uoregon.tau.paraprof;

import java.io.IOException;
import java.util.Iterator;
import java.util.List;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Map;
import java.util.StringTokenizer;
import java.util.TreeMap;

import javax.swing.JFrame;
import javax.swing.JOptionPane;

import java.lang.Math;
import java.text.DecimalFormat;
import java.text.FieldPosition;
import java.net.URL;
import edu.uoregon.tau.common.Utility;
import edu.uoregon.tau.perfdmf.*;
import edu.uoregon.tau.vis.HeatMapWindow;

import java.awt.Toolkit;

public class CommunicationMatrix {

	private Map/*<String, double[][][]>*/ maps = new TreeMap/*<String, double[][][]>*/();
	private Map/*<String, double[]>*/ maxs = new TreeMap/*<String, double[]>*/(); 
	private Map/*<String, double[]>*/ mins = new TreeMap/*<String, double[]>*/(); 
	private int size = 0;
	private final static String allPaths = "All Paths";
	private static final int COUNT = 0;
	private static final int MAX = 1;
	private static final int MIN = 2;
	private static final int MEAN = 3;
	private static final int STDDEV = 4;
	private static final int VOLUME = 5;
	private JFrame window = null;
	private int numEvents = 0;

	public CommunicationMatrix() {
		super();
	}

	public JFrame doCommunicationMatrix(DataSource dataSource, JFrame mainFrame) {
	    boolean foundData = false;
	    int threadID = 0;
	    size = dataSource.getAllThreads().size();
        for (Iterator it = dataSource.getAllThreads().iterator(); it.hasNext();) {
        	edu.uoregon.tau.perfdmf.Thread thread = (edu.uoregon.tau.perfdmf.Thread) it.next();
            for (Iterator it2 = thread.getUserEventProfiles().iterator(); it2.hasNext();) {
            	UserEventProfile uep = (UserEventProfile) it2.next();
                if (uep != null && uep.getNumSamples() > 0) {
					String event = uep.getName();
					if (event.startsWith("Message size sent to node ") && event.indexOf("=>") == -1) {
						// split the string
						extractData(uep, threadID, event, event, allPaths);
					}
					else if (event.startsWith("Message size sent to node ") && event.indexOf("=>") >= 0) {
						foundData = true;
						StringTokenizer st = new StringTokenizer(event, ":");
						String first = st.nextToken().trim();
						String path = st.nextToken().trim();
						// now, split up the path, and handle each node 
						StringTokenizer st2 = new StringTokenizer(path, "=>");
						StringBuffer sb = new StringBuffer();
						String tmp = null;
						while (st2.hasMoreTokens()) {
							if (sb.length() > 0) {
								sb.append(" => ");
							}
							tmp = st2.nextToken().trim();
							sb.append(tmp);
							extractData(uep, threadID, event, first, tmp);
						}
						// do this to get "* => MPI_Isend()" and all equivalents
	//					extractData(input, thread, event, first, "* => "+tmp);
					}
                }
			}
            threadID++;
		}
		if (!foundData) {
			JOptionPane.showMessageDialog(mainFrame, "This trial does not have communication matrix data.\nTo collect communication matrix data, set the environment variable TAU_COMM_MATRIX=1 before executing your application.",
					"No Communication Matrix Data", JOptionPane.ERROR_MESSAGE);
			return null;
		}
		massageData();
		window = new HeatMapWindow("Message Size Heat Maps", maps, maxs, mins, size);
        URL url = Utility.getResource("tau32x32.gif");
		if (url != null) 
			window.setIconImage(Toolkit.getDefaultToolkit().getImage(url));

		return window;
    }

	private void massageData() {
        for (Iterator it = maps.keySet().iterator(); it.hasNext();) {
        	String key = (String)it.next();
			double[][][] map = (double[][][])maps.get(key);
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

	private void extractData(UserEventProfile uep, int thread, String event, String first, String path) {
		double numEvents, eventMax, eventMin, eventMean, eventSumSqr, stdev, volume = 0;
		double[][][] map = new double[6][size][size];

		if (maps.keySet().contains(path)) {
			map = (double[][][])maps.get(path);
		} else {
			maps.put(path, map);
		}

		StringTokenizer st = new StringTokenizer(first, "Message size sent to node ");
		if (st.hasMoreTokens()) {
			int receiver = Integer.parseInt(st.nextToken());

			numEvents = uep.getNumSamples();
			map[COUNT][thread][receiver] += numEvents;
			
			eventMax = uep.getMaxValue();
			map[MAX][thread][receiver] = Math.max(eventMax, map[1][thread][receiver]);
			
			eventMin = uep.getMinValue();
			if (map[MIN][thread][receiver] > 0) {
				map[MIN][thread][receiver] = Math.min(map[MIN][thread][receiver],eventMin);
			} else {
				map[MIN][thread][receiver] = eventMin;
			}
			
			// we'll recompute this later.
			eventMean = uep.getMeanValue();
			map[MEAN][thread][receiver] += eventMean;
			
			eventSumSqr = uep.getStdDev();
			map[STDDEV][thread][receiver] += eventSumSqr;
			
			volume = numEvents * eventMean;
			map[VOLUME][thread][receiver] += volume;
		}
	}
	
	public JFrame getWindow() {
		return window;
	}


}