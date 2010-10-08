
package edu.uoregon.tau.perfexplorer.client;

import java.awt.Toolkit;
import java.net.URL;
import java.text.DecimalFormat;
import java.util.Map;
import java.util.StringTokenizer;

import javax.swing.JFrame;
import javax.swing.JOptionPane;

import edu.uoregon.tau.common.Utility;
import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.vis.HeatMapData;
import edu.uoregon.tau.vis.HeatMapWindow;

public class CommunicationMatrix {

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
	//private int numEvents = 0;

	public CommunicationMatrix() {
		super();
	}

	public JFrame doCommunicationMatrix() {
		CommunicationMatrix matrix = new CommunicationMatrix();

		// get the selected trial
		PerfExplorerModel model = PerfExplorerModel.getModel();
		
		// get the DB
		Map<String, double[][]> userEvents = PerfExplorerConnection.getConnection().getUserEventData(model);
		
		// this has been validated at the client
		Trial trial = (Trial)model.getCurrentSelection();
		// compute the total number of threads
		int threadsPerContext = Integer.parseInt(trial.getField("threads_per_context"));
		int threadsPerNode = Integer.parseInt(trial.getField("contexts_per_node")) * threadsPerContext;
		size = threadsPerNode * Integer.parseInt(trial.getField("node_count"));

		// start a timer
	    long start = System.currentTimeMillis();
	    
	    // get the list of user events
	    //numEvents = userEvents.keySet().size();
	    
	    // if data is not found, terminate early and issue a warning
	    boolean foundData = false;
	    
	    // declare the heatmap data object
	    mapData = new HeatMapData(size);

	    // iterate over the user events and threads
		for (String event : userEvents.keySet()) {
			double[][] data = userEvents.get(event);
			for (int thread = 0 ; thread < size ; thread++) {
				// don't process if this thread doesn't have this event
				if (data[thread][0] == 0) continue;
				
				// find events with communication matrix data - handle flat profile events
				if (event.startsWith("Message size sent to node ") && !event.contains("=>")) {
					foundData = true;
					// split the string
					extractData(data, thread, event, event, allPaths);
				}
				// find events with communication matrix data - handle context profile events
				else if (event.startsWith("Message size sent to node ") && event.contains("=>")) {
					foundData = true;
					// split the string of callpath function names from the user event name
					StringTokenizer st = new StringTokenizer(event, ":");
					// first is the name of the user event
					String first = st.nextToken().trim();
					// path is the callpath
					String path = st.nextToken().trim();
					// now, split up the path, and handle each node 
					StringTokenizer st2 = new StringTokenizer(path, "=>");
					String tmp = null;
					while (st2.hasMoreTokens()) {
						tmp = st2.nextToken().trim();
						// get the user event data for this node in the callpath
						extractData(data, thread, event, first, tmp);
					}
				}
			}
		}
		
		// if no communication data found, give the user an error message.
		if (!foundData) {
			JOptionPane.showMessageDialog(PerfExplorerClient.getMainFrame(), "This trial does not have communication matrix data.\nTo collect communication matrix data, set the environment variable TAU_COMM_MATRIX=1 before executing your application.",
					"No Communication Matrix Data", JOptionPane.ERROR_MESSAGE);
			return null;
		}

		// recompute the mean, standard deviation and min and max
		mapData.massageData();
		
		// output some timing information
	    float elapsedTimeMillis = System.currentTimeMillis()-start;
	    float elapsedTimeSec = elapsedTimeMillis/1000F;
		DecimalFormat f = new DecimalFormat("0.000");
	    System.out.println("Total time to process data: " + f.format(elapsedTimeSec) + " seconds");

		window = new HeatMapWindow("Message Size Heat Maps", mapData);
        URL url = Utility.getResource("tau32x32.gif");
		if (url != null) 
			window.setIconImage(Toolkit.getDefaultToolkit().getImage(url));

		window.setVisible(true);
		return matrix.getWindow();
    }

	private void extractData(double[][] data, Integer thread, String event, String first, String path) {
		double numEvents, eventMax, eventMin, eventMean, eventSumSqr, volume = 0;
		double[] empty = {0,0,0,0,0,0};

		StringTokenizer st = new StringTokenizer(first, "Message size sent to node ");
		if (st.hasMoreTokens()) {
			int receiver = Integer.parseInt(st.nextToken());

			double[] pointData = mapData.get(thread, receiver, path);
			if (pointData == null) {
				pointData = empty;
			}

			numEvents = data[thread][COUNT];
			pointData[COUNT] += numEvents;
			
			eventMax = data[thread][MAX];
			pointData[MAX] = Math.max(eventMax, pointData[MAX]);
			
			eventMin = data[thread][MIN];
			if (pointData[MIN] > 0) {
				pointData[MIN] = Math.min(pointData[MIN],eventMin);
			} else {
				pointData[MIN] = eventMin;
			}
			
			// we'll recompute this later.
			eventMean = data[thread][MEAN];
			pointData[MEAN] += eventMean;
			
			// we'll recompute this later.
			eventSumSqr = data[thread][STDDEV];
			pointData[STDDEV] += eventSumSqr;
			
			volume = numEvents * eventMean;
			pointData[VOLUME] += volume;
			mapData.put(thread, receiver, path, pointData);
		}
	}
	
	public JFrame getWindow() {
		return window;
	}


}