package edu.uoregon.tau.paraprof;

import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;
import java.util.Hashtable;
import java.util.Enumeration;

import org.apache.batik.ext.swing.GridBagConstants;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.renderer.xy.XYBarRenderer;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.data.xy.XYSeries;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.statistics.SimpleHistogramBin;
import org.jfree.data.statistics.SimpleHistogramDataset;

import edu.uoregon.tau.common.Utility;
import edu.uoregon.tau.perfdmf.FunctionProfile;
import edu.uoregon.tau.perfdmf.DataSource;
import edu.uoregon.tau.perfdmf.DataSourceException;
import edu.uoregon.tau.perfdmf.Function;
import edu.uoregon.tau.perfdmf.Thread;
import edu.uoregon.tau.perfdmf.Snapshot;

public class MonSummaryWindow extends JFrame {

    private ParaProfTrial ppTrial;
    private DataSource dataSource;
    private Function function;
    private Thread thread;

    private JSlider slider;
    private JLabel indexLabel = new JLabel("");
    //    private JLabel timeLabel = new JLabel("");

    private JCheckBox animateCheckbox = new JCheckBox("Animate");

    private ChartPanel chartPanel;

    private double lastTime;
    private int numBins;
    private int numHistogramsPerEvent;

    private int numDataPoints;
    private int currentVisibleDataPoints;
    private int selectedDataPoint = -1;

    private Animator animationThread;
    private long animationInterval = 1000; // time in milliseconds

    String canonicalName;
    String trialPath;
    private MonProbe probeThread;
    private long probeInterval = 5000; // time in milliseconds

    private Hashtable functionsToSeries;
    private XYSeriesCollection dataset;
    private Thread snapshotThread; // there can be only 1
    private int seriesCount;
    private JFreeChart chart;
    
    // *CWL* Should be abstracted out. Copied from SnapshotControlWindow.java.
    private class Animator extends java.lang.Thread {

        private volatile boolean stop = false;

        Runnable runner = new Runnable() {
        	public void run() {
        		try {
        			if (slider.getValue() >= slider.getMaximum()) {
        				slider.setValue(0);
        				long time = System.currentTimeMillis();
        				lastTime = time;
        			} else {
        				slider.setValue(slider.getValue() + 1);
        			}
        			// System.out.println("Animation tick");
        		} catch (Exception e) {
        			// *CWL* need to throw something suitable for
        			// failure to sleep.
        		}
        	}
        };

	    public void run() {
	    	stop = false;
	    	while (!stop) {
	    		try {
	    			SwingUtilities.invokeLater(runner);	
	    			java.lang.Thread.sleep(animationInterval);
	    		} catch (Exception e) {
	    			// Who cares if we were interrupted
	    		}
	    	}
	    }

        public void end() {
            stop = true;
        }

    }

    private class MonProbe extends java.lang.Thread {

	private boolean allDone = false;

	Runnable runner = new Runnable() {
		public void run() {
		    try {
			// Look for new available data and add them
		    	findAndLoadNewData();
		    } catch (Exception e) {
			// *CWL* need to throw something suitable for
			// failure to sleep.
		    }
		}
	    };

        public void run() {
        	try {
        		while (true) {
        			if (allDone) {
        				return;
        			}
        			SwingUtilities.invokeLater(runner);
        			java.lang.Thread.sleep(probeInterval);
        		}
        	} catch (Exception e) {
        		// Who cares if we were interrupted
        	}
        }

	public void end() {
	    allDone = true;
	}

    }

    // *CWL* Should this be a module?
    public static class ScrollFlowLayout extends FlowLayout {
        public final static int LEFT_TO_RIGHT = 0;
        public final static int RIGHT_TO_LEFT = 1;
        public final static int TOP_TO_BOTTOM = 2;

        private int orientation = LEFT_TO_RIGHT, rows, cols;
        private ScrollPane sp = null;

        public ScrollFlowLayout() {
            super();
        }

        public ScrollFlowLayout(ScrollPane p) {
            super();
            this.sp = p;
        }

        public ScrollFlowLayout(ScrollPane p, int orientation) {
            super();
            this.sp = p;
            this.orientation = orientation;
        }

        public int getTallestElement(Container c) {
            int ncomponents = c.getComponentCount();
            int h = 0;
            for (int i = 0; i < ncomponents; i++) {
                Component m = c.getComponent(i);
                int x = m.getMinimumSize().height;
                h = (h > x) ? h : x;
            }
            return h;
        }

        public int getWidestElement(Container c) {
            int ncomponents = c.getComponentCount();
            int w = 0;
            for (int i = 0; i < ncomponents; i++) {
                Component m = c.getComponent(i);
                int x = m.getMinimumSize().width;
                w = (w > x) ? w : x;
            }
            return w;
        }

        public Dimension preferredLayoutSize(Container target) {
            int hBounds = target.getSize().height - target.getInsets().bottom - target.getInsets().top;

            int wBounds = target.getSize().width;
            if (sp != null) {
                hBounds = sp.getSize().height - sp.getInsets().bottom - sp.getInsets().top;

                wBounds = sp.getSize().width;
            }
            int ncomponents = target.getComponentCount();
            if (ncomponents == 0)
                return new Dimension(0, 0);
            int widest = getWidestElement(target);
            int tallest = getTallestElement(target);

            int rows = (tallest > hBounds) ? 1 : hBounds / tallest;
            int cols = (int) Math.ceil((double) ncomponents / rows);
            if (orientation == LEFT_TO_RIGHT) {
                cols = (widest > wBounds) ? 1 : wBounds / widest;
                rows = (int) Math.ceil((double) ncomponents / cols);
            }

            Dimension d = new Dimension(cols * widest, rows * tallest);
            return d;
        }

        public void layoutContainer(Container target) {
            synchronized (target.getTreeLock()) {
                int hBounds = target.getSize().height - target.getInsets().bottom - target.getInsets().top;
                int wBounds = target.getSize().width;
                if (sp != null) {
                    hBounds = sp.getSize().height - sp.getInsets().bottom - sp.getInsets().top;

                    wBounds = sp.getSize().width;
                }
                int ncomponents = target.getComponentCount();
                int widest = getWidestElement(target);
                int tallest = getTallestElement(target);
                if (ncomponents == 0) {
                    return;
                }
                target.invalidate();
                if (orientation == LEFT_TO_RIGHT) {
                    int i = 0;
                    int cols = (widest > wBounds) ? 1 : wBounds / widest;
                    int rows = (int) Math.ceil((double) ncomponents / cols);

                    for (int x = 0; x < rows; x++) {
                        for (int y = 0; y < cols; y++) {
                            if (i < ncomponents) {
                                int px = y * widest;
                                int py = x * tallest;
                                target.getComponent(i).setBounds(px, py, widest, tallest);
                            }
                            i++;
                        }
                    }
                } else if (orientation == TOP_TO_BOTTOM) {
                    int rows = (tallest > hBounds) ? 1 : hBounds / tallest;
                    int cols = (int) Math.ceil((double) ncomponents / rows);
                    int i = 0;
                    for (int y = 0; y < cols; y++) {
                        for (int x = 0; x < rows; x++) {
                            if (i < ncomponents) {
                                int px = y * widest;
                                int py = x * tallest;
                                target.getComponent(i).setBounds(px, py, widest, tallest);
                            }
                            i++;
                        }
                    }
                }
            }
        }
    }
    
    private static class Histogram {
        public String name;
        public int bins[];
        public double minValue, maxValue;
    }

    private List<Histogram> histograms = new ArrayList<Histogram>();

    public static MonSummaryWindow createMonSummaryWindow(ParaProfTrial ppTrial, 
					    Component invoker) {
        try {
            MonSummaryWindow monw = new MonSummaryWindow(ppTrial, 
							 invoker);
            return monw;
        } catch (Exception e) {
            throw new DataSourceException(e);
        }
    }

    private void processData() throws FileNotFoundException, IOException {
        int invocationIndex = thread.getNodeID();
        String histogramFileName = "tau.histograms." + invocationIndex;

        FileInputStream fis = new FileInputStream(new File(histogramFileName));
        InputStreamReader inReader = new InputStreamReader(fis);
        BufferedReader br = new BufferedReader(inReader);

        int numEvents = Integer.parseInt(br.readLine());
        numHistogramsPerEvent = Integer.parseInt(br.readLine());
        numBins = Integer.parseInt(br.readLine());

        for (int i = 0; i < numHistogramsPerEvent; i++) {
            Histogram histogram = new Histogram();
            histogram.name = br.readLine();
            histogram.bins = new int[numBins];
            histograms.add(histogram);
        }

        boolean found = false;
        for (int e = 0; e < numEvents; e++) {
            String eventName = br.readLine().trim();
            if (eventName.equals(function.getName())) {
                found = true;
                for (int i = 0; i < numHistogramsPerEvent; i++) {
                    StringTokenizer tokenizer = 
			new StringTokenizer(br.readLine(), " \t\n\r");
                    histograms.get(i).minValue = 
			Double.parseDouble(tokenizer.nextToken());
                    histograms.get(i).maxValue = 
			Double.parseDouble(tokenizer.nextToken());
                    for (int b = 0; b < numBins; b++) {
                        histograms.get(i).bins[b] = 
			    Integer.parseInt(tokenizer.nextToken());
                    }
                }
            } else {
                // skip
                for (int i = 0; i < numHistogramsPerEvent; i++) {
                    br.readLine();
                }
            }
        }
        if (!found) {
            System.err.println("Warning, Function \"" + function.getName() + 
			       "\" not found in histogram");
        }

    }

    private MonSummaryWindow(ParaProfTrial ppTrial, 
			     Component invoker) 
	throws Exception {
        this.ppTrial = ppTrial;
        dataSource = ppTrial.getDataSource();
        functionsToSeries = new Hashtable();
        
    	String reverseName = ppTrial.getPathReverse();
    	String pathName = ppTrial.getPath();
    	int index = reverseName.indexOf(".");
    	canonicalName = reverseName.substring(0, index);

    	// This is a short-term quick-and-dirty solution to a
    	//   proper solution (which may require flow-control)
    	int pathIndex = pathName.lastIndexOf(File.separator);
    	if (pathIndex != -1)  {
    		trialPath = pathName.substring(0, pathIndex);
    	} else {
    		trialPath = ".";
    	}
	    
        setTitle("TAU: ParaProf: Monitoring: " + 
        		ppTrial.getTrialIdentifier(ParaProf.preferences.getShowPathTitleInReverse()));
        setLocation(WindowPlacer.getNewLocation(this, invoker));
        setSize(new Dimension(300, 180));

        addWindowListener(new WindowAdapter() {
        	public void windowClosing(WindowEvent e) {
        		if (probeThread != null) {
        			probeThread.end();
        		}
        		if (animationThread != null) {
        			animationThread.end();
        		}
        	}
        });

        // *CWL* Using this until we can avoid using the hack of
        //    representing dumps as nodes in paraprof.
        snapshotThread = dataSource.getThread(0,0,0);
        numDataPoints = snapshotThread.getNumSnapshots();
        selectedDataPoint = 0;

        // Do only a partial load of the existing data. We will load the
        //   others over time or as and when they become available through
        //   a second timing thread which probes for new data.
        if (numDataPoints < 10) {
        	currentVisibleDataPoints = numDataPoints;
        } else {
        	currentVisibleDataPoints = 10;
        }
        slider = new JSlider(0, currentVisibleDataPoints-1);
        slider.setSnapToTicks(true);
        slider.setPaintTicks(true);
        slider.setValue(slider.getMaximum());
        slider.setBackground(Color.white);

        slider.addChangeListener(new ChangeListener() {
		public void stateChanged(ChangeEvent e) {
		    selectedDataPoint = slider.getValue();
		    setLabels();
		    // Update the XY line chart.
		    updateDataset();
		}
	    });

        animateCheckbox.setBackground(Color.white);

        animateCheckbox.addActionListener(new ActionListener() {
		public void actionPerformed(ActionEvent evt) {
		    try {
			if (animateCheckbox.isSelected()) {
			    animationThread = new Animator();
			    animationThread.start();
			} else {
			    animationThread.end();
			    animationThread = null;
			}

		    } catch (Exception e) {
			ParaProfUtils.handleException(e);
		    }
		}
	    });
        setLabels();
        
        dataset = new XYSeriesCollection();
        //updateDataset();
        chart = ChartFactory.createXYLineChart("TauMon: " + canonicalName, 
        					"Dump ID", "Mean Exclusive Time", dataset,
        					PlotOrientation.VERTICAL, 
        					true, // legend
        					true, // tooltips
        					false); // urls

//      ((XYLineAndShapeRenderer) chart.getXYPlot().getRenderer()).setMargin(0.10);
        ((XYLineAndShapeRenderer) chart.getXYPlot().getRenderer()).setOutlinePaint(Color.black);

        Utility.applyDefaultChartTheme(chart);        
        
        JPanel chartPanel  = new ChartPanel(chart);
        System.out.println("size = " + chartPanel.getPreferredSize());
        chartPanel.setMinimumSize(chartPanel.getPreferredSize());
        
        
        JPanel panel = new JPanel();

        panel.setBackground(Color.white);
        panel.setLayout(new GridBagLayout());

        
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.fill = GridBagConstants.BOTH;
        Utility.addCompItem(panel, chartPanel, gbc, 0, 0, 2, 1);
        gbc.fill = GridBagConstants.HORIZONTAL;
        Utility.addCompItem(panel, indexLabel, gbc, 0, 1, 2, 1);
        gbc.fill = GridBagConstants.HORIZONTAL;
        Utility.addCompItem(panel, slider, gbc, 0, 2, 2, 1);
        gbc.fill = GridBagConstants.NONE;
        //        Utility.addCompItem(panel, timeLabel, gbc, 0, 2, 2, 1);
        //        gbc.anchor = GridBagConstants.SOUTH;
        Utility.addCompItem(panel, animateCheckbox, gbc, 0, 3, 1, 1);
        getContentPane().add(panel);
        pack();
        
        ParaProfUtils.setFrameIcon(this);

        // Start the probe thread
        probeThread = new MonProbe();
        probeThread.start();

    }

    private void setLabels() {
        indexLabel.setText("Monitoring Dump " + selectedDataPoint +
			   " of " + (currentVisibleDataPoints-1));

	// *CWL* Again, based on the assumption that each dump
	//    becomes a node in paraprof. Will need to change.
	// NOTE: No meaningful value at this point. Might be a good
	//    idea to put a proper value to it at record time.
	/*
	long startTime = 
	    dataSource.getThread(selectedDataPoint,
				 0, 0).getStartTime();
	startTime /= 1e6;
        timeLabel.setText("StartTime: " + startTime + " Seconds");
	*/
    }
    
    private void updateDataset() {
    	// blow the old data away (unfortunately)
    	dataset.removeAllSeries();
        Enumeration locatedSeries = functionsToSeries.elements();
        while (locatedSeries.hasMoreElements()) {
        	XYSeries thisSeries = ((XYSeries)locatedSeries.nextElement());
        	thisSeries.clear();
        }  
        XYSeries currentSeries;
        // Populate with data
        for (int i=0; i<currentVisibleDataPoints; i++) { 
            java.lang.Number xval = (java.lang.Number)i;

        	List functionProfileList = snapshotThread.getFunctionProfiles();
        	for (int f=0; f<functionProfileList.size(); f++) {
            	FunctionProfile profile = (FunctionProfile)functionProfileList.get(f);
            	String functionName = profile.getName();
            	XYSeries series = (XYSeries)functionsToSeries.get(functionName);
            	if (series == null) {
            		currentSeries = new XYSeries(functionName);
            		functionsToSeries.put(functionName, currentSeries);
            	} else {
            		currentSeries = series; 
            	}
            	if (i <= selectedDataPoint) {
            		double yval = profile.getExclusive(i,0); // *CWL* assume 0 = TIME
            		currentSeries.add(xval, yval);
            	} else {
            		currentSeries.add(xval, null);
            	}
        	}
        }
        locatedSeries = functionsToSeries.elements();
        while (locatedSeries.hasMoreElements()) {
        	dataset.addSeries((XYSeries)locatedSeries.nextElement());
        }   
    }
    
    private void findAndLoadNewData() {
    	// find out if more data is available by attempting a reload
		try {
			boolean success = dataSource.reloadData();
			if (!success) {
				System.err.println("WARNING: Data Source reload failed");
			}
		} catch (Exception e) {
			// *CWL* figure out what to do here.
		}
		int newNumDataPoints = dataSource.getMeanData().getNumSnapshots();
		if (newNumDataPoints > numDataPoints) {
			// we have new data available, update.
			numDataPoints = newNumDataPoints;
		}
		// try to push 10 more data points into visualization.
		//   *CWL* Build in some flow control code here eventually.
    	int targetDataPoints = currentVisibleDataPoints + 10;
    	if (targetDataPoints >= numDataPoints) {
    		// numDataPoints is as far as we can go
    		currentVisibleDataPoints = numDataPoints;
    	} else {
    		currentVisibleDataPoints = targetDataPoints;    		
    	}
		slider.setMaximum(currentVisibleDataPoints-1);
    }
    
    private void findAndLoadNewDataProfileDeprecated() {
    	int targetDataPoint = currentVisibleDataPoints + 10 - 1; // index
    	String targetFileName =
    		trialPath + File.separator + canonicalName + "." + 
    		targetDataPoint + ".0.0";

    	File targetFile = new File(targetFileName);
    	if (targetFile.isFile()) {
    		// load files from current to the target
    		for (int i=currentVisibleDataPoints; i<=targetDataPoint; i++) {
    			dataSource.addThread(i,0,0);
    		}
    		try {
    			boolean success = dataSource.reloadData();
    			if (!success) {
    				System.err.println("WARNING: Data Source reload failed");
    			} else {
    				// Inform observers that more data has been added to the trials.
    				//   This is not technically necessary, but can be nice to give
    				//   paraprof a nice rounded feel to it.
    			}
    		} catch (Exception e) {
    			// *CWL* figure out what to do here.
    		}
    		currentVisibleDataPoints = targetDataPoint+1;
    		slider.setMaximum(currentVisibleDataPoints-1);
    	}
    }
    

}
