package edu.uoregon.tau.perfexplorer.glue;

import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.*;

import java.lang.Math;

import java.net.URL;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.StringTokenizer;
import java.text.DecimalFormat;

import org.jfree.chart.ChartPanel;

import edu.uoregon.tau.common.Utility;
import edu.uoregon.tau.perfexplorer.client.HeatMap;
import edu.uoregon.tau.perfexplorer.client.HeatLegend;
import edu.uoregon.tau.perfexplorer.client.VerticalLabelUI;
import edu.uoregon.tau.perfexplorer.client.PerfExplorerChartJMenuBar;
import edu.uoregon.tau.perfexplorer.client.PerfExplorerClient;

public class BuildMessageHeatMap extends AbstractPerformanceOperation implements ActionListener {

	private Map<String, double[][][]> maps = new HashMap<String, double[][][]>();
	private Map<String, double[]> maxs = new HashMap<String, double[]>(); 
	private Map<String, double[]> mins = new HashMap<String, double[]>(); 
/*	private double[][][] map = null;
	private double[] max = {0,0,0,0,0}; 
	private double[] min = {Double.MAX_VALUE,Double.MAX_VALUE,Double.MAX_VALUE,Double.MAX_VALUE,Double.MAX_VALUE}; 
*/	private int size = 0;
	private JComboBox pathSelector = new JComboBox();
	private final static String allPaths = "All Paths";
	private String currentPath = allPaths;
	private JPanel mainPanel = null;

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

			for (String key : maps.keySet()) {
				this.pathSelector.addItem(key);
			}

			window = new JFrame("Message Size Heat Maps");
	        URL url = Utility.getResource("tau32x32.gif");
			if (url != null) 
				window.setIconImage(Toolkit.getDefaultToolkit().getImage(url));
			try {
				centerFrame(window);
			} catch (NullPointerException e) {
				// we didn't have a client window open.
			}

			drawFigures();

		}
		return this.outputs;
	}

	private void drawFigures() {
		mainPanel = new JPanel(new GridBagLayout());
		window.getContentPane().add(mainPanel);
		GridBagConstraints c = new GridBagConstraints();
		c.fill = GridBagConstraints.BOTH;
		c.anchor = GridBagConstraints.CENTER;
		c.weightx = 0.5;
		c.insets = new Insets(2,2,2,2);

		c.gridx = 0;
		c.gridy = 0;
		mainPanel.add(buildMapPanel(0, "NUMBER OF CALLS", "NumEvents"),c);
		c.gridx = 1;
		mainPanel.add(buildMapPanel(1, "MAX MESSAGE BYTES", "MaxMessageSize"),c);
		c.gridx = 2;
		mainPanel.add(buildMapPanel(2, "MIN MESSAGE BYTES", "MinMessageSize"),c);

		c.gridx = 0;
		c.gridy = 1;
		mainPanel.add(buildMapPanel(3, "MEAN MESSAGE BYTES", "MeanMessageSize"),c);
		c.gridx = 1;
		mainPanel.add(buildMapPanel(4, "MESSAGE BYTES STDDEV", "MessageSizeStdDev"),c);
		c.gridx = 2;
		mainPanel.add(buildOptionPanel("DISPLAY OPTIONS"),c);
		window.pack();
		window.setVisible(true);
	}

	private Component buildOptionPanel(String label) {
		JPanel panel = new JPanel(new GridBagLayout());
		panel.setBorder(BorderFactory.createLineBorder(Color.BLACK));
		GridBagConstraints c = new GridBagConstraints();
		c.fill = GridBagConstraints.BOTH;
		c.anchor = GridBagConstraints.CENTER;
		c.weightx = 0.01;
		c.insets = new Insets(2,2,2,2);

		// title across the top
		c.gridx = 0;
		c.gridy = 0;
		c.gridwidth = 5;
		JLabel title = new JLabel(label, JLabel.CENTER);
		title.setFont(new Font("PE", title.getFont().getStyle(), title.getFont().getSize()*2));
		panel.add(title,c);

		this.pathSelector.setSelectedItem(currentPath);
		this.pathSelector.addActionListener(this);
		c.gridy = 1;
		panel.add(new JLabel("Callpath:"),c);
		c.gridy = 2;
		panel.add(this.pathSelector,c);

		return panel;
	}

	private void extractData(PerformanceResult input, Integer thread, String event, String first, String path) {
		double numEvents, eventMax, eventMin, eventMean, eventSumSqr, stdev = 0;
		double[][][] map = new double[5][size][size];
		double[] max = {0,0,0,0,0}; 
		double[] min = {Double.MAX_VALUE,Double.MAX_VALUE,Double.MAX_VALUE,Double.MAX_VALUE,Double.MAX_VALUE}; 
		
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
			String receiver = st.nextToken();
			numEvents = input.getUsereventNumevents(thread, event);
			map[0][thread][Integer.parseInt(receiver)] = numEvents;
			max[0] = max[0] < numEvents ? numEvents : max[0];
			min[0] = numEvents > 0 && min[0] > numEvents ? numEvents : min[0];
			eventMax = input.getUsereventMax(thread, event);
			map[1][thread][Integer.parseInt(receiver)] = eventMax;
			max[1] = max[1] < eventMax ? eventMax : max[1];
			min[1] = eventMax > 0 && min[1] > eventMax ? eventMax : min[1];
			eventMin = input.getUsereventMin(thread, event);
			map[2][thread][Integer.parseInt(receiver)] = eventMin;
			max[2] = max[2] < eventMin ? eventMin : max[2];
			min[2] = eventMin > 0 && min[2] > eventMin ? eventMin : min[2];
			eventMean = input.getUsereventMean(thread, event);
			map[3][thread][Integer.parseInt(receiver)] = eventMean;
			max[3] = max[3] < eventMean ? eventMean : max[3];
			min[3] = eventMean > 0 && min[3] > eventMean ? eventMean : min[3];
			eventSumSqr = input.getUsereventSumsqr(thread, event);
			if (numEvents > 0)
				stdev = Math.sqrt(Math.abs((eventSumSqr/numEvents)-(eventMean*eventMean)));
			else
				stdev = 0;
			map[4][thread][Integer.parseInt(receiver)] = stdev;
			max[4] = max[4] < stdev ? stdev : max[4];
			min[4] = stdev > 0 && min[4] > stdev ? stdev : min[4];
		}
	}

	private JPanel buildMapPanel(int index, String label, String filename) {
		JPanel panel = new JPanel(new GridBagLayout());
		panel.setBorder(BorderFactory.createLineBorder(Color.BLACK));
		GridBagConstraints c = new GridBagConstraints();
		c.fill = GridBagConstraints.BOTH;
		c.anchor = GridBagConstraints.CENTER;
		c.weightx = 0.01;
		c.insets = new Insets(2,2,2,2);
		DecimalFormat f = new DecimalFormat("0.##E0");

		// title across the top
		c.gridx = 0;
		c.gridy = 0;
		c.gridwidth = 5;
		JLabel title = new JLabel(label, JLabel.CENTER);
		title.setFont(new Font("PE", title.getFont().getStyle(), title.getFont().getSize()*2));
		panel.add(title,c);

		// the x axis and the top of the legend
		c.gridwidth = 1;
		c.gridy = 1;
		c.gridx = 1;
		panel.add(new JLabel("0", JLabel.CENTER),c);
		c.gridx = 2;
		c.weightx = 0.99;
		panel.add(new JLabel("RECEIVER", JLabel.CENTER),c);
		c.weightx = 0.01;
		c.gridx = 3;
		panel.add(new JLabel(Integer.toString(size-1), JLabel.CENTER),c);

		// the y axis and the map and the legend
		c.gridx = 0;
		c.gridy = 2;
		panel.add(new JLabel("0", JLabel.CENTER),c);
		c.gridy = 3;
		c.weighty = 0.99;
		JLabel vertical = new JLabel("SENDER", JLabel.CENTER);
		vertical.setUI(new VerticalLabelUI(false));
		panel.add(vertical,c);
		c.weighty = 0.01;
		c.gridx = 1;
		c.gridy = 2;
		c.gridwidth = 3;
		c.gridheight = 3;
	    panel.add(new HeatMap(maps.get(currentPath)[index], size, maxs.get(currentPath)[index], mins.get(currentPath)[index], filename), c);
		c.gridwidth = 1;
		c.gridheight = 1;
		c.gridy = 2;
		c.gridx = 4;
		panel.add(new JLabel(f.format(maxs.get(currentPath)[index]), JLabel.CENTER),c);
		c.gridy = 3;
		c.weighty = 0.99;
	    panel.add(new HeatLegend(), c);
	    panel.add(new JPanel(), c);
		c.weighty = 0.01;

		// the bottom of the y axis and the bottom of the legend
		c.gridx = 0;
		c.gridy = 4;
		panel.add(new JLabel(Integer.toString(size-1), JLabel.CENTER),c);
		c.gridx = 4;
		panel.add(new JLabel(f.format(mins.get(currentPath)[index]), JLabel.CENTER),c);
		return panel;
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
        frame.setSize(new java.awt.Dimension(windowWidth, windowHeight));
 	}

	public void actionPerformed(ActionEvent actionEvent) {
		try {
			Object eventSrc = actionEvent.getSource();
			if (eventSrc.equals(this.pathSelector)) {
				String newPath = (String)this.pathSelector.getSelectedItem();
				if (!newPath.equals(currentPath)) {
					currentPath = newPath;
					window.remove(mainPanel);
					mainPanel = null;
					drawFigures();
				}
			}
		} catch (Exception e) {
			System.err.println("actionPerformed Exception: " + e.getMessage());
			e.printStackTrace();
		} 
	}
}
