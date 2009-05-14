package edu.uoregon.tau.vis;

import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.text.DecimalFormat;
import java.util.Map;

import javax.swing.*;

import edu.uoregon.tau.common.ImageExport;

public class HeatMapWindow extends JFrame implements ActionListener, ImageExport {

	private SteppedComboBox pathSelector = null;
	private SteppedComboBox figureSelector = null;
	private JPanel mainPanel = null;
	private JPanel mapPanel;
	private Map/*<String, double[][][]>*/ maps = null;
	private Map/*<String, double[]>*/ maxs = null;
	private Map/*<String, double[]>*/ mins = null;
	private HeatMapData mapData = null;
	private final static String allPaths = "All Paths";
	private final static String CALLS = "NUMBER OF CALLS";
	private final static String MAX = "MAX MESSAGE BYTES";
	private final static String MIN = "MIN MESSAGE BYTES";
	private final static String MEAN = "MEAN MESSAGE BYTES";
	private final static String STDDEV = "MESSAGE BYTES STDDEV";
	private final static String VOLUME = "TOTAL VOLUME BYTES";
	private final static String[] figures = {CALLS, MAX, MIN, MEAN, STDDEV, VOLUME};
	private String currentPath = allPaths;
	private String currentFigure = CALLS;
	private final static String filenamePrefix = "HeatMap";
	private int size = 0;
	public final static int viewSize = 512;  // the size of the heatmap in the interface
	public final static int maxCells = 256;   // the number of heatmap cells, max, to show
	public final static int viewRatio = 2;   // the ratio between those two
	private HeatMap heatMap = null;

	public HeatMapWindow(String title, HeatMapData mapData) {
		super(title);
		this.mapData = mapData;
		this.maxs = mapData.getMaxs();
		this.mins = mapData.getMins();
		this.size = mapData.getSize();
		pathSelector = new SteppedComboBox(mapData.getPaths().toArray());
		Dimension d = pathSelector.getPreferredSize();
	    pathSelector.setPreferredSize(new Dimension(50, d.height));
	    pathSelector.setPopupWidth(d.width);
		figureSelector = new SteppedComboBox(figures);
		d = figureSelector.getPreferredSize();
	    figureSelector.setPreferredSize(new Dimension(50, d.height));
	    figureSelector.setPopupWidth(d.width);
		drawFigures(true);
		// exit when the user closes the main window.
		addWindowListener(new WindowAdapter() {
			public void windowClosing(WindowEvent e) {
				heatMap.goAway();
				dispose();
				System.gc();
				// printMemoryStats("WINDOW CLOSED");
			}
		});
		// printMemoryStats("WINDOW OPEN");
	}

	private void drawFigures(boolean centerWindow) {
		// which figure type is requested?
		int dataIndex = 0;
		for (dataIndex = 0 ; dataIndex < figures.length ; dataIndex++) {
			if (figures[dataIndex].equals(currentFigure)) {
				break;
			}
		}

		// build the split pane
		JSplitPane splitPane = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT);
		splitPane.setResizeWeight(1);
		splitPane.setOneTouchExpandable(true);
		mapPanel = buildMapPanel(dataIndex, currentFigure);
		splitPane.setLeftComponent(mapPanel);
		splitPane.setRightComponent(buildOptionPanel("DISPLAY OPTIONS"));
		
		// set up the constraints for the main panel
		GridBagConstraints c = new GridBagConstraints();
		c.fill = GridBagConstraints.BOTH;
		c.anchor = GridBagConstraints.CENTER;
		c.weightx = 0.99;
		c.weighty = 0.99;
		c.insets = new Insets(2,2,2,2);
		c.gridx = 0;
		c.gridy = 0;

		// add the split pane to the main panel, and add the main panel to the window
		mainPanel = new JPanel(new GridBagLayout());
		mainPanel.add(splitPane,c);
		this.getContentPane().add(mainPanel);
		
        if (centerWindow) {
	        Toolkit tk = Toolkit.getDefaultToolkit();
	        Dimension screenDimension = tk.getScreenSize();
	        int screenHeight = screenDimension.height;
	        int screenWidth = screenDimension.width;
	        //Window Stuff.
	        int windowWidth = 1000;
	        int windowHeight = 800;
	        //Find the center position with respect to this window.
	        int xPosition = (screenWidth - windowWidth) / 2;
	        int yPosition = (screenHeight - windowHeight) / 2;
	        setLocation(xPosition, yPosition);
        }
		this.pack();
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
		this.pathSelector.addKeyListener(this.heatMap.getScanner());
		c.gridy = 1;
		panel.add(new JLabel("Callpath:"),c);
		c.gridy = 2;
		panel.add(this.pathSelector,c);
		
		this.figureSelector.setSelectedItem(currentPath);
		this.figureSelector.addActionListener(this);
		this.figureSelector.addKeyListener(this.heatMap.getScanner());
		c.gridy = 3;
		panel.add(new JLabel("Dataset:"),c);
		c.gridy = 4;
		panel.add(this.figureSelector,c);

		return panel;
	}

	private JPanel buildMapPanel(int index, String label) {
		JPanel panel = new JPanel(new GridBagLayout());
		panel.setBorder(BorderFactory.createLineBorder(Color.BLACK));
		GridBagConstraints c = new GridBagConstraints();
		c.fill = GridBagConstraints.BOTH;
		c.anchor = GridBagConstraints.CENTER;
		c.weightx = 0.01;
		c.insets = new Insets(2,2,2,2);
		DecimalFormat f = new DecimalFormat("0.00E0");
		DecimalFormat f2 = new DecimalFormat("0.##");

		// title across the top
		c.gridx = 0;
		c.gridy = 0;
		c.gridwidth = 5;
		JLabel title = new JLabel(label, JLabel.CENTER);
		title.setFont(new Font("PE", title.getFont().getStyle(), title.getFont().getSize()*2));
		panel.add(title,c);
		JLabel path = new JLabel(currentPath, JLabel.CENTER);
		c.gridy = 1;
		panel.add(path,c);

		// the x axis and the top of the legend
		c.gridwidth = 1;
		c.gridy = 2;
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
		c.gridy = 3;
		panel.add(new JLabel("0", JLabel.CENTER),c);
		c.gridy = 4;
		c.weighty = 0.99;
		JLabel vertical = new JLabel("SENDER", JLabel.CENTER);
		vertical.setUI(new VerticalLabelUI(false));
		panel.add(vertical,c);
		c.weighty = 0.01;
		c.gridx = 1;
		c.gridy = 3;
		c.gridwidth = 3;
		c.gridheight = 3;
		if (heatMap == null) {
			this.heatMap = new HeatMap(mapData, index, currentPath, filenamePrefix);
		} else {
			this.heatMap.update(mapData, index, currentPath, filenamePrefix);
		}
		JScrollPane scroller = new JScrollPane(heatMap);
		scroller.setPreferredSize(new Dimension(viewSize,viewSize));
		scroller.setSize(new Dimension(viewSize,viewSize));
	    panel.add(scroller, c);
		c.gridwidth = 1;
		c.gridheight = 1;
		c.gridy = 3;
		c.gridx = 4;
		double tmp = mapData.getMax(currentPath, index);
		if (tmp > 999) {
			panel.add(new JLabel(f.format(tmp), JLabel.CENTER),c);
		} else {
			panel.add(new JLabel(f2.format(tmp), JLabel.CENTER),c);
		}
		c.gridy = 4;
		c.weighty = 0.99;
		c.fill = GridBagConstraints.VERTICAL;
	    panel.add(new HeatLegend(), c);
		c.fill = GridBagConstraints.BOTH;
	    panel.add(new JPanel(), c);
		c.weighty = 0.01;

		// the bottom of the y axis and the bottom of the legend
		c.gridx = 0;
		c.gridy = 5;
		tmp = mapData.getMin(currentPath, index);
		panel.add(new JLabel(Integer.toString(size-1), JLabel.CENTER),c);
		c.gridx = 4;
		if (tmp > 999) {
			panel.add(new JLabel(f.format(tmp), JLabel.CENTER),c);
		} else {
			panel.add(new JLabel(f2.format(tmp), JLabel.CENTER),c);
		}
		return panel;
	}

	public void actionPerformed(ActionEvent actionEvent) {
		try {
			Object eventSrc = actionEvent.getSource();
			Dimension oldSize = this.getSize();
			//System.out.println("oldSize: " + oldSize.width + " X " + oldSize.height);
			if (eventSrc.equals(this.pathSelector)) {
				String newPath = (String)this.pathSelector.getSelectedItem();
				if (!newPath.equals(currentPath)) {
					currentPath = newPath;
					redrawHeatMap(oldSize);
				}
			}
			if (eventSrc.equals(this.figureSelector)) {
				String newFigure = (String)this.figureSelector.getSelectedItem();
				if (!newFigure.equals(currentFigure)) {
					currentFigure = newFigure;
					redrawHeatMap(oldSize);
				}
			}
		} catch (Exception e) {
			System.err.println("actionPerformed Exception: " + e.getMessage());
			e.printStackTrace();
		} 
	}

	private void redrawHeatMap(Dimension oldSize) {
		this.setVisible(false);
		this.remove(mainPanel);
		mainPanel = null;
		System.gc();
		drawFigures(false);
		this.setSize(oldSize);
		// new heatmap, new scanner, so add listeners
		this.figureSelector.addKeyListener(this.heatMap.getScanner());
		this.figureSelector.addKeyListener(this.heatMap.getScanner());
		this.heatMap.requestFocus();
		this.setVisible(true);
	}

	/**
	 * @return the heatMap
	 */
	public HeatMap getHeatMap() {
		return heatMap;
	}

    public void export(Graphics2D g2d, boolean toScreen, boolean fullWindow, boolean drawHeader) {
        //heatMap.paint(g2d);
        mapPanel.setDoubleBuffered(false);
        heatMap.setDoubleBuffered(false);
        mapPanel.paintAll(g2d);
        heatMap.setDoubleBuffered(true);
        mapPanel.setDoubleBuffered(true);
    }

    public Dimension getImageSize(boolean fullScreen, boolean header) {
        return mapPanel.getSize();
    }

	private static void printMemoryStats(String header) {
		DecimalFormat f = new DecimalFormat("#.## MB");
		System.out.print(header + " - ");
		System.out.print("Memory - Free: " + f.format(java.lang.Runtime.getRuntime().freeMemory()/1000000.0));
		System.out.print("\tTotal: " + f.format(java.lang.Runtime.getRuntime().totalMemory()/1000000.0));
		System.out.println("\tMax: " + f.format(java.lang.Runtime.getRuntime().maxMemory()/1000000.0));
	}
	
}
