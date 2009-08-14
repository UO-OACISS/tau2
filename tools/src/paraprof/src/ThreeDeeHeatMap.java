package edu.uoregon.tau.paraprof;

import java.awt.*;
import java.awt.event.*;
import java.text.DecimalFormat;
import java.util.*;
import java.util.List;

import javax.swing.*;

import edu.uoregon.tau.common.ImageExport;
import edu.uoregon.tau.perfdmf.DataSource;
import edu.uoregon.tau.vis.*;
import edu.uoregon.tau.perfdmf.Thread;

public class ThreeDeeHeatMap extends JFrame implements ActionListener, ImageExport {
    private SteppedComboBox pathSelector = null;
    private SteppedComboBox figureSelector = null;
    private JPanel mainPanel = null;
    private JPanel mapPanel;
    private Map/*<String, double[][][]>*/maps = null;
    private Map/*<String, double[]>*/maxs = null;
    private Map/*<String, double[]>*/mins = null;
    private HeatMapData mapData = null;
    private final static String allPaths = "All Paths";
    private final static String CALLS = "NUMBER OF CALLS";
    private final static String MAX = "MAX MESSAGE BYTES";
    private final static String MIN = "MIN MESSAGE BYTES";
    private final static String MEAN = "MEAN MESSAGE BYTES";
    private final static String STDDEV = "MESSAGE BYTES STDDEV";
    private final static String VOLUME = "TOTAL VOLUME BYTES";
    private final static String[] figures = { CALLS, MAX, MIN, MEAN, STDDEV, VOLUME };
    private String currentPath = allPaths;
    private String currentFigure = CALLS;
    private final static String filenamePrefix = "HeatMap";
    private int size = 0;
    public final static int viewSize = 512; // the size of the heatmap in the interface
    public final static int maxCells = 256; // the number of heatmap cells, max, to show
    public final static int viewRatio = 2; // the ratio between those two
    private HeatMap heatMap = null;
    
    private List threadNames;

    public ThreeDeeHeatMap(String title, HeatMapData mapData, ParaProfTrial ppTrial) {
        super(title);
        
        
        if (threadNames == null) {
            threadNames = new ArrayList();

            for (Iterator it = ppTrial.getDataSource().getAllThreads().iterator(); it.hasNext();) {
                Thread thread = (Thread) it.next();

                if (ppTrial.getDataSource().getExecutionType() == DataSource.EXEC_TYPE_MPI) {
                    threadNames.add(Integer.toString(thread.getNodeID()));
                } else if (ppTrial.getDataSource().getExecutionType() == DataSource.EXEC_TYPE_HYBRID) {
                    threadNames.add(thread.getNodeID() + ":" + thread.getThreadID());
                } else {
                    threadNames.add(thread.getNodeID() + ":" + thread.getContextID() + ":" + thread.getThreadID());
                }

            }
        }
        
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
                //               heatMap.goAway();
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
        for (dataIndex = 0; dataIndex < figures.length; dataIndex++) {
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
        c.insets = new Insets(2, 2, 2, 2);
        c.gridx = 0;
        c.gridy = 0;

        // add the split pane to the main panel, and add the main panel to the window
        mainPanel = new JPanel(new GridBagLayout());
        mainPanel.add(splitPane, c);
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
        c.insets = new Insets(2, 2, 2, 2);

        // title across the top
        c.gridx = 0;
        c.gridy = 0;
        c.gridwidth = 5;
        JLabel title = new JLabel(label, JLabel.CENTER);
        title.setFont(new Font("PE", title.getFont().getStyle(), title.getFont().getSize() * 2));
        panel.add(title, c);

        this.pathSelector.setSelectedItem(currentPath);
        this.pathSelector.addActionListener(this);
        //        this.pathSelector.addKeyListener(this.heatMap.getScanner());
        c.gridy = 1;
        panel.add(new JLabel("Callpath:"), c);
        c.gridy = 2;
        panel.add(this.pathSelector, c);

        this.figureSelector.setSelectedItem(currentPath);
        this.figureSelector.addActionListener(this);
        //        this.figureSelector.addKeyListener(this.heatMap.getScanner());
        c.gridy = 3;
        panel.add(new JLabel("Dataset:"), c);
        c.gridy = 4;
        panel.add(this.figureSelector, c);

        return panel;
    }

    private JPanel buildMapPanel(int index, String label) {

        // Create the visRenderer and register it with the canvas
        VisRenderer visRenderer = new VisRenderer();

        // Create the canvas
        VisCanvas visCanvas = new VisCanvas(visRenderer);
        visCanvas.getActualCanvas().setSize(600, 600);

        ColorScale colorScale = new ColorScale();

        // Create some values
        float values[][] = new float[100][4];
        for (int i = 0; i < 100; i++) {
            values[i][0] = i * (float) Math.sin(i); // value for the x axis
            values[i][1] = i * (float) Math.cos(i); // value for the y axis
            values[i][2] = i * i; // value for the z axis
            values[i][3] = i; // value for the color axis
        }

        BarPlot barPlot = new BarPlot();
        Axes axes = new Axes();

        List[] axisStrings = new ArrayList[4];

        for (int i = 0; i < 4; i++) {

            axisStrings[i] = new ArrayList();

            axisStrings[i].add("0");
            axisStrings[i].add(PlotFactory.getSaneDoubleString(1 * .25));
            axisStrings[i].add(PlotFactory.getSaneDoubleString(1 * .50));
            axisStrings[i].add(PlotFactory.getSaneDoubleString(1 * .75));
            axisStrings[i].add(Float.toString(1));
        }

        axes.setStrings("receiver", "sender", "z", threadNames, threadNames, axisStrings[2]);
        axes.setOnEdge(true);

        int size = mapData.getSize();

        float heightValues[][] = new float[size][size];
        float colorValues[][] = new float[size][size];

//        for (int s = 0; s < size; s++) {
//            for (int r = 0; r < size; r++) {
//                
//            }
//        }
        
        mapData.reset();
        while (mapData.hasNext()) {
            HeatMapData.NextValue next = (HeatMapData.NextValue) mapData.next();
            int x = next.receiver;
            int y = next.sender;
            double value = next.getValue(currentPath, index);
            heightValues[x][y] = (float)value;
            colorValues[x][y] = (float)value;
        }

        barPlot.initialize(axes, 18, 18, 18, heightValues, colorValues, colorScale);

        // Create the scatterPlot
        //ScatterPlot scatterPlot = PlotFactory.createScatterPlot("x axis", "y axis", "z axis", 
        //       "color axis", values, true, colorScale);

        // Set the size
        //scatterPlot.setSize(10, 10, 10);

        // point at the center of the scatterPlot
        visRenderer.setAim(new Vec(5, 5, 5));

        // Add the drawable objects to the visRenderer (the scatterPlot will draw the axes)
        //visRenderer.addShape(scatterPlot);
        visRenderer.addShape(barPlot);
        visRenderer.addShape(colorScale);

        JPanel panel = new JPanel();
        panel.setLayout(new GridBagLayout());
        
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.fill = GridBagConstraints.BOTH;
        gbc.anchor = GridBagConstraints.CENTER;
        gbc.weightx = 1;
        gbc.weighty = 1;
        gbc.insets = new Insets(2, 2, 2, 2);

        panel.add(visCanvas.getActualCanvas(),gbc);
        return panel;
    }

    public void actionPerformed(ActionEvent actionEvent) {
        try {
            Object eventSrc = actionEvent.getSource();
            Dimension oldSize = this.getSize();
            //System.out.println("oldSize: " + oldSize.width + " X " + oldSize.height);
            if (eventSrc.equals(this.pathSelector)) {
                String newPath = (String) this.pathSelector.getSelectedItem();
                if (!newPath.equals(currentPath)) {
                    currentPath = newPath;
                    redrawHeatMap(oldSize);
                }
            }
            if (eventSrc.equals(this.figureSelector)) {
                String newFigure = (String) this.figureSelector.getSelectedItem();
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
//        this.figureSelector.addKeyListener(this.heatMap.getScanner());
//        this.figureSelector.addKeyListener(this.heatMap.getScanner());
//        this.heatMap.requestFocus();
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
        System.out.print("Memory - Free: " + f.format(java.lang.Runtime.getRuntime().freeMemory() / 1000000.0));
        System.out.print("\tTotal: " + f.format(java.lang.Runtime.getRuntime().totalMemory() / 1000000.0));
        System.out.println("\tMax: " + f.format(java.lang.Runtime.getRuntime().maxMemory() / 1000000.0));
    }

}
