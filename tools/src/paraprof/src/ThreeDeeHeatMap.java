package edu.uoregon.tau.paraprof;

import java.awt.*;
import java.awt.event.*;
import java.text.DecimalFormat;
import java.util.*;
import java.util.List;

import javax.swing.*;

import edu.uoregon.tau.common.ImageExport;
import edu.uoregon.tau.perfdmf.DataSource;
import edu.uoregon.tau.perfdmf.UtilFncs;
import edu.uoregon.tau.vis.*;
import edu.uoregon.tau.perfdmf.Thread;

public class ThreeDeeHeatMap extends JFrame implements ActionListener, ImageExport, VisCanvasListener {
    private SteppedComboBox pathSelector = null;
    private SteppedComboBox heightComboBox = null;
    private SteppedComboBox colorComboBox = null;
    private JPanel mainPanel = null;
    private JPanel mapPanel;
    private HeatMapData mapData = null;
    private final static String allPaths = "All Paths";

    private final static int CALLS = 0;
    private final static int MAX = 1;
    private final static int MIN = 2;
    private final static int MEAN = 3;
    private final static int STDDEV = 4;
    private final static int VOLUME = 5;

    private final static String[] metricStrings = { "Number of calls", "Max message size (bytes)", "Min message size (bytes)",
            "Mean message size (bytes)", "Std. dev. message size (bytes)", "Message volume (bytes)" };

    private String currentPath = allPaths;
    private int heightMetric = VOLUME;
    private int colorMetric = MAX;


    private BarPlot barPlot;
    private Axes axes;
    private VisRenderer visRenderer;
    private ColorScale colorScale;
    private JSplitPane splitPane;

    private List threadNames;

    public ThreeDeeHeatMap(String title, HeatMapData mapData, ParaProfTrial ppTrial, Component invoker) {
        super(title);

        setSize(ParaProfUtils.checkSize(new Dimension(1000, 750)));
        this.setLocation(WindowPlacer.getNewLocation(this, invoker));

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

        pathSelector = new SteppedComboBox(mapData.getPaths().toArray());
        Dimension d = pathSelector.getPreferredSize();
        pathSelector.setPreferredSize(new Dimension(50, d.height));
        pathSelector.setPopupWidth(d.width);

        heightComboBox = new SteppedComboBox(metricStrings);
        heightComboBox.setSelectedIndex(heightMetric);
        d = heightComboBox.getPreferredSize();
        heightComboBox.setPreferredSize(new Dimension(50, d.height));
        heightComboBox.setPopupWidth(d.width);

        colorComboBox = new SteppedComboBox(metricStrings);
        colorComboBox.setSelectedIndex(colorMetric);
        d = colorComboBox.getPreferredSize();
        colorComboBox.setPreferredSize(new Dimension(50, d.height));
        colorComboBox.setPopupWidth(d.width);

        buildPanels();
        // exit when the user closes the main window.
        addWindowListener(new WindowAdapter() {
            public void windowClosing(WindowEvent e) {
                //               heatMap.goAway();
                dispose();
                System.gc();
                // printMemoryStats("WINDOW CLOSED");
            }
        });

        setSize(ParaProfUtils.checkSize(new Dimension(1000, 750)));
        setSize(new Dimension(1000, 750));
        this.setLocation(WindowPlacer.getNewLocation(this, invoker));
        // printMemoryStats("WINDOW OPEN");
    }

    private void buildPanels() {
        mapPanel = buildMapPanel();

        // Create the control panel, if desired
        JTabbedPane tabbedPane = new JTabbedPane();
        tabbedPane.setTabLayoutPolicy(JTabbedPane.SCROLL_TAB_LAYOUT);
        tabbedPane.addTab("Plot", barPlot.getControlPanel(visRenderer));
        tabbedPane.addTab("Axes", barPlot.getAxes().getControlPanel(visRenderer));
        tabbedPane.addTab("ColorScale", colorScale.getControlPanel(visRenderer));
        tabbedPane.addTab("Render", visRenderer.getControlPanel());
        tabbedPane.setMinimumSize(new Dimension(300, 260));

        JPanel rightPanel = new JPanel();

        rightPanel.setLayout(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();

        gbc.gridx = 0;
        gbc.gridy = 0;
        gbc.weightx = 0.5;
        gbc.weighty = 1;
        gbc.fill = GridBagConstraints.BOTH;
        gbc.anchor = GridBagConstraints.CENTER;

        rightPanel.add(buildOptionPanel("Display Options"), gbc);

        gbc.gridy = 1;
        gbc.anchor = GridBagConstraints.SOUTH;
        gbc.weighty = 0;
        rightPanel.add(tabbedPane, gbc);

        // set up the constraints for the main panel
        GridBagConstraints c = new GridBagConstraints();
        c.fill = GridBagConstraints.BOTH;
        c.anchor = GridBagConstraints.CENTER;
        c.weightx = 0.99;
        c.weighty = 0.99;
        c.insets = new Insets(2, 2, 2, 2);
        c.gridx = 0;
        c.gridy = 0;

        // build the split pane
        splitPane = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT, mapPanel, rightPanel);
        splitPane.setResizeWeight(1.0);
        splitPane.setOneTouchExpandable(true);
        splitPane.setContinuousLayout(true);

        // add the split pane to the main panel, and add the main panel to the window
        mainPanel = new JPanel(new GridBagLayout());
        mainPanel.add(splitPane, c);
        this.getContentPane().add(mainPanel);

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

        heightComboBox.setSelectedItem(currentPath);
        heightComboBox.addActionListener(this);
        //        this.figureSelector.addKeyListener(this.heatMap.getScanner());
        c.gridy = 3;
        panel.add(new JLabel("Height Value:"), c);
        c.gridy = 4;
        panel.add(this.heightComboBox, c);

        colorComboBox.setSelectedItem(currentPath);
        colorComboBox.addActionListener(this);
        //        this.figureSelector.addKeyListener(this.heatMap.getScanner());
        c.gridy = 5;
        panel.add(new JLabel("Color Value:"), c);
        c.gridy = 6;
        panel.add(this.colorComboBox, c);

        return panel;
    }

    private void processData() {

        List heightAxisStrings = new ArrayList();

        float maxHeightValue = (float) mapData.getMax(currentPath, heightMetric);
        heightAxisStrings.add("0");
        heightAxisStrings.add(PlotFactory.getSaneDoubleString(maxHeightValue * .25));
        heightAxisStrings.add(PlotFactory.getSaneDoubleString(maxHeightValue * .50));
        heightAxisStrings.add(PlotFactory.getSaneDoubleString(maxHeightValue * .75));
        heightAxisStrings.add(PlotFactory.getSaneDoubleString(maxHeightValue));

        axes.setStrings("receiver", "sender", metricStrings[heightMetric], threadNames, threadNames, heightAxisStrings);
        axes.setOnEdge(true);

        float minColorValue = (float) mapData.getMin(currentPath, colorMetric);
        float maxColorValue = (float) mapData.getMax(currentPath, colorMetric);

        //colorScale.setStrings(Float.toString(minColorValue), Float.toString(maxColorValue), metricStrings[colorMetric]);
        colorScale.setStrings(UtilFncs.formatDouble(minColorValue, 6, true), UtilFncs.formatDouble(maxColorValue, 6, true),
                metricStrings[colorMetric]);

        int size = mapData.getSize();

        float heightValues[][] = new float[size][size];
        float colorValues[][] = new float[size][size];

        mapData.reset();
        while (mapData.hasNext()) {
            HeatMapData.NextValue next = (HeatMapData.NextValue) mapData.next();
            int x = next.receiver;
            int y = next.sender;
            heightValues[x][y] = (float) next.getValue(currentPath, heightMetric);
            colorValues[x][y] = (float) next.getValue(currentPath, colorMetric) - minColorValue;
        }

        if (barPlot == null) {
            barPlot = new BarPlot(axes, colorScale);
        }
        barPlot.setValues(18, 18, 8, heightValues, colorValues);

    }

    private JPanel buildMapPanel() {

        // Create the visRenderer and register it with the canvas
        visRenderer = new VisRenderer();

        // Create the canvas
        VisCanvas visCanvas = new VisCanvas(visRenderer);
        visCanvas.getActualCanvas().setSize(900, 900);
        visRenderer.setVisCanvasListener(this);
        colorScale = new ColorScale();

        axes = new Axes();

        processData();

        // point at the center of the plot
        visRenderer.setAim(new Vec(5, 5, 0));

        // Add the drawable objects to the visRenderer (the scatterPlot will draw the axes)
        //visRenderer.addShape(scatterPlot);
        visRenderer.addShape(barPlot);
        visRenderer.addShape(colorScale);

        barPlot.setScaleZ(0.0f);

        (new java.lang.Thread() {
            private float scaleZ = 0.0f;
            private float scaleZTarget = 1.0f;
            private long duration = 350;
            private float sleepAmount = 0;

            public void run() {
                try {

                    while (!visRenderer.isReadyToDraw()) {
                        sleep(100);
                    }

                    long init = System.currentTimeMillis();
                    while (true) {
                        sleep((long) sleepAmount);
                        long start = System.currentTimeMillis();
                        visRenderer.redraw();
                        long stop = System.currentTimeMillis();

                        long renderCost = stop - start;
                        if (renderCost <= 0) {
                            renderCost = 1;
                        }
                        long timeProgress = stop - init;
                        long numSteps = (duration - timeProgress) / renderCost;
                        //System.out.println("numSteps = " + numSteps);

                        float scaleStep = (scaleZTarget - scaleZ) / (float) numSteps;

                        scaleZ += scaleStep;
                        float val = (float) Math.log(((scaleZ * 9) + 1)) / (float) Math.log(10.0f);
                        //System.out.println(val);
                        barPlot.setScaleZ(val);
                        if (scaleZ >= 1.0f) {
                            barPlot.setScaleZ(1.0f);
                            visRenderer.redraw();
                            return;
                        }
                    }
                } catch (InterruptedException ex) {}
            }
        }).start();

        JPanel panel = new JPanel() {
            public Dimension getMinimumSize() {
                return new Dimension(10, 10);
            }
        };
        panel.setLayout(new GridBagLayout());

        GridBagConstraints gbc = new GridBagConstraints();
        gbc.fill = GridBagConstraints.BOTH;
        gbc.anchor = GridBagConstraints.CENTER;
        gbc.weightx = 1;
        gbc.weighty = 1;
        gbc.insets = new Insets(2, 2, 2, 2);

        panel.add(visCanvas.getActualCanvas(), gbc);
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
            if (eventSrc.equals(this.heightComboBox)) {
                if (heightComboBox.getSelectedIndex() != heightMetric) {
                    heightMetric = heightComboBox.getSelectedIndex();
                    redrawHeatMap(oldSize);
                }
            }
            if (eventSrc.equals(this.colorComboBox)) {
                if (colorComboBox.getSelectedIndex() != colorMetric) {
                    colorMetric = colorComboBox.getSelectedIndex();
                    redrawHeatMap(oldSize);
                }
            }
        } catch (Exception e) {
            ParaProfUtils.handleException(e);
        }
    }

    private void redrawHeatMap(Dimension oldSize) {
        processData();
        visRenderer.redraw();

        //        this.setVisible(false);
        //        this.remove(mainPanel);
        //        mainPanel = null;
        //        drawFigures(false);
        //        this.setVisible(true);
    }



    public void export(Graphics2D g2d, boolean toScreen, boolean fullWindow, boolean drawHeader) {
        //heatMap.paint(g2d);
        mapPanel.setDoubleBuffered(false);
        mapPanel.paintAll(g2d);
        mapPanel.setDoubleBuffered(true);
    }

    public Dimension getImageSize(boolean fullScreen, boolean header) {
        return mapPanel.getSize();
    }

    public void createNewCanvas() {

        VisCanvas visCanvas = new VisCanvas(visRenderer);

        JPanel panel = new JPanel();
        panel.setLayout(new GridBagLayout());

        GridBagConstraints gbc = new GridBagConstraints();
        gbc.fill = GridBagConstraints.BOTH;
        gbc.anchor = GridBagConstraints.CENTER;
        gbc.weightx = 1;
        gbc.weighty = 1;
        gbc.insets = new Insets(2, 2, 2, 2);

        panel.add(visCanvas.getActualCanvas(), gbc);

        splitPane.setLeftComponent(panel);

    }

}
