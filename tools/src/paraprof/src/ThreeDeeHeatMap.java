package edu.uoregon.tau.paraprof;

import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.awt.image.ImageObserver;
import java.awt.print.PageFormat;
import java.awt.print.Printable;
import java.awt.print.PrinterException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import javax.swing.*;

import edu.uoregon.tau.common.StoppableThread;
import edu.uoregon.tau.perfdmf.DataSource;
import edu.uoregon.tau.perfdmf.Thread;
import edu.uoregon.tau.perfdmf.UtilFncs;
import edu.uoregon.tau.vis.*;

public class ThreeDeeHeatMap extends JFrame implements ActionListener, ThreeDeeImageProvider, VisCanvasListener, Printable {
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
    private VisCanvas visCanvas;
    private ColorScale colorScale;
    private JSplitPane splitPane;

    private List threadNames;

    float oldHeightValues[][];
    float oldColorValues[][];

    float newHeightValues[][];
    float newColorValues[][];

    float curHeightValues[][];
    float curColorValues[][];

    StoppableThread animator;

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
        pathSelector.setMinimumSize(new Dimension(50, d.height));
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
                dispose();
                System.gc();
            }
        });

        setSize(ParaProfUtils.checkSize(new Dimension(1000, 750)));
        setSize(new Dimension(1000, 750));
        this.setLocation(WindowPlacer.getNewLocation(this, invoker));
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

    private void copyMatrix(float[][] dest, float[][] src, int size) {
        for (int x = 0; x < size; x++) {
            for (int y = 0; y < size; y++) {
                dest[x][y] = src[x][y];
            }
        }
    }

    private void zeroMatrix(float[][] matrix, int size) {
        for (int x = 0; x < size; x++) {
            for (int y = 0; y < size; y++) {
                matrix[x][y] = 0;
            }
        }
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

        // First, check if an animator is still running, if so, tell it to stop and then wait for it
        if (animator != null) {
            animator.requestStop();
            try {
                animator.join();
            } catch (InterruptedException ex) {}
        }

        float minColorValue = (float) mapData.getMin(currentPath, colorMetric);
        float maxColorValue = (float) mapData.getMax(currentPath, colorMetric);
        float minHeightValue = (float) mapData.getMin(currentPath, heightMetric);
        float maxHeightValue = (float) mapData.getMax(currentPath, heightMetric);

        List heightAxisStrings = new ArrayList();
        heightAxisStrings.add("0");
        heightAxisStrings.add(PlotFactory.getSaneDoubleString(maxHeightValue * .25));
        heightAxisStrings.add(PlotFactory.getSaneDoubleString(maxHeightValue * .50));
        heightAxisStrings.add(PlotFactory.getSaneDoubleString(maxHeightValue * .75));
        heightAxisStrings.add(PlotFactory.getSaneDoubleString(maxHeightValue));

        axes.setStrings("receiver", "sender", metricStrings[heightMetric], threadNames, threadNames, heightAxisStrings);
        axes.setOnEdge(true);

        //colorScale.setStrings(Float.toString(minColorValue), Float.toString(maxColorValue), metricStrings[colorMetric]);
        colorScale.setStrings(UtilFncs.formatDouble(minColorValue, 6, true), UtilFncs.formatDouble(maxColorValue, 6, true),
                metricStrings[colorMetric]);

        int size = mapData.getSize();

        zeroMatrix(newHeightValues, size);
        zeroMatrix(newColorValues, size);

        mapData.reset();
        while (mapData.hasNext()) {
            HeatMapData.NextValue next = (HeatMapData.NextValue) mapData.next();
            int x = next.receiver;
            int y = next.sender;
            newHeightValues[x][y] = (float) next.getValue(currentPath, heightMetric) / maxHeightValue;
            newColorValues[x][y] = (float) next.getValue(currentPath, colorMetric) - minColorValue;
        }

        boolean animate = false;
        if (!animate) {
            barPlot.setValues(newHeightValues, newColorValues);
            visRenderer.redraw();
        } else {

            boolean first = false;
            if (animator != null) {
                for (int x = 0; x < size; x++) {
                    for (int y = 0; y < size; y++) {
                        oldHeightValues[x][y] = curHeightValues[x][y];
                        oldColorValues[x][y] = curColorValues[x][y];
                    }
                }
            } else {
                first = true;
            }

            final boolean fromZero = first;
            animator = new StoppableThread() {
                private float ratio = 0.0f;
                private float scaleZTarget = 1.0f;
                private long duration = 350;
                private float sleepAmount = 0;

                public void run() {
                    try {
                        if (fromZero) {
                            barPlot.setScaleZ(0.0f);
                        }
                        while (!visRenderer.isReadyToDraw()) {
                            sleep(100);
                        }

                        int size = mapData.getSize();

                        long init = System.currentTimeMillis();
                        while (!stopRequested()) {
                            sleep(10);
                            //sleep((long) sleepAmount);
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

                            float ratioStep;
                            if (numSteps != 0) {
                                ratioStep = (scaleZTarget - ratio) / (float) numSteps;
                                ratio += ratioStep;
                            } else {
                                ratio = 1.0f;
                            }

                            float val = (float) Math.log(((ratio * 9) + 1)) / (float) Math.log(10.0f);

                            for (int x = 0; x < size; x++) {
                                for (int y = 0; y < size; y++) {
                                    float diffHeight = (newHeightValues[x][y] - oldHeightValues[x][y]);
                                    float diffColor = (newColorValues[x][y] - oldColorValues[x][y]);
                                    curHeightValues[x][y] = oldHeightValues[x][y] + (val * diffHeight);
                                    curColorValues[x][y] = oldColorValues[x][y] + (val * diffColor);
                                }
                            }
                            barPlot.setValues(curHeightValues, curColorValues);

                            if (fromZero) {
                                barPlot.setScaleZ(val);
                            }
                            if (ratio >= 1.0f) {
                                copyMatrix(curHeightValues, newHeightValues, size);
                                copyMatrix(curColorValues, newColorValues, size);
                                barPlot.setValues(newHeightValues, newColorValues);
                                barPlot.setScaleZ(1.0f);
                                visRenderer.redraw();
                                return;
                            }
                        }
                    } catch (InterruptedException ex) {}
                }
            };
            animator.start();
        }
    }

    private JPanel buildMapPanel() {

        // Create the visRenderer and register it with the canvas
        visRenderer = new VisRenderer();

        // Create the canvas
        visCanvas = new VisCanvas(visRenderer);
        visCanvas.getActualCanvas().setSize(9100, 9100);
        visRenderer.setVisCanvasListener(this);
        colorScale = new ColorScale();

        axes = new Axes();

        int size = mapData.getSize();

        newHeightValues = new float[size][size];
        newColorValues = new float[size][size];

        oldHeightValues = new float[size][size];
        oldColorValues = new float[size][size];

        curHeightValues = new float[size][size];
        curColorValues = new float[size][size];

        barPlot = new BarPlot(axes, colorScale);

        // We make the 3d comm matrix a little bit more squished than normal
        barPlot.setValues(18, 18, 8, oldHeightValues, oldColorValues);

        processData();

        // point at the center of the plot
        visRenderer.setAim(new Vec(5, 5, 0));

        // Add the drawable objects to the visRenderer (the scatterPlot will draw the axes)
        //visRenderer.addShape(scatterPlot);
        visRenderer.addShape(barPlot);
        visRenderer.addShape(colorScale);

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
            if (eventSrc.equals(this.pathSelector)) {
                String newPath = (String) this.pathSelector.getSelectedItem();
                if (!newPath.equals(currentPath)) {
                    currentPath = newPath;
                    redrawHeatMap();
                }
            }
            if (eventSrc.equals(this.heightComboBox)) {
                if (heightComboBox.getSelectedIndex() != heightMetric) {
                    heightMetric = heightComboBox.getSelectedIndex();
                    redrawHeatMap();
                }
            }
            if (eventSrc.equals(this.colorComboBox)) {
                if (colorComboBox.getSelectedIndex() != colorMetric) {
                    colorMetric = colorComboBox.getSelectedIndex();
                    redrawHeatMap();
                }
            }
        } catch (Exception e) {
            ParaProfUtils.handleException(e);
        }
    }

    private void redrawHeatMap() {
        // processData can't be run on the AWT thread, it will wait for a possible
        // animator thread, which uses JOGL, which always runs on th AWT thread
        java.lang.Thread thread = new java.lang.Thread(new Runnable() {
            public void run() {
                processData();
            }
        });
        thread.start();

    }

   

    public void createNewCanvas() {

        visCanvas = new VisCanvas(visRenderer);
        visCanvas.getActualCanvas().setSize(9900, 9900);
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

    public Component getComponent() {
        return this;
    }

    public BufferedImage getImage() {
        return visRenderer.createScreenShot();
    }

    public int print(Graphics graphics, PageFormat pageFormat, int pageIndex) throws PrinterException {
        try {
            if (pageIndex >= 1) {
                return NO_SUCH_PAGE;
            }

            ParaProfUtils.scaleForPrint(graphics, pageFormat, visCanvas.getWidth(), visCanvas.getHeight());

            BufferedImage screenShot = visRenderer.createScreenShot();

            ImageObserver imageObserver = new ImageObserver() {
                public boolean imageUpdate(Image image, int a, int b, int c, int d, int e) {
                    return false;
                }
            };

            graphics.drawImage(screenShot, 0, 0, Color.black, imageObserver);

            return Printable.PAGE_EXISTS;
        } catch (Exception e) {
            ParaProfUtils.handleException(e);
            return NO_SUCH_PAGE;
        }
    }

}
