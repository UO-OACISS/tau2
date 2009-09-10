package edu.uoregon.tau.paraprof;

import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.awt.image.ImageObserver;
import java.awt.print.PageFormat;
import java.awt.print.Printable;
import java.awt.print.PrinterException;
import java.net.URL;
import java.util.*;
import java.util.List;

import javax.swing.*;

import edu.uoregon.tau.common.StoppableThread;
import edu.uoregon.tau.common.Utility;
import edu.uoregon.tau.paraprof.interfaces.ParaProfWindow;
import edu.uoregon.tau.perfdmf.*;
import edu.uoregon.tau.perfdmf.Thread;
import edu.uoregon.tau.vis.*;

/**
 * 3D Communication Matrix Window 
 * 
 * <P>CVS $Id: ThreeDeeCommMatrixWindow.java,v 1.5 2009/09/10 00:13:49 amorris Exp $</P>
 * @author Alan Morris, Kevin Huck
 * @version $Revision: 1.5 $
 */
public class ThreeDeeCommMatrixWindow extends JFrame implements ParaProfWindow, ActionListener, ThreeDeeImageProvider,
        VisCanvasListener, Printable {

    private ParaProfTrial ppTrial;
    private SteppedComboBox pathSelector;
    private SteppedComboBox heightComboBox;
    private SteppedComboBox colorComboBox;
    private JSplitPane splitPane;

    private HeatMapData mapData;
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

    private List threadNames;

    // 3D elements
    private BarPlot barPlot;
    private Axes axes;
    private VisRenderer visRenderer;
    private VisCanvas visCanvas;
    private ColorScale colorScale;

    // old, cur, new are used for animation so we know where we came from, where we are, and where we are going
    private float oldHeightValues[][], oldColorValues[][];
    private float curHeightValues[][], curColorValues[][];
    private float newHeightValues[][], newColorValues[][];

    private StoppableThread animator;

    public ThreeDeeCommMatrixWindow(String title, HeatMapData mapData, ParaProfTrial ppTrial, Component invoker) {
        this.ppTrial = ppTrial;
        this.mapData = mapData;

        buildPanels();

        // handle close window event
        addWindowListener(new WindowAdapter() {
            public void windowClosing(WindowEvent e) {
                closeThisWindow();
            }
        });

        JMenuBar mainMenu = new JMenuBar();

        JMenu fileMenu = ParaProfUtils.createFileMenu(this, this, this);
        fileMenu.getPopupMenu().setLightWeightPopupEnabled(false);
        mainMenu.add(fileMenu);

        JMenu windowsMenu = ParaProfUtils.createWindowsMenu(ppTrial, this);
        windowsMenu.getPopupMenu().setLightWeightPopupEnabled(false);
        mainMenu.add(windowsMenu);

        JMenu helpMenu = ParaProfUtils.createHelpMenu(this, this);
        helpMenu.getPopupMenu().setLightWeightPopupEnabled(false);
        mainMenu.add(helpMenu);

        setJMenuBar(mainMenu);

        //Set the help window text if required.
        if (ParaProf.getHelpWindow().isVisible()) {
            help(false);
        }

        ParaProf.incrementNumWindows();

        setSize(ParaProfUtils.checkSize(new Dimension(1000, 750)));
        setSize(new Dimension(1000, 750));
        this.setLocation(WindowPlacer.getNewLocation(this, invoker));

        ParaProfUtils.setFrameIcon(this);
        this.setTitle("TAU: ParaProf: 3D Communication Matrix: "
                + ppTrial.getTrialIdentifier(ParaProf.preferences.getShowPathTitleInReverse()));
    }

    public static ThreeDeeCommMatrixWindow createCommunicationMatrixWindow(ParaProfTrial ppTrial, JFrame parentFrame) {

        HeatMapData mapData = generateData(ppTrial.getDataSource(), parentFrame);
        if (mapData == null) {
            return null;
        }

        ThreeDeeCommMatrixWindow window = new ThreeDeeCommMatrixWindow("3D Communication Matrix", mapData, ppTrial, parentFrame);

        return window;
    }

    private void buildPanels() {
        JPanel graphicsPanel = buildGraphicsPanel();

        // Create the 3d control panel
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
        splitPane = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT, graphicsPanel, rightPanel);
        splitPane.setResizeWeight(1.0);
        splitPane.setOneTouchExpandable(true);
        splitPane.setContinuousLayout(true);

        this.getContentPane().add(splitPane);
    }

    public void closeThisWindow() {
        setVisible(false);

        ParaProf.decrementNumWindows();

        if (barPlot != null) {
            barPlot.cleanUp();
        }

        if (visRenderer != null) {
            visRenderer.cleanUp();
        }

        visRenderer = null;

        dispose();
        ParaProf.decrementNumWindows();
    }

    public void help(boolean display) {
        ParaProf.getHelpWindow().clearText();
        if (display) {
            ParaProf.getHelpWindow().setVisible(true);
        }
        ParaProf.getHelpWindow().writeText("3D Communication Matrix Window");
        ParaProf.getHelpWindow().writeText("");
        ParaProf.getHelpWindow().writeText("This window shows communication data between nodes.");
    }

    private static HeatMapData generateData(DataSource dataSource, JFrame mainFrame) {
        boolean foundData = false;
        int threadID = 0;
        int size = dataSource.getAllThreads().size();
        // declare the heatmap data object
        HeatMapData mapData = new HeatMapData(size);

        for (Iterator it = dataSource.getAllThreads().iterator(); it.hasNext();) {
            Thread thread = (Thread) it.next();
            for (Iterator it2 = thread.getUserEventProfiles(); it2.hasNext();) {
                UserEventProfile uep = (UserEventProfile) it2.next();
                if (uep != null && uep.getNumSamples() > 0) {
                    String event = uep.getName();
                    if (event.startsWith("Message size sent to node ") && event.indexOf("=>") == -1) {
                        foundData = true;
                        // split the string
                        extractData(mapData, uep, threadID, event, event, allPaths);
                    } else if (event.startsWith("Message size sent to node ") && event.indexOf("=>") >= 0) {
                        foundData = true;
                        StringTokenizer st = new StringTokenizer(event, ":");
                        String first = st.nextToken().trim();
                        String path = st.nextToken().trim();
                        // now, split up the path, and handle each node 
                        StringTokenizer st2 = new StringTokenizer(path, "=>");
                        String tmp = null;
                        while (st2.hasMoreTokens()) {
                            tmp = st2.nextToken().trim();
                            extractData(mapData, uep, threadID, event, first, tmp);
                        }
                    }
                }
            }
            threadID++;
        }
        if (!foundData) {
            JOptionPane.showMessageDialog(
                    mainFrame,
                    "This trial does not have communication matrix data.\nTo collect communication matrix data, set the environment variable TAU_COMM_MATRIX=1 before executing your application.",
                    "No Communication Matrix Data", JOptionPane.ERROR_MESSAGE);
            return null;
        }
        mapData.massageData();
        return mapData;
    }

    private static void extractData(HeatMapData mapData, UserEventProfile uep, int thread, String event, String first, String path) {
        double numEvents, eventMax, eventMin, eventMean, eventSumSqr, stdev, volume = 0;
        double[] empty = { 0, 0, 0, 0, 0, 0 };

        StringTokenizer st = new StringTokenizer(first, "Message size sent to node ");
        if (st.hasMoreTokens()) {
            int receiver = Integer.parseInt(st.nextToken());

            double[] pointData = mapData.get(thread, receiver, path);
            if (pointData == null) {
                pointData = empty;
            }

            numEvents = uep.getNumSamples();
            pointData[CALLS] += numEvents;

            eventMax = uep.getMaxValue();
            pointData[MAX] = Math.max(eventMax, pointData[MAX]);

            eventMin = uep.getMinValue();
            if (pointData[MIN] > 0) {
                pointData[MIN] = Math.min(pointData[MIN], eventMin);
            } else {
                pointData[MIN] = eventMin;
            }

            // we'll recompute this later.
            eventMean = uep.getMeanValue();
            pointData[MEAN] += eventMean;

            // we'll recompute this later.
            eventSumSqr = uep.getStdDev();
            pointData[STDDEV] += eventSumSqr;

            volume = numEvents * eventMean;
            pointData[VOLUME] += volume;
            mapData.put(thread, receiver, path, pointData);
        }
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

        pathSelector = new SteppedComboBox(mapData.getPaths().toArray());
        heightComboBox = new SteppedComboBox(metricStrings);
        colorComboBox = new SteppedComboBox(metricStrings);
        colorComboBox.setWidth(50);
        pathSelector.setWidth(50);
        heightComboBox.setWidth(50);

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

    private JPanel buildGraphicsPanel() {

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

        threadNames = ppTrial.getThreadNames();

        barPlot = new BarPlot(axes, colorScale);

        // We make the 3d comm matrix a little bit more squished than normal
        barPlot.setValues(18, 18, 8, oldHeightValues, oldColorValues);

        processData();

        // point at the center of the plot
        visRenderer.setAim(new Vec(5, 5, 0));

        // Add the drawable objects to the visRenderer (the scatterPlot will draw the axes)
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
            if (eventSrc.equals(pathSelector)) {
                String newPath = (String) pathSelector.getSelectedItem();
                if (!newPath.equals(currentPath)) {
                    currentPath = newPath;
                    redrawHeatMap();
                }
            }
            if (eventSrc.equals(heightComboBox)) {
                if (heightComboBox.getSelectedIndex() != heightMetric) {
                    heightMetric = heightComboBox.getSelectedIndex();
                    redrawHeatMap();
                }
            }
            if (eventSrc.equals(colorComboBox)) {
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

    /**
     * Called when FSAA is turned on, we need a new canvas
     */
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

    public JFrame getFrame() {
        return this;
    }
}
