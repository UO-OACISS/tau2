package edu.uoregon.tau.paraprof;

import java.awt.*;
import java.awt.event.*;
import java.util.*;
import java.util.List;

import javax.swing.*;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.Range;
import org.jfree.data.xy.DefaultTableXYDataset;
import org.jfree.data.xy.XYSeries;

import edu.uoregon.tau.common.ImageExport;
import edu.uoregon.tau.paraprof.interfaces.ParaProfWindow;
import edu.uoregon.tau.paraprof.interfaces.ToolBarListener;
import edu.uoregon.tau.paraprof.util.ObjectFilter;
import edu.uoregon.tau.perfdmf.FunctionProfile;
import edu.uoregon.tau.perfdmf.Snapshot;
import edu.uoregon.tau.perfdmf.Thread;

public class SnapshotBreakdownWindow extends JFrame implements Observer, ParaProfWindow, ImageExport, ToolBarListener {

    private ParaProfTrial ppTrial;
    private Thread thread;

    private ChartPanel panel;
    private DataSorter dataSorter;

    private static int defaultWidth = 750;
    private static int defaultHeight = 610;

    private DefaultTableXYDataset dataSet = new DefaultTableXYDataset();

    private boolean timeline = true;
    private boolean differential = true;
    private boolean middleTime = true;
    private boolean square = true;
    private boolean topTen = true;

    private final static String ST_TOP_TEN = "Top 20";
    private final static String ST_DIFFERENTIAL = "Differential";
    private final static String ST_SQUARE = "Square";
    private final static String ST_TIMELINE = "Timeline";

    private final static int STYLE_STACKED = 0;
    private final static int STYLE_TRANSPARENT_AREA = 1;
    private final static int STYLE_LINE = 2;

    private final static String ST_STYLE_STACKED = "Stacked";
    private final static String ST_STYLE_TRANSPARENT_AREA = "Area";
    private final static String ST_STYLE_LINE = "Line";

    private JFreeChart chart;

    private JToggleButton button_topTen = new JToggleButton(ST_TOP_TEN, topTen);
    private JToggleButton button_square = new JToggleButton(ST_SQUARE, square);
    private JToggleButton button_timeline = new JToggleButton(ST_TIMELINE, square);
    private JToggleButton button_differential = new JToggleButton(ST_DIFFERENTIAL, differential);

    private final static int topNum = 20;
    private ObjectFilter filter;
    private int style = STYLE_STACKED;

    public SnapshotBreakdownWindow(ParaProfTrial ppTrial, Thread thread, Component owner) {
        this.ppTrial = ppTrial;
        this.thread = thread;

        PPThread ppThread = new PPThread(thread, ppTrial);

        filter = new ObjectFilter(thread.getSnapshots());
        filter.addObserver(this);
        
        ppTrial.addObserver(this);

        this.setTitle("TAU: ParaProf: Snapshots for " + ppThread.getFullName() + " - "
                + ppTrial.getTrialIdentifier(ParaProf.preferences.getShowPathTitleInReverse()));

        addWindowListener(new WindowAdapter() {
            public void windowClosing(WindowEvent evt) {
                closeThisWindow();
            }
        });

        dataSorter = new DataSorter(ppTrial);
        //model = new ThreadSnapshotBarChartModel(this, dataSorter, ppTrial, thread);
        //panel = new BarChartPanel(model);
        //      panel.getBarChart().setLeftJustified(true);
        //      panel.getBarChart().setAutoResize(true);
        //      panel.getVerticalScrollBar().setUnitIncrement(35);
        //        getContentPane().add(panel);

        createToolbar();
        chart = createChart();

        panel = new ChartPanel(chart);
        getContentPane().add(panel);

        setSize(ParaProfUtils.checkSize(new Dimension(defaultWidth, defaultHeight)));
        setLocation(WindowPlacer.getNewLocation(this, owner));

        //        pack();
        setupMenus();

        ParaProfUtils.setFrameIcon(this);
        ParaProf.incrementNumWindows();

    }

    private void recreateChart() {
        topTen = button_topTen.isSelected();
        square = button_square.isSelected();
        timeline = button_timeline.isSelected();
        differential = button_differential.isSelected();

        getContentPane().remove(panel);
        chart = createChart();
        panel = new ChartPanel(chart);
        getContentPane().add(panel);

        // redraw
        validate();
        repaint();
    }

    private void createToolbar() {

        ActionListener toolbarListener = new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                recreateChart();
            }
        };

        Vector styleList = new Vector();
        styleList.add(ST_STYLE_STACKED);
        styleList.add(ST_STYLE_TRANSPARENT_AREA);
        styleList.add(ST_STYLE_LINE);
        final JComboBox styleBox = new JComboBox(styleList);
        styleBox.setMaximumSize(styleBox.getPreferredSize());

        styleBox.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                setStyle(styleBox.getSelectedIndex());
                toolBarUsed();
            }
        });

        button_topTen.addActionListener(toolbarListener);
        button_square.addActionListener(toolbarListener);
        button_differential.addActionListener(toolbarListener);
        button_timeline.addActionListener(toolbarListener);

        JToolBar bar = new JToolBar();

        bar.add(button_topTen);
        bar.add(button_square);
        bar.add(button_differential);
        bar.add(button_timeline);

        bar.addSeparator();
        bar.add(styleBox);

        bar.addSeparator();
        ParaProfUtils.createMetricToolbarItems(bar, ppTrial, dataSorter, this);
        bar.addSeparator();

        getContentPane().add(bar, BorderLayout.NORTH);
    }

    public void setStyle(int style) {
        this.style = style;
    }

    private JFreeChart createChart() {
        dataSet = new DefaultTableXYDataset();
        dataSet.removeAllSeries();
        dataSorter.setDescendingOrder(true);
        List snapshots = thread.getSnapshots();
        
        // override the trial's selected snapshot with the final one (-1)
        // so that the sort order and list of functions does not vary based on the
        // slider in the snapshot controller
        dataSorter.setSelectedSnapshotOverride(true);
        dataSorter.setSelectedSnapshot(-1);
        
        List functions = dataSorter.getBasicFunctionProfiles(thread);

        //long firstTime = ((Snapshot) snapshots.get(0)).getTimestamp();
        long firstTime = thread.getStartTime();
        long duration = 0;

        int max = functions.size();
        if (topTen) {
            max = Math.min(topNum + 1, functions.size());
        }

        for (int y = 0; y < max; y++) {
            FunctionProfile fp = (FunctionProfile) functions.get(y);

            XYSeries s;

            if (topTen && y == topNum) {
                s = new XYSeries("Other", true, false);
            } else {
                String str = fp.getName();
                while (str.indexOf("[{") != -1) {
                    int a = str.indexOf("[{");
                    int b = str.indexOf("}]");
                    str = str.substring(0, a) + str.substring(b + 2);
                }
                s = new XYSeries(str, true, false);
            }

            int start = 0;
            int stop = snapshots.size();

            List filteredSnapshots = filter.getFilteredObjects();
            for (int i = 0; i < filteredSnapshots.size(); i++) {
                int x = ((Snapshot) filteredSnapshots.get(i)).getID();

                int snapshotID = x;
                double value;

                if (topTen && y == topNum) {
                    value = 0;
                    for (int z = y; z < functions.size(); z++) {
                        FunctionProfile f = (FunctionProfile) functions.get(z);

                        if (differential && snapshotID != 0) {
                            value += dataSorter.getValue(f, snapshotID) - dataSorter.getValue(f, snapshotID - 1);
                        } else {
                            value += dataSorter.getValue(f, snapshotID);
                        }

                        //                        if (differential && snapshotID != 0) {
                        //                            value += f.getExclusive(snapshotID, ppTrial.getDefaultMetricID())
                        //                                    - f.getExclusive(snapshotID - 1, ppTrial.getDefaultMetricID());
                        //                        } else {
                        //                            value += f.getExclusive(snapshotID, ppTrial.getDefaultMetricID());
                        //                        }
                    }

                } else {

                    if (differential && snapshotID != 0) {
                        value = dataSorter.getValue(fp, snapshotID) - dataSorter.getValue(fp, snapshotID - 1);
                    } else {
                        value = dataSorter.getValue(fp, snapshotID);
                    }

                    //                    if (differential && snapshotID != 0) {
                    //                        value = fp.getExclusive(snapshotID, ppTrial.getDefaultMetricID())
                    //                                - fp.getExclusive(snapshotID - 1, ppTrial.getDefaultMetricID());
                    //                    } else {
                    //                        value = fp.getExclusive(snapshotID, ppTrial.getDefaultMetricID());
                    //                    }
                }

                Snapshot snapshot = (Snapshot) snapshots.get(x);
                long time = snapshot.getTimestamp() - firstTime;
                duration = time;

                if (timeline) {
                    long lastTime;
                    long prevTime;
                    if (x == 0) {
                        lastTime = firstTime;
                        prevTime = 0;
                    } else {
                        Snapshot last = (Snapshot) snapshots.get(x - 1);
                        lastTime = last.getTimestamp();
                        prevTime = last.getTimestamp() - firstTime;
                    }
                    if (square) {
                        s.add(0.0001 + (double) (prevTime) / 1000000, value);

                        s.add((double) (time) / 1000000, value);

                    } else if (middleTime) {

                        double bobtime = time - ((snapshot.getTimestamp() - lastTime) / 2);
                        s.add((double) (bobtime) / 1000000, value);

                    } else {
                        s.add((double) (time) / 1000000, value);
                    }
                } else {
                    if (square) {
                        s.add(x, value);
                        s.add(x + 0.9999, value);
                    } else {
                        s.add(x, value);
                    }
                }

            }
            dataSet.addSeries(s);
        }

        ParaProfMetric ppMetric = (ParaProfMetric)dataSorter.getSelectedMetric();
        int units = dataSorter.getValueType().getUnits(0, ppMetric);
        String suffix = dataSorter.getValueType().getSuffix(units, ppMetric).trim();

        String yAxisLabel = dataSorter.getValueType() + " (" + suffix + ")";

        JFreeChart chart = null;
        if (style == STYLE_STACKED) {
            chart = ChartFactory.createStackedXYAreaChart("Snapshot Breakdown", "Snapshots", // domain axis label
                    yAxisLabel, // range axis label
                    dataSet, // data
                    PlotOrientation.VERTICAL, // the plot orientation
                    true, // legend
                    true, // tooltips
                    false // urls
            );

        } else if (style == STYLE_TRANSPARENT_AREA) {
            chart = ChartFactory.createXYAreaChart("Snapshot Breakdown", "Snapshots", // domain axis label
                    yAxisLabel, // range axis label
                    dataSet, // data
                    PlotOrientation.VERTICAL, // the plot orientation
                    true, // legend
                    true, // tooltips
                    false // urls
            );
            //chart.getPlot().setForegroundAlpha(0.65f);
            chart.getPlot().setForegroundAlpha(0.4f);

        } else if (style == STYLE_LINE) {
            chart = ChartFactory.createXYLineChart("Snapshot Breakdown", "Snapshots", // domain axis label
                    yAxisLabel, // range axis label
                    dataSet, // data
                    PlotOrientation.VERTICAL, // the plot orientation
                    true, // legend
                    true, // tooltips
                    false // urls
            );

        } else {
            throw new ParaProfException("Unrecognized style: " + style);
        }

        if (timeline) {
            XYPlot plot = chart.getXYPlot();
            NumberAxis axis = new NumberAxis("Timeline (seconds)");
            axis.setRange(new Range(0, (double) duration / 1000000));
            plot.setDomainAxis(0, axis);
        } else {
            XYPlot plot = chart.getXYPlot();
            NumberAxis axis = new NumberAxis("Snapshots");
            axis.setRange(new Range(0, (double) snapshots.size()));
            plot.setDomainAxis(0, axis);
        }

        return chart;
    }

    private void setupMenus() {
        JMenuBar mainMenu = new JMenuBar();

        //sJCheckBoxMenuItem areaBox = new JCheckBoxMenuItem("Areatrue);

        JMenu filterMenu = new JMenu("Filter");
        JMenuItem filterSnapshots = new JMenuItem("Filter Snapshots");
        filterSnapshots.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                filter.showFrame("Filter Snapshots");
            }
        });
        filterMenu.add(filterSnapshots);
        //optionsMenu.add(new)

        mainMenu.add(ParaProfUtils.createFileMenu(this, panel, this));
        //mainMenu.add(optionsMenu);
        mainMenu.add(filterMenu);
        mainMenu.add(ParaProfUtils.createWindowsMenu(ppTrial, this));
        if (ParaProf.scripts.size() > 0) {
            mainMenu.add(ParaProfUtils.createScriptMenu(ppTrial, this));
        }
        mainMenu.add(ParaProfUtils.createHelpMenu(this, this));

        setJMenuBar(mainMenu);
    }

    public void update(Observable o, Object arg) {
        recreateChart();
    }

    public void closeThisWindow() {
        try {
            filter.closeWindow();
            setVisible(false);
            ppTrial.deleteObserver(this);
            ParaProf.decrementNumWindows();

        } catch (Exception e) {
            // do nothing
            e.printStackTrace();
        }
        dispose();

    }

    public void help(boolean display) {
    // TODO Auto-generated method stub
    }

    public void export(Graphics2D g2D, boolean toScreen, boolean fullWindow, boolean drawHeader) {
        panel.setDoubleBuffered(false);
        panel.paintAll(g2D);
        panel.setDoubleBuffered(true);
    }

    public Dimension getImageSize(boolean fullScreen, boolean header) {
        return panel.getSize();
    }

    public void toolBarUsed() {
        recreateChart();
    }

    public JFrame getFrame() {
        return this;
    }

}
