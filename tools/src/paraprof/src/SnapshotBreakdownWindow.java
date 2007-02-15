package edu.uoregon.tau.paraprof;

import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.List;
import java.util.Observable;
import java.util.Observer;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.Range;
import org.jfree.data.xy.DefaultTableXYDataset;
import org.jfree.data.xy.XYSeries;

import edu.uoregon.tau.paraprof.interfaces.ParaProfWindow;
import edu.uoregon.tau.perfdmf.FunctionProfile;
import edu.uoregon.tau.perfdmf.Snapshot;
import edu.uoregon.tau.perfdmf.Thread;

public class SnapshotBreakdownWindow extends JFrame implements ActionListener, Observer, ChangeListener, ParaProfWindow {

    private ParaProfTrial ppTrial;
    private Thread thread;

    //private BarChartPanel panel;
    //private AbstractBarChartModel model;
    private ChartPanel panel;
    private DataSorter dataSorter;

    private static int defaultWidth = 750;
    private static int defaultHeight = 610;

    private JMenu optionsMenu;
    private JCheckBoxMenuItem nameCheckBox = new JCheckBoxMenuItem("Sort By Name", false);
    private JCheckBoxMenuItem normalizeCheckBox = new JCheckBoxMenuItem("Normalize Bars", true);
    private JCheckBoxMenuItem orderByMeanCheckBox = new JCheckBoxMenuItem("Order By Mean", true);
    private JCheckBoxMenuItem orderCheckBox = new JCheckBoxMenuItem("Descending Order", true);
    private JCheckBoxMenuItem stackBarsCheckBox = new JCheckBoxMenuItem("Stack Bars Together", true);
    private JCheckBoxMenuItem metaDataCheckBox;

    private DefaultTableXYDataset dataSet = new DefaultTableXYDataset();

    private boolean timeline = true;
    private boolean differential = true;
    private boolean middleTime = true;
    private boolean square = true;
    private boolean topTen = true;

    private final static String ST_TOP_TEN = "Top Ten";
    private final static String ST_DIFFERENTIAL = "Differential";
    private final static String ST_SQUARE = "Square";
    private final static String ST_TIMELINE = "Timeline";

    private JFreeChart chart;

    private JToggleButton button_topTen = new JToggleButton(ST_TOP_TEN, topTen);
    private JToggleButton button_square = new JToggleButton(ST_SQUARE, square);
    private JToggleButton button_timeline = new JToggleButton(ST_TIMELINE, square);
    private JToggleButton button_differential = new JToggleButton(ST_DIFFERENTIAL, differential);

    private final static int topNum = 10;
    
    public SnapshotBreakdownWindow(ParaProfTrial ppTrial, Thread thread, Component owner) {
        this.ppTrial = ppTrial;
        this.thread = thread;

        PPThread ppThread = new PPThread(thread, ppTrial);

        this.setTitle("TAU: ParaProf: Snapshots for " + ppThread.getFullName() + " - "
                + ppTrial.getTrialIdentifier(ParaProf.preferences.getShowPathTitleInReverse()));

        dataSorter = new DataSorter(ppTrial);
        //model = new ThreadSnapshotBarChartModel(this, dataSorter, ppTrial, thread);
        //panel = new BarChartPanel(model);
        //      panel.getBarChart().setLeftJustified(true);
        //      panel.getBarChart().setAutoResize(true);
        //      panel.getVerticalScrollBar().setUnitIncrement(35);
        //        getContentPane().add(panel);

        ActionListener toolbarListener = new ActionListener() {

            public void actionPerformed(ActionEvent e) {
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

        };

        JToolBar bar = new JToolBar();

        button_topTen.addActionListener(toolbarListener);
        button_square.addActionListener(toolbarListener);
        button_differential.addActionListener(toolbarListener);
        button_timeline.addActionListener(toolbarListener);
        bar.add(button_topTen);
        bar.add(button_square);
        bar.add(button_differential);
        bar.add(button_timeline);

        //getContentPane().setLayout(new GridBagLayout());
        getContentPane().add(bar, BorderLayout.NORTH);

        chart = createChart();

        panel = new ChartPanel(chart);
        getContentPane().add(panel);

        setSize(ParaProfUtils.checkSize(new Dimension(defaultWidth, defaultHeight)));
        setLocation(WindowPlacer.getNewLocation(this, owner));

        //        pack();
        setupMenus();

        ParaProfUtils.setFrameIcon(this);
    }

    private JFreeChart createChart() {
        dataSet = new DefaultTableXYDataset();
        dataSet.removeAllSeries();
        dataSorter.setDescendingOrder(true);
        List snapshots = thread.getSnapshots();
        List functions = dataSorter.getBasicFunctionProfiles(thread);

        long firstTime = ((Snapshot) snapshots.get(0)).getTimestamp();
        long lastTime = firstTime;

        int max = functions.size();
        if (topTen) {
            max = Math.min(topNum+1, functions.size());
        }

        for (int y = 0; y < max; y++) {
            FunctionProfile fp = (FunctionProfile) functions.get(y);

            XYSeries s;
            
            if (topTen && y == topNum) {
                s = new XYSeries("Other", true, false);
            } else {
                s = new XYSeries(fp.getName(), true, false);
            }
            for (int x = 1; x < snapshots.size() - 1; x++) {
                int snapshotID = x;
                double value;

                if (topTen && y == topNum) {
                    value = 0;
                    for (int z = y; z < functions.size(); z++) {
                        FunctionProfile f = (FunctionProfile) functions.get(z);
                        if (differential && snapshotID != 0) {
                            value += f.getExclusive(snapshotID, ppTrial.getDefaultMetricID())
                                    - f.getExclusive(snapshotID - 1, ppTrial.getDefaultMetricID());
                        } else {
                            value += f.getExclusive(snapshotID, ppTrial.getDefaultMetricID());
                        }
                    }

                } else {

                    if (differential && snapshotID != 0) {
                        value = fp.getExclusive(snapshotID, ppTrial.getDefaultMetricID())
                                - fp.getExclusive(snapshotID - 1, ppTrial.getDefaultMetricID());
                    } else {
                        value = fp.getExclusive(snapshotID, ppTrial.getDefaultMetricID());
                    }
                }

                Snapshot snapshot = (Snapshot) snapshots.get(x);
                long time = snapshot.getTimestamp() - firstTime;
                lastTime = time;

                if (timeline) {
                    if (square) {
                        Snapshot last = (Snapshot) snapshots.get(x - 1);
                        long prevTime = last.getTimestamp() - firstTime;
                        s.add(0.0001 + (double) (prevTime) / 1000000, value);

                        s.add((double) (time) / 1000000, value);

                    } else if (middleTime) {
                        Snapshot last = (Snapshot) snapshots.get(x - 1);

                        double bobtime = time - ((snapshot.getTimestamp() - last.getTimestamp()) / 2);
                        s.add((double) (bobtime) / 1000000, value);

                    } else {
                        s.add((double) (time) / 1000000, value);
                    }
                } else {
                    if (square) {
                        s.add(x - 0.9999, value);
                        s.add(x, value);
                    } else {
                        s.add(x, value);
                    }
                }

            }
            dataSet.addSeries(s);
        }

        JFreeChart chart = ChartFactory.createStackedXYAreaChart("Snapshot Breakdown", "Snapshots", // domain axis label
                //JFreeChart chart = ChartFactory.createXYLineChart("Snapshot Breakdown", "Snapshots", // domain axis label
                //JFreeChart chart = ChartFactory.createXYAreaChart("Snapshot Breakdown", "Snapshots", // domain axis label
                "Exclusive value (microseconds)", // range axis label
                dataSet, // data
                PlotOrientation.VERTICAL, // the plot orientation
                true, // legend
                true, // tooltips
                false // urls
        );

        //chart.getPlot().

        //chart.getPlot().setForegroundAlpha(0.65f);
        //chart.getPlot().setForegroundAlpha(0.4f);
        //

        if (timeline) {
            XYPlot plot = chart.getXYPlot();
            NumberAxis axis = new NumberAxis("Timeline (seconds)");
            axis.setRange(new Range(0, (double) lastTime / 1000000));
            plot.setDomainAxis(0, axis);
        }

        return chart;
        //ChartFactory.createStackedAreaXYChart()

    }

    private void setupMenus() {
        JMenuBar mainMenu = new JMenuBar();

        optionsMenu = new JMenu("Options");

        metaDataCheckBox = new JCheckBoxMenuItem("Show Meta Data in Panel", true);
        metaDataCheckBox.addActionListener(this);
        optionsMenu.add(metaDataCheckBox);

        optionsMenu.add(new JSeparator());

        nameCheckBox.addActionListener(this);
        optionsMenu.add(nameCheckBox);

        normalizeCheckBox.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                setNormalized(normalizeCheckBox.isSelected());
            }
        });
        optionsMenu.add(normalizeCheckBox);

        orderByMeanCheckBox.addActionListener(this);
        optionsMenu.add(orderByMeanCheckBox);

        orderCheckBox.addActionListener(this);
        optionsMenu.add(orderCheckBox);

        stackBarsCheckBox.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                setStackBars(stackBarsCheckBox.isSelected());
            }
        });
        optionsMenu.add(stackBarsCheckBox);

        optionsMenu.add(new JSeparator());

        //sJCheckBoxMenuItem areaBox = new JCheckBoxMenuItem("Areatrue);

        //optionsMenu.add(new)

        mainMenu.add(ParaProfUtils.createFileMenu(this, panel, panel));
        mainMenu.add(optionsMenu);
        mainMenu.add(ParaProfUtils.createWindowsMenu(ppTrial, this));
        if (ParaProf.scripts.size() > 0) {
            mainMenu.add(ParaProfUtils.createScriptMenu(ppTrial, this));
        }
        mainMenu.add(ParaProfUtils.createHelpMenu(this, this));

        setJMenuBar(mainMenu);

    }

    public void actionPerformed(ActionEvent e) {
    // TODO Auto-generated method stub

    }

    public void update(Observable o, Object arg) {
    // TODO Auto-generated method stub

    }

    public void stateChanged(ChangeEvent e) {
    // TODO Auto-generated method stub

    }

    public void closeThisWindow() {
    // TODO Auto-generated method stub

    }

    public void help(boolean display) {
    // TODO Auto-generated method stub

    }

    public boolean getNormalized() {
        return normalizeCheckBox.isSelected();
    }

    public void setNormalized(boolean value) {
    //        normalizeCheckBox.setSelected(value);
    //        panel.getBarChart().setNormalized(normalizeCheckBox.isSelected());
    //        panel.repaint();
    }

    public void setStackBars(boolean value) {
    //        stackBarsCheckBox.setSelected(value);
    //
    //        if (value) {
    //            normalizeCheckBox.setEnabled(true);
    //            orderByMeanCheckBox.setEnabled(true);
    //
    //            panel.getBarChart().setNormalized(getNormalized());
    //            panel.getBarChart().setStacked(true);
    //
    //        } else {
    //            normalizeCheckBox.setSelected(false);
    //            normalizeCheckBox.setEnabled(false);
    //            orderByMeanCheckBox.setSelected(true);
    //            orderByMeanCheckBox.setEnabled(false);
    //
    //            panel.getBarChart().setNormalized(getNormalized());
    //            panel.getBarChart().setStacked(false);
    //        }
    //
    //        panel.repaint();
    }

}
