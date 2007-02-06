package edu.uoregon.tau.paraprof;

import java.awt.Component;
import java.awt.Dimension;
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
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.DefaultTableXYDataset;
import org.jfree.data.xy.XYSeries;

import edu.uoregon.tau.paraprof.interfaces.ParaProfWindow;
import edu.uoregon.tau.perfdmf.FunctionProfile;
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

        JFreeChart chart = createChart();

        panel = new ChartPanel(chart);
        getContentPane().add(panel);

        setSize(ParaProfUtils.checkSize(new Dimension(defaultWidth, defaultHeight)));
        setLocation(WindowPlacer.getNewLocation(this, owner));

        setupMenus();

        ParaProfUtils.setFrameIcon(this);
    }

    private JFreeChart createChart() {

        dataSet.removeAllSeries();
        dataSorter.setDescendingOrder(true);
        List snapshots = thread.getSnapshots();
        List list = dataSorter.getBasicFunctionProfiles(thread);

        for (int y = 0; y < list.size(); y++) {
            FunctionProfile fp = (FunctionProfile) list.get(y);
            XYSeries s = new XYSeries(fp.getName(), true, false);
            for (int x = 1; x < snapshots.size() - 1; x++) {
                int snapshotID = x;
                double value;
                int differential = 1;

                if (differential == 1 && snapshotID != 0) {
                    value = fp.getExclusive(snapshotID, ppTrial.getDefaultMetricID())
                            - fp.getExclusive(snapshotID - 1, ppTrial.getDefaultMetricID());
                } else {
                    value = fp.getExclusive(snapshotID, ppTrial.getDefaultMetricID());
                }

                s.add(x, value);
            }
            dataSet.addSeries(s);
        }

  
        
        //JFreeChart chart = ChartFactory.createStackedXYAreaChart("Snapshot Breakdown", "Snapshots", // domain axis label
        //JFreeChart chart = ChartFactory.createXYLineChart("Snapshot Breakdown", "Snapshots", // domain axis label
        JFreeChart chart = ChartFactory.createXYAreaChart("Snapshot Breakdown", "Snapshots", // domain axis label
                "Exclusive value (microseconds)", // range axis label
                dataSet, // data
                PlotOrientation.VERTICAL, // the plot orientation
                true, // legend
                true, // tooltips
                false // urls
        );
        
        //chart.getPlot().
        
        //chart.getPlot().setForegroundAlpha(0.65f);
        chart.getPlot().setForegroundAlpha(0.4f);

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
