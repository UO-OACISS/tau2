package edu.uoregon.tau.paraprof;

import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.print.PageFormat;
import java.awt.print.Printable;
import java.awt.print.PrinterException;
import java.util.List;
import java.util.Observable;
import java.util.Observer;

import javax.swing.*;
import javax.swing.event.*;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.axis.NumberTickUnit;
import org.jfree.chart.labels.XYToolTipGenerator;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.renderer.xy.XYBarRenderer;
import org.jfree.chart.renderer.xy.XYItemRenderer;
import org.jfree.data.statistics.HistogramDataset;
import org.jfree.data.xy.XYDataset;

import edu.uoregon.tau.common.ImageExport;
import edu.uoregon.tau.common.Utility;
import edu.uoregon.tau.paraprof.enums.ValueType;
import edu.uoregon.tau.paraprof.interfaces.ParaProfWindow;
import edu.uoregon.tau.paraprof.interfaces.UnitListener;
import edu.uoregon.tau.perfdmf.Function;
import edu.uoregon.tau.perfdmf.UtilFncs;

/**
 * HistogramWindow
 * This is the histogram window
 *  
 * <P>CVS $Id: HistogramWindow.java,v 1.8 2009/04/07 20:31:44 amorris Exp $</P>
 * @author  Robert Bell, Alan Morris
 * @version $Revision: 1.8 $
 * @see     HistogramWindowPanel
 */
public class HistogramWindow extends JFrame implements ActionListener, MenuListener, Observer, ChangeListener, ParaProfWindow,
        UnitListener, Printable, ImageExport {

    private ParaProfTrial ppTrial;
    private DataSorter dataSorter;
    private Function function;

    private ChartPanel chartPanel;

    // hold on to these two for 'menuSelected'
    private JMenu unitsSubMenu = null;

    private List data = null;

    private int units = ParaProf.preferences.getUnits();

    private JCheckBoxMenuItem slidersCheckBox = null;
    private JLabel numBinsLabel = new JLabel("Number of Bins");
    private JSlider numBinsSlider = new JSlider(0, 100, 10);
    private int numBins = 10;

    public HistogramWindow(ParaProfTrial ppTrial, Function function, Component invoker) {
        this.ppTrial = ppTrial;
        ppTrial.addObserver(this);

        this.dataSorter = new DataSorter(ppTrial);
        this.function = function;

        setTitle("TAU: ParaProf: Histogram: " + ppTrial.getTrialIdentifier(ParaProf.preferences.getShowPathTitleInReverse()));
        ParaProfUtils.setFrameIcon(this);
        setSize(ParaProfUtils.checkSize(new java.awt.Dimension(670, 630)));
        setLocation(WindowPlacer.getNewLocation(this, invoker));

        //Add some window listener code
        addWindowListener(new java.awt.event.WindowAdapter() {
            public void windowClosing(java.awt.event.WindowEvent evt) {
                thisWindowClosing(evt);
            }
        });

        //Set the help window text if required.
        if (ParaProf.getHelpWindow().isVisible()) {
            this.help(false);
        }

        sortLocalData();

        numBinsSlider.setPaintTicks(true);
        numBinsSlider.setMajorTickSpacing(50);
        numBinsSlider.setMinorTickSpacing(10);
        numBinsSlider.setPaintLabels(true);
        numBinsSlider.setSnapToTicks(false);
        numBinsSlider.addChangeListener(this);

        // set up the layout system
        getContentPane().setLayout(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);

        JFreeChart chart = createChart();

        chartPanel = new ChartPanel(chart);

        gbc.fill = GridBagConstraints.BOTH;
        gbc.anchor = GridBagConstraints.CENTER;
        gbc.weightx = 0.95;
        gbc.weighty = 0.98;
        addCompItem(chartPanel, gbc, 0, 0, 1, 1);

        setupMenus();

        ParaProf.incrementNumWindows();
    }

    private void setupMenus() {
        JMenuBar mainMenu = new JMenuBar();
        JMenu subMenu = null;

        // options menu
        JMenu optionsMenu = new JMenu("Options");

        slidersCheckBox = new JCheckBoxMenuItem("Show Number of Bins Slider", false);
        slidersCheckBox.addActionListener(this);
        optionsMenu.add(slidersCheckBox);

        ButtonGroup group = null;
        JRadioButtonMenuItem button = null;

        // units submenu
        unitsSubMenu = ParaProfUtils.createUnitsMenu(this, units, true);
        optionsMenu.add(unitsSubMenu);

        //Set the value type options.
        subMenu = new JMenu("Select Value Type");
        group = new ButtonGroup();

        button = new JRadioButtonMenuItem("Exclusive", true);
        button.addActionListener(this);
        group.add(button);
        subMenu.add(button);

        button = new JRadioButtonMenuItem("Inclusive", false);
        button.addActionListener(this);
        group.add(button);
        subMenu.add(button);

        button = new JRadioButtonMenuItem("Number of Calls", false);
        button.addActionListener(this);
        group.add(button);
        subMenu.add(button);

        button = new JRadioButtonMenuItem("Number of Child Calls", false);
        button.addActionListener(this);
        group.add(button);
        subMenu.add(button);

        button = new JRadioButtonMenuItem("Inclusive per Call", false);
        button.addActionListener(this);
        group.add(button);
        subMenu.add(button);

        button = new JRadioButtonMenuItem("Exclusive per Call", false);
        button.addActionListener(this);
        group.add(button);
        subMenu.add(button);

        optionsMenu.add(subMenu);

        optionsMenu.addMenuListener(this);

        //Now, add all the menus to the main menu.
        mainMenu.add(ParaProfUtils.createFileMenu(this, this, this));
        mainMenu.add(optionsMenu);
        //mainMenu.add(ParaProfUtils.createTrialMenu(ppTrial, this));
        mainMenu.add(ParaProfUtils.createWindowsMenu(ppTrial, this));
        mainMenu.add(ParaProfUtils.createHelpMenu(this, this));

        setJMenuBar(mainMenu);
    }

    public void actionPerformed(ActionEvent evt) {
        try {
            Object EventSrc = evt.getSource();

            if (EventSrc instanceof JMenuItem) {
                String arg = evt.getActionCommand();
                if (arg.equals("Exclusive")) {
                    dataSorter.setValueType(ValueType.EXCLUSIVE);
                    sortLocalData();
                } else if (arg.equals("Inclusive")) {
                    dataSorter.setValueType(ValueType.INCLUSIVE);
                    sortLocalData();
                } else if (arg.equals("Number of Calls")) {
                    dataSorter.setValueType(ValueType.NUMCALLS);
                    sortLocalData();
                } else if (arg.equals("Number of Child Calls")) {
                    dataSorter.setValueType(ValueType.NUMSUBR);
                    sortLocalData();
                } else if (arg.equals("Inclusive per Call")) {
                    dataSorter.setValueType(ValueType.INCLUSIVE_PER_CALL);
                    sortLocalData();
                } else if (arg.equals("Exclusive per Call")) {
                    dataSorter.setValueType(ValueType.EXCLUSIVE_PER_CALL);
                    sortLocalData();
                } else if (arg.equals("Show Number of Bins Slider")) {
                    if (slidersCheckBox.isSelected()) {
                        displaySliders(true);
                    } else {
                        displaySliders(false);
                    }
                } else {
                    throw new ParaProfException("The menu item '" + arg + "' is not implemented!");
                }
            }
        } catch (Exception e) {
            ParaProfUtils.handleException(e);
        }
    }

    private void displaySliders(boolean displaySliders) {
        Container contentPane = this.getContentPane();
        GridBagConstraints gbc = new GridBagConstraints();
        if (displaySliders) {
            contentPane.remove(chartPanel);

            gbc.fill = GridBagConstraints.NONE;
            gbc.anchor = GridBagConstraints.EAST;
            gbc.weightx = 0.10;
            gbc.weighty = 0.01;
            addCompItem(numBinsLabel, gbc, 0, 1, 1, 1);

            gbc.fill = GridBagConstraints.HORIZONTAL;
            gbc.anchor = GridBagConstraints.WEST;
            gbc.weightx = 0.70;
            gbc.weighty = 0.01;
            addCompItem(numBinsSlider, gbc, 1, 1, 1, 1);

            gbc.fill = GridBagConstraints.BOTH;
            gbc.anchor = GridBagConstraints.CENTER;
            gbc.weightx = 1.0;
            gbc.weighty = 0.99;
            addCompItem(chartPanel, gbc, 0, 2, 2, 1);
        } else {
            contentPane.remove(numBinsLabel);
            contentPane.remove(numBinsSlider);
            contentPane.remove(chartPanel);

            gbc.fill = GridBagConstraints.BOTH;
            gbc.anchor = GridBagConstraints.CENTER;
            gbc.weightx = 1;
            gbc.weighty = 1;
            addCompItem(chartPanel, gbc, 0, 1, 1, 1);
        }

        //Now call validate so that these component changes are displayed.
        validate();
    }

    public void menuSelected(MenuEvent evt) {
        try {
            if (ppTrial.isTimeMetric()) {
                unitsSubMenu.setEnabled(true);
            } else {
                unitsSubMenu.setEnabled(false);
            }
        } catch (Exception e) {
            ParaProfUtils.handleException(e);
        }
    }

    public void menuDeselected(MenuEvent evt) {}

    public void menuCanceled(MenuEvent evt) {}

    // listener for the numBinsSlider
    public void stateChanged(ChangeEvent event) {
        try {
            setNumBins(numBinsSlider.getValue());
        } catch (Exception e) {
            ParaProfUtils.handleException(e);
        }

    }

    public void update(Observable o, Object arg) {
        String tmpString = (String) arg;
        if (tmpString.equals("prefEvent")) {
            redraw();
        } else if (tmpString.equals("colorEvent")) {
            redraw();
        } else if (tmpString.equals("dataEvent")) {
            dataSorter.setSelectedMetricID(ppTrial.getDefaultMetricID());
            sortLocalData();
            redraw();
        } else if (tmpString.equals("subWindowCloseEvent")) {
            closeThisWindow();
        }
    }

    public void help(boolean display) {
        ParaProf.getHelpWindow().clearText();
        if (display)
            ParaProf.getHelpWindow().show();
        ParaProf.getHelpWindow().writeText("This is the histogram window");
        ParaProf.getHelpWindow().writeText("");
        ParaProf.getHelpWindow().writeText("This window shows you a histogram of all of the values for this function.");
        ParaProf.getHelpWindow().writeText("");
        ParaProf.getHelpWindow().writeText("Use the options menu to select different types of data to display.");
        ParaProf.getHelpWindow().writeText("");
    }

    private void sortLocalData() {
        data = dataSorter.getFunctionData(function, false, false);
        redraw();
    }

    public String getHeaderString() {
        if (ppTrial.getDataSource().getPhasesPresent()) {
            String starter;
            if (function.isCallPathFunction()) {
                starter = "Phase: " + UtilFncs.getLeftSide(function.getName()) + "\nName: "
                        + UtilFncs.getRightSide(function.getName());
            } else {
                starter = "Name: " + function.getName();
            }

            starter = starter + "\nMetric: " + ppTrial.getMetricName(dataSorter.getSelectedMetricID()) + "\nValue: "
                    + dataSorter.getValueType();

            if ((dataSorter.getValueType() == ValueType.NUMCALLS || dataSorter.getValueType() == ValueType.NUMSUBR)) {
                return starter;
            } else {
                return starter + "\nUnits: "
                        + UtilFncs.getUnitsString(units, dataSorter.isTimeMetric(), dataSorter.isDerivedMetric()) + "\n";
            }
        } else {
            if (dataSorter.getValueType() == ValueType.NUMCALLS || dataSorter.getValueType() == ValueType.NUMSUBR) {
                return "Metric Name: " + (ppTrial.getMetricName(ppTrial.getDefaultMetricID())) + "\n" + "Name: "
                        + ParaProfUtils.getDisplayName(function) + "\n" + "Value Type: " + dataSorter.getValueType() + "\n";
            } else {
                return "Metric Name: " + (ppTrial.getMetricName(ppTrial.getDefaultMetricID())) + "\n" + "Name: "
                        + ParaProfUtils.getDisplayName(function) + "\n" + "Value Type: " + dataSorter.getValueType() + "\n"
                        + "Units: " + UtilFncs.getUnitsString(units, ppTrial.isTimeMetric(), ppTrial.isDerivedMetric()) + "\n";
            }
        }
    }

    public List getData() {
        return data;
    }

    private void addCompItem(Component c, GridBagConstraints gbc, int x, int y, int w, int h) {
        gbc.gridx = x;
        gbc.gridy = y;
        gbc.gridwidth = w;
        gbc.gridheight = h;
        getContentPane().add(c, gbc);
    }

    void thisWindowClosing(java.awt.event.WindowEvent e) {
        closeThisWindow();
    }

    public void closeThisWindow() {
        try {
            setVisible(false);
            ppTrial.deleteObserver(this);
            ParaProf.decrementNumWindows();
        } catch (Exception e) {
            // do nothing
        }
        dispose();
    }

    public int units() {

        if (!dataSorter.isTimeMetric()) // we don't do units for non-time metrics
            return 0;

        if (dataSorter.getValueType() == ValueType.NUMCALLS || dataSorter.getValueType() == ValueType.NUMSUBR)
            return 0;

        return units;
    }

    public void setNumBins(int numBins) {
        this.numBins = Math.max(1, numBins);
        redraw();
    }

    public int getNumBins() {
        return numBins;
    }

    private JFreeChart createChart() {
        HistogramDataset dataset = new HistogramDataset();

        double maxValue = 0;
        double minValue = 0;
        boolean start = true;

        for (int i = 0; i < data.size(); i++) {
            PPFunctionProfile ppFunctionProfile = (PPFunctionProfile) data.get(i);
            double value = ppFunctionProfile.getValue();
            if (start) {
                minValue = value;
                start = false;
            }
            maxValue = Math.max(maxValue, value);
            minValue = Math.min(minValue, value);
        }

        double[] values = new double[data.size()];
        for (int i = 0; i < data.size(); i++) {
            PPFunctionProfile ppFunctionProfile = (PPFunctionProfile) data.get(i);
            values[i] = ppFunctionProfile.getValue();
        }

        dataset.addSeries(function.getName(), values, numBins, minValue, maxValue);

        String xAxis = dataSorter.getValueType().toString();

        if ((dataSorter.getValueType() != ValueType.NUMCALLS && dataSorter.getValueType() != ValueType.NUMSUBR)) {
            xAxis = xAxis + " " + ppTrial.getMetricName(dataSorter.getSelectedMetricID()) + " ("
                    + UtilFncs.getUnitsString(units, dataSorter.isTimeMetric(), dataSorter.isDerivedMetric()) + ")";
        }

        JFreeChart chart = ChartFactory.createHistogram(function.getName(), xAxis, "Threads", dataset, PlotOrientation.VERTICAL,
                false, // legend
                true, // tooltips
                false); // urls

        chart.getXYPlot().getDomainAxis().setUpperBound(maxValue);
        chart.getXYPlot().getDomainAxis().setLowerBound(minValue);

        Utility.applyDefaultChartTheme(chart);

        NumberAxis numberAxis = (NumberAxis) chart.getXYPlot().getDomainAxis();
        numberAxis.setNumberFormatOverride(ParaProfUtils.createNumberFormatter(units()));
        numberAxis.setTickLabelsVisible(true);

        numberAxis.setTickUnit(new NumberTickUnit((maxValue - minValue) / 10));

        final double binWidth = (maxValue - minValue) / numBins;

        // create the tooltip generator
        XYItemRenderer renderer = chart.getXYPlot().getRenderer();
        renderer.setToolTipGenerator(new XYToolTipGenerator() {
            public String generateToolTip(XYDataset dataset, int arg1, int arg2) {
                String minString = UtilFncs.getOutputString(units(), dataset.getXValue(arg1, arg2) - (binWidth / 2), 5);
                String maxString = UtilFncs.getOutputString(units(), dataset.getXValue(arg1, arg2) + (binWidth / 2), 5);

                return "<html>Number of threads: " + (int) dataset.getYValue(arg1, arg2) + "<br>Range minimum: " + minString
                        + "<br>Range maximum: " + maxString + "</html>";
            }
        });

        if (numBins < 25) {
            // it looks nicer with a margin, but only when we're at a low number of bars
            ((XYBarRenderer) chart.getXYPlot().getRenderer()).setMargin(0.10);
        }

        ((XYBarRenderer) chart.getXYPlot().getRenderer()).setOutlinePaint(Color.black);

        return chart;
    }

    private void redraw() {
        if (chartPanel != null) {
            chartPanel.setChart(createChart());
        }
    }

    public void setUnits(int units) {
        this.units = units;

        redraw();
    }

    public int print(Graphics graphics, PageFormat pageFormat, int pageIndex) throws PrinterException {
        return chartPanel.print(graphics, pageFormat, pageIndex);
    }

    public void export(Graphics2D g2D, boolean toScreen, boolean fullWindow, boolean drawHeader) {
        chartPanel.setDoubleBuffered(false);
        chartPanel.paintAll(g2D);
        chartPanel.setDoubleBuffered(true);
    }

    public Dimension getImageSize(boolean fullScreen, boolean header) {
        return chartPanel.getSize();
    }

    public JFrame getFrame() {
        return this;
    }

}