package edu.uoregon.tau.paraprof;

import java.awt.*;
import java.awt.event.*;
import java.util.*;
import java.util.List;

import javax.swing.*;
import javax.swing.event.*;

import edu.uoregon.tau.paraprof.barchart.*;
import edu.uoregon.tau.paraprof.enums.SortType;
import edu.uoregon.tau.paraprof.enums.ValueType;
import edu.uoregon.tau.paraprof.interfaces.ParaProfWindow;
import edu.uoregon.tau.paraprof.interfaces.SearchableOwner;
import edu.uoregon.tau.paraprof.interfaces.UnitListener;
import edu.uoregon.tau.perfdmf.Function;
import edu.uoregon.tau.perfdmf.Thread;
import edu.uoregon.tau.perfdmf.UtilFncs;

/**
 * The FunctionBarChartWindow displays performance data in many ways.  
 * All functions for one thread, all threads for one function, all phases for one function.
 * 
 * TODO : 
 * 1) Need to replace constructors with a factory, get rid of "changeToPhase..."
 * 2) Need to track all ppTrials (Observers) for comparisonChart 
 * 
 * <P>CVS $Id: FunctionBarChartWindow.java,v 1.5 2006/02/21 02:31:51 amorris Exp $</P>
 * @author  Robert Bell, Alan Morris
 * @version $Revision: 1.5 $
 * @see     FunctionBarChartModel
 * @see     ThreadBarChartModel
 */
public class FunctionBarChartWindow extends JFrame implements KeyListener, SearchableOwner, ActionListener, MenuListener,
        Observer, ChangeListener, ParaProfWindow, UnitListener {

    private ParaProfTrial ppTrial;
    private DataSorter dataSorter;

    private Function function;

    private JMenu optionsMenu;
    private JMenu unitsSubMenu;

    private JCheckBoxMenuItem sortByNCTCheckbox;
    private JCheckBoxMenuItem sortByNameCheckBox;
    private JCheckBoxMenuItem descendingOrderCheckBox;
    private JCheckBoxMenuItem showValuesAsPercent;
    private JCheckBoxMenuItem showMetaData;
    private JCheckBoxMenuItem showFindPanelBox;

    private JLabel barLengthLabel = new JLabel("Bar Width");
    private JSlider barLengthSlider = new JSlider(0, 2000, 400);

    private BarChartPanel panel;

    private int units = ParaProf.preferences.getUnits();

    private BarChartModel model;

    // Phase support
    private boolean phaseDisplay;

    private SearchPanel searchPanel;

    private PPThread ppThread;
    private Function phase;

    private boolean comparisonChart;

    // Initializes Chart as a single function across threads
    public FunctionBarChartWindow(ParaProfTrial ppTrial, Function function, Component parent) {
        this.ppTrial = ppTrial;
        this.function = function;
        dataSorter = new DataSorter(ppTrial);

        model = new FunctionBarChartModel(this, dataSorter, function);
        panel = new BarChartPanel(model, null);
        initialize(parent);

        this.setTitle("Function Data Window: " + ppTrial.getTrialIdentifier(ParaProf.preferences.getShowPathTitleInReverse()));
    }

    // Initializes Chart as a single thread across functions
    public FunctionBarChartWindow(ParaProfTrial ppTrial, Thread thread, Function phase, Component parent) {
        this.ppTrial = ppTrial;
        this.ppThread = new PPThread(thread, ppTrial);
        this.phase = phase;
        dataSorter = new DataSorter(ppTrial);
        dataSorter.setPhase(phase);

        barLengthSlider.setValue(250);
        model = new ThreadBarChartModel(this, dataSorter, ppThread);
        panel = new BarChartPanel(model, null);
        initialize(parent);

        panel.getBarChart().setLeftJustified(false);

        String phaseString = "";
        if (phase != null) {
            phaseString = " Phase: " + phase.getName();
        }

        if (thread.getNodeID() == -1) {
            this.setTitle("Mean Data - " + ppTrial.getTrialIdentifier(ParaProf.preferences.getShowPathTitleInReverse())
                    + phaseString);
        } else if (thread.getNodeID() == -2) {
            this.setTitle("Total Data - " + ppTrial.getTrialIdentifier(ParaProf.preferences.getShowPathTitleInReverse())
                    + phaseString);
        } else if (thread.getNodeID() == -3) {
            this.setTitle("Standard Deviation Data - "
                    + ppTrial.getTrialIdentifier(ParaProf.preferences.getShowPathTitleInReverse()) + phaseString);
        } else {
            this.setTitle(ppThread.getName() + " - "
                    + ppTrial.getTrialIdentifier(ParaProf.preferences.getShowPathTitleInReverse()) + phaseString);
        }
    }

    private FunctionBarChartWindow() {
    }

    
    public Function getFunction() {
        return function;
    }

    
    public static FunctionBarChartWindow CreateComparisonWindow(ParaProfTrial ppTrial, Thread thread, Component invoker) {
        FunctionBarChartWindow window = new FunctionBarChartWindow();

        window.setTitle("Comparison Window");
        window.dataSorter = new DataSorter(ppTrial);
        window.ppTrial = ppTrial;
        window.comparisonChart = true;
        window.model = new ComparisonBarChartModel(window, ppTrial, thread, window.dataSorter);

        window.panel = new BarChartPanel(window.model, null);
        //window.addThread(ppTrial, thread);

        window.initialize(invoker);
        window.panel.getBarChart().setLeftJustified(false);
        window.panel.getBarChart().setSingleLine(false);

        window.setHeader();
        return window;
    }

    public void addThread(ParaProfTrial ppTrial, Thread thread) {
        ppTrial.addObserver(this);

        ComparisonBarChartModel comp = (ComparisonBarChartModel) model;
        comp.addThread(ppTrial, thread);
        comp.reloadData();
        this.setHeader();
    }

    private void initialize(Component parent) {

        ppTrial.addObserver(this);

        panel.getBarChart().setBarLength(barLengthSlider.getValue());

        int windowWidth = 650;
        int windowHeight = 550;
        setSize(ParaProfUtils.checkSize(new java.awt.Dimension(windowWidth, windowHeight)));
        setLocation(WindowPlacer.getNewLocation(this, parent));

        addWindowListener(new java.awt.event.WindowAdapter() {
            public void windowClosing(java.awt.event.WindowEvent evt) {
                thisWindowClosing(evt);
            }
        });

        // Set up the data sorter
        dataSorter.setSelectedMetricID(ppTrial.getDefaultMetricID());
        //dataSorter.setValueType(ValueType.EXCLUSIVE_PERCENT);

        //Set the help window text if required.
        if (ParaProf.helpWindow.isVisible()) {
            this.help(false);
        }

        //Setting up the layout system for the main window.
        getContentPane().setLayout(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(2, 2, 2, 2);

        BarChart barChart = panel.getBarChart();

//        if (function != null) {
//            barChart.setLeftJustified(true);
//        }

        JScrollBar vScrollBar = panel.getVerticalScrollBar();
        vScrollBar.setUnitIncrement(35);

        this.addKeyListener(this);

        setupMenus();

        this.setHeader();

        sortLocalData();

        barLengthSlider.setPaintTicks(true);
        barLengthSlider.setMajorTickSpacing(400);
        barLengthSlider.setMinorTickSpacing(50);
        barLengthSlider.setPaintLabels(true);
        barLengthSlider.setSnapToTicks(false);
        barLengthSlider.addChangeListener(this);

        // add the scrollpane
        gbc.fill = GridBagConstraints.BOTH;
        gbc.anchor = GridBagConstraints.CENTER;
        gbc.weightx = 100;
        gbc.weighty = 100;
        addCompItem(panel, gbc, 0, 0, 1, 1);
        ParaProf.incrementNumWindows();

    }

    public void changeToPhaseDisplay(Thread thread) {

        this.ppThread = new PPThread(thread, ppTrial);
        phaseDisplay = true;

        this.setTitle(ParaProfUtils.getThreadIdentifier(thread) + " - Function Data: "
                + ppTrial.getTrialIdentifier(ParaProf.preferences.getShowPathTitleInReverse()));

        // in case we were opened on "main => foo", switch to foo
        this.function = ppTrial.getDataSource().getFunction(UtilFncs.getLeftSide(function.getName()));

        // we're no longer in a phase
        this.phase = null;
        setHeader();
        sortByNCTCheckbox.setSelected(false);
        optionsMenu.remove(sortByNCTCheckbox);
        descendingOrderCheckBox.setSelected(true);

        sortLocalData();
    }

    private Component createMetricMenu(final ValueType valueType, boolean enabled, ButtonGroup group) {
        JRadioButtonMenuItem button = null;

        if (ppTrial.getNumberOfMetrics() == 1) {
            button = new JRadioButtonMenuItem(valueType.toString(), enabled);

            button.addActionListener(new ActionListener() {
                public void actionPerformed(ActionEvent evt) {
                    dataSorter.setValueType(valueType);
                    sortLocalData();
                    panel.repaint();
                }
            });
            group.add(button);
            return button;
        } else {
            JMenu subSubMenu = new JMenu(valueType.toString() + "...");
            for (int i = 0; i < ppTrial.getNumberOfMetrics(); i++) {

                if (i == dataSorter.getSelectedMetricID() && enabled) {
                    button = new JRadioButtonMenuItem(ppTrial.getMetric(i).getName(), true);
                } else {
                    button = new JRadioButtonMenuItem(ppTrial.getMetric(i).getName());
                }
                final int metricID = i;

                button.addActionListener(new ActionListener() {

                    public void actionPerformed(ActionEvent evt) {
                        dataSorter.setSelectedMetricID(metricID);
                        dataSorter.setValueType(valueType);
                        sortLocalData();
                        panel.repaint();
                    }
                });
                group.add(button);
                subSubMenu.add(button);
            }
            return subSubMenu;
        }
    }

    public void setMetricID(int id) {
        dataSorter.setSelectedMetricID(id);
        sortLocalData();
        panel.repaint();
    }
    public int getMetricID() {
        return dataSorter.getSelectedMetricID();
    }
    
    public void setValueType(ValueType type) {
        dataSorter.setValueType(type);
        sortLocalData();
        panel.repaint();
    }
    public ValueType getValueType() {
        return dataSorter.getValueType();
    }
    
    
    private void setupMenus() {
        JMenuBar mainMenu = new JMenuBar();
        mainMenu.addKeyListener(this);
        JMenu subMenu = null;
        JMenuItem menuItem = null;

        optionsMenu = new JMenu("Options");

        JCheckBoxMenuItem box = null;
        ButtonGroup group = null;
        JRadioButtonMenuItem button = null;

        box = new JCheckBoxMenuItem("Show Width Slider", false);
        box.addActionListener(this);
        optionsMenu.add(box);

        showFindPanelBox = new JCheckBoxMenuItem("Show Find Panel", false);
        showFindPanelBox.addActionListener(this);
        optionsMenu.add(showFindPanelBox);

        showMetaData = new JCheckBoxMenuItem("Show Meta Data in Panel", true);
        showMetaData.addActionListener(this);
        optionsMenu.add(showMetaData);

        optionsMenu.add(new JSeparator());

        ActionListener sortData = new ActionListener() {
            public void actionPerformed(ActionEvent evt) {
                sortLocalData();
                panel.repaint();
            }
        };

        sortByNCTCheckbox = new JCheckBoxMenuItem("Sort By N,C,T", dataSorter.getSortType() == SortType.NCT);
        sortByNameCheckBox = new JCheckBoxMenuItem("Sort By Name", dataSorter.getSortType() == SortType.NAME);

        if (function != null) {
            sortByNCTCheckbox.addActionListener(sortData);
            optionsMenu.add(sortByNCTCheckbox);
        } else {
            sortByNameCheckBox.addActionListener(sortData);
            optionsMenu.add(sortByNameCheckBox);
        }

        descendingOrderCheckBox = new JCheckBoxMenuItem("Descending Order", dataSorter.getDescendingOrder());
        descendingOrderCheckBox.addActionListener(sortData);
        optionsMenu.add(descendingOrderCheckBox);

        showValuesAsPercent = new JCheckBoxMenuItem("Show Values as Percent", ParaProf.preferences.getShowValuesAsPercent());
        showValuesAsPercent.addActionListener(sortData);
        optionsMenu.add(showValuesAsPercent);

        unitsSubMenu = ParaProfUtils.createUnitsMenu(this, units, true);
        optionsMenu.add(unitsSubMenu);

        //Set the value type options.
        subMenu = new JMenu("Select Metric...");
        group = new ButtonGroup();

        subMenu.add(createMetricMenu(ValueType.EXCLUSIVE, dataSorter.getValueType() == ValueType.EXCLUSIVE
                || dataSorter.getValueType() == ValueType.EXCLUSIVE_PERCENT, group));
        subMenu.add(createMetricMenu(ValueType.INCLUSIVE, dataSorter.getValueType() == ValueType.INCLUSIVE
                || dataSorter.getValueType() == ValueType.INCLUSIVE_PERCENT, group));
        subMenu.add(createMetricMenu(ValueType.INCLUSIVE_PER_CALL, dataSorter.getValueType() == ValueType.INCLUSIVE_PER_CALL,
                group));
        subMenu.add(createMetricMenu(ValueType.EXCLUSIVE_PER_CALL, dataSorter.getValueType() == ValueType.EXCLUSIVE_PER_CALL,
                group));

        button = new JRadioButtonMenuItem("Number of Calls", dataSorter.getValueType() == ValueType.NUMCALLS);
        button.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent evt) {
                dataSorter.setValueType(ValueType.NUMCALLS);
                sortLocalData();
                panel.repaint();
            }
        });
        group.add(button);
        subMenu.add(button);

        button = new JRadioButtonMenuItem("Number of Child Calls", dataSorter.getValueType() == ValueType.NUMSUBR);
        button.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent evt) {
                dataSorter.setValueType(ValueType.NUMSUBR);
                sortLocalData();
                panel.repaint();
            }
        });
        group.add(button);
        subMenu.add(button);

        optionsMenu.add(subMenu);
        optionsMenu.addMenuListener(this);

        //Now, add all the menus to the main menu.
        mainMenu.add(ParaProfUtils.createFileMenu(this, panel, panel));

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
                if (arg.equals("Show Width Slider")) {
                    if (((JCheckBoxMenuItem) EventSrc).isSelected()) {
                        showWidthSlider(true);
                    } else {
                        showWidthSlider(false);
                    }
                } else if (arg.equals("Show Meta Data in Panel")) {
                    this.setHeader();
                } else if (arg.equals("Show Find Panel")) {
                    if (showFindPanelBox.isSelected()) {
                        showSearchPanel(true);
                    } else {
                        showSearchPanel(false);
                    }
                } else {
                    throw new ParaProfException("Menu system not implemented properly: " + arg);
                }
            }
        } catch (Exception e) {
            ParaProfUtils.handleException(e);
        }
    }
    
    
    

    public void stateChanged(ChangeEvent event) {
        try {
            panel.getBarChart().setBarLength(barLengthSlider.getValue());
            panel.repaint();
        } catch (Exception e) {
            ParaProfUtils.handleException(e);
        }
    }

    public void menuSelected(MenuEvent evt) {
        try {

            if (dataSorter.getValueType() == ValueType.EXCLUSIVE || dataSorter.getValueType() == ValueType.INCLUSIVE
                    || dataSorter.getValueType() == ValueType.EXCLUSIVE_PERCENT
                    || dataSorter.getValueType() == ValueType.INCLUSIVE_PERCENT) {
                showValuesAsPercent.setEnabled(true);

                if (showValuesAsPercent.isSelected()) {
                    unitsSubMenu.setEnabled(false);
                } else {

                    String metricName = ppTrial.getMetricName(dataSorter.getSelectedMetricID());
                    metricName = metricName.toUpperCase();
                    if (dataSorter.isTimeMetric()) {
                        unitsSubMenu.setEnabled(true);
                    } else {
                        unitsSubMenu.setEnabled(false);
                    }
                }
            } else {
                showValuesAsPercent.setEnabled(false);
                if (dataSorter.getValueType() == ValueType.EXCLUSIVE_PER_CALL
                        || dataSorter.getValueType() == ValueType.INCLUSIVE_PER_CALL) {
                    if (dataSorter.isTimeMetric()) {
                        unitsSubMenu.setEnabled(true);
                    } else {
                        unitsSubMenu.setEnabled(false);
                    }
                }
            }

        } catch (Exception e) {
            ParaProfUtils.handleException(e);
        }
    }

    public void menuDeselected(MenuEvent evt) {
    }

    public void menuCanceled(MenuEvent evt) {
    }

    public void update(Observable o, Object arg) {
        String tmpString = (String) arg;
        if (tmpString.equals("prefEvent")) {
            this.setHeader();
            panel.repaint();
        } else if (tmpString.equals("colorEvent")) {
            panel.repaint();
        } else if (tmpString.equals("dataEvent")) {
            dataSorter.setSelectedMetricID(ppTrial.getDefaultMetricID());
            setupMenus();
            validate();
            sortLocalData();
            this.setHeader();
            panel.repaint();
        } else if (tmpString.equals("subWindowCloseEvent")) {
            closeThisWindow();
        }
    }

    public void help(boolean display) {
        //Show the ParaProf help window.
        ParaProf.helpWindow.clearText();
        if (display) {
            ParaProf.helpWindow.show();
        }

        if (function != null) {
            ParaProf.helpWindow.writeText("This is the function data window for:");
            ParaProf.helpWindow.writeText(ParaProfUtils.getFunctionName(function));
            ParaProf.helpWindow.writeText("");
            ParaProf.helpWindow.writeText("This window shows you this function's statistics across all the threads.");
            ParaProf.helpWindow.writeText("");
            ParaProf.helpWindow.writeText("Use the options menu to select different ways of displaying the data.");
            ParaProf.helpWindow.writeText("");
            ParaProf.helpWindow.writeText("Right click anywhere within this window to bring up a popup");
            ParaProf.helpWindow.writeText("menu. In this menu you can change or reset the default color");
            ParaProf.helpWindow.writeText("for this function.");
        } else {
            ParaProf.helpWindow.writeText("This is the thread data window");
            ParaProf.helpWindow.writeText("");
            ParaProf.helpWindow.writeText("This window shows you the values for all functions on this thread.");
            ParaProf.helpWindow.writeText("");
            ParaProf.helpWindow.writeText("Use the options menu to select different ways of displaying the data.");
            ParaProf.helpWindow.writeText("");
            ParaProf.helpWindow.writeText("Right click on any function within this window to bring up a popup");
            ParaProf.helpWindow.writeText("menu. In this menu you can change or reset the default color");
            ParaProf.helpWindow.writeText("for the function, or to show more details about the function.");
            ParaProf.helpWindow.writeText("You can also left click any function to highlight it in the system.");
        }
    }

    public DataSorter getDataSorter() {
        return dataSorter;
    }

    public void sortLocalData() {
        dataSorter.setDescendingOrder(descendingOrderCheckBox.isSelected());

        if (function != null) { // function
            if (sortByNCTCheckbox.isSelected()) {
                dataSorter.setSortType(SortType.NCT);
            } else {
                dataSorter.setSortType(SortType.VALUE);
            }

        } else { // thread
            if (sortByNameCheckBox.isSelected()) {
                dataSorter.setSortType(SortType.NAME);
            } else {
                dataSorter.setSortType(SortType.VALUE);
            }
        }

        if (showValuesAsPercent.isSelected()) {
            if (dataSorter.getValueType() == ValueType.EXCLUSIVE) {
                dataSorter.setValueType(ValueType.EXCLUSIVE_PERCENT);
            } else if (dataSorter.getValueType() == ValueType.INCLUSIVE) {
                dataSorter.setValueType(ValueType.INCLUSIVE_PERCENT);
            }
        } else {
            if (dataSorter.getValueType() == ValueType.EXCLUSIVE_PERCENT) {
                dataSorter.setValueType(ValueType.EXCLUSIVE);
            } else if (dataSorter.getValueType() == ValueType.INCLUSIVE_PERCENT) {
                dataSorter.setValueType(ValueType.INCLUSIVE);
            }
        }
        setHeader();

        model.reloadData();
    }

    public int units() {
        if (showValuesAsPercent.isEnabled() && showValuesAsPercent.isSelected())
            return 0;

        if (!dataSorter.isTimeMetric()) // we don't do units for non-time metrics
            return 0;

        if (dataSorter.getValueType() == ValueType.NUMCALLS || dataSorter.getValueType() == ValueType.NUMSUBR)
            return 0;

        return units;
    }

    public Dimension getViewportSize() {
        return panel.getViewport().getExtentSize();
    }

    public Rectangle getViewRect() {
        return panel.getViewport().getViewRect();
    }

    public void setVerticalScrollBarPosition(int position) {
        JScrollBar scrollBar = panel.getVerticalScrollBar();
        scrollBar.setValue(position);
    }

    // This process is separated into two functions to provide the option of obtaining the current 
    // header string being used for the panel without resetting the actual header. 
    // Printing and image generation use this functionality.
    public void setHeader() {
        if (showMetaData.isSelected()) {
            JTextArea jTextArea = new JTextArea();
            jTextArea.setLineWrap(true);
            jTextArea.setWrapStyleWord(true);
            jTextArea.setEditable(false);
            jTextArea.setFont(ParaProf.preferencesWindow.getFont());
            jTextArea.append(this.getHeaderString());
            jTextArea.addKeyListener(this);
            jTextArea.setMargin(new Insets(3,3,3,3));
            if (comparisonChart) {
                
                JPanel legendPanel = new LegendPanel(((ComparisonBarChartModel) model).getLegendModel());
                JPanel holder = new JPanel();
                holder.setLayout(new GridBagLayout());
                GridBagConstraints gbc = new GridBagConstraints();
                gbc.fill = GridBagConstraints.BOTH;
                gbc.anchor = GridBagConstraints.WEST;
                gbc.weightx = 0;
                gbc.weighty = 1;
                ParaProfUtils.addCompItem(holder, jTextArea, gbc, 0,0,1,1);
                gbc.weightx = 1;
                gbc.weighty = 1;
                gbc.anchor = GridBagConstraints.WEST;
                ParaProfUtils.addCompItem(holder, legendPanel, gbc, 1,0,1,1);
                
                panel.setColumnHeaderView(holder);
            } else {
                panel.setColumnHeaderView(jTextArea);
            }
        } else {
            panel.setColumnHeaderView(null);
        }
    }

    public String getHeaderString() {
        if (function != null) {
            String header = "";
            if (ppTrial.getDataSource().getPhasesPresent() && function.isCallPathFunction()) {
                header += "Phase: " + UtilFncs.getLeftSide(function.getName()) + "\nName: "
                        + UtilFncs.getRightSide(function.getName());
            } else {
                header += "Name: " + function.getName();
            }

            header += "\nMetric Name: " + ppTrial.getMetricName(dataSorter.getSelectedMetricID()) + "\nValue: "
                    + dataSorter.getValueType();

            if ((dataSorter.getValueType() == ValueType.NUMCALLS || dataSorter.getValueType() == ValueType.NUMSUBR)
                    || showValuesAsPercent.isSelected()) {
                // nothing
            } else {
                header += "\nUnits: " + UtilFncs.getUnitsString(units, dataSorter.isTimeMetric(), dataSorter.isDerivedMetric());
            }
            return header + "\n";
        } else {
            String starter;

            if (phase != null) {
                starter = "Phase: " + phase + "\nMetric: " + (ppTrial.getMetricName(dataSorter.getSelectedMetricID())) + "\n"
                        + "Value: " + dataSorter.getValueType();
            } else {
                starter = "Metric: " + (ppTrial.getMetricName(dataSorter.getSelectedMetricID())) + "\n" + "Value: "
                        + dataSorter.getValueType();
            }

            if ((dataSorter.getValueType() == ValueType.NUMCALLS || dataSorter.getValueType() == ValueType.NUMSUBR)
                    || showValuesAsPercent.isSelected()) {
                return starter + "\n";
            } else {
                return starter + "\nUnits: "
                        + UtilFncs.getUnitsString(units, dataSorter.isTimeMetric(), dataSorter.isDerivedMetric()) + "\n";
            }
        }
    }

    private void showWidthSlider(boolean displaySliders) {
        GridBagConstraints gbc = new GridBagConstraints();
        if (displaySliders) {
            getContentPane().remove(panel);

            gbc.insets = new Insets(5, 5, 5, 5);
            gbc.fill = GridBagConstraints.NONE;
            gbc.anchor = GridBagConstraints.EAST;
            gbc.weightx = 0.10;
            gbc.weighty = 0.01;
            addCompItem(barLengthLabel, gbc, 0, 0, 1, 1);

            gbc.fill = GridBagConstraints.HORIZONTAL;
            gbc.anchor = GridBagConstraints.WEST;
            gbc.weightx = 0.70;
            gbc.weighty = 0.01;
            addCompItem(barLengthSlider, gbc, 1, 0, 1, 1);

            gbc.insets = new Insets(0, 0, 0, 0);
            gbc.fill = GridBagConstraints.BOTH;
            gbc.anchor = GridBagConstraints.CENTER;
            gbc.weightx = 1.0;
            gbc.weighty = 0.99;
            addCompItem(panel, gbc, 0, 1, 2, 1);
        } else {
            getContentPane().remove(barLengthLabel);
            getContentPane().remove(barLengthSlider);
            getContentPane().remove(panel);

            gbc.fill = GridBagConstraints.BOTH;
            gbc.anchor = GridBagConstraints.CENTER;
            gbc.weightx = 100;
            gbc.weighty = 100;
            addCompItem(panel, gbc, 0, 0, 1, 1);
        }

        //Now call validate so that these componant changes are displayed.
        validate();
    }

    public void showSearchPanel(boolean show) {
        if (show) {
            if (searchPanel == null) {
                searchPanel = new SearchPanel(this, panel.getBarChart().getSearcher());
                GridBagConstraints gbc = new GridBagConstraints();
                gbc.insets = new Insets(5, 5, 5, 5);
                gbc.fill = GridBagConstraints.HORIZONTAL;
                gbc.anchor = GridBagConstraints.CENTER;
                gbc.weightx = 0.10;
                gbc.weighty = 0.01;
                addCompItem(searchPanel, gbc, 0, 3, 2, 1);
                searchPanel.setFocus();
            }
        } else {
            getContentPane().remove(searchPanel);
            searchPanel = null;
        }

        showFindPanelBox.setSelected(show);
        validate();
    }

    private void addCompItem(Component c, GridBagConstraints gbc, int x, int y, int w, int h) {
        gbc.gridx = x;
        gbc.gridy = y;
        gbc.gridwidth = w;
        gbc.gridheight = h;
        getContentPane().add(c, gbc);
    }

    //Respond correctly when this window is closed.
    void thisWindowClosing(java.awt.event.WindowEvent e) {
        closeThisWindow();
    }

    public void closeThisWindow() {
        try {
            setVisible(false);
            
            if (comparisonChart) {
                ParaProf.theComparisonWindow = null;
                List trialList = ((ComparisonBarChartModel)model).getPpTrials();
                for (Iterator it=trialList.iterator(); it.hasNext();) {
                    ParaProfTrial trial = (ParaProfTrial)it.next();
                    trial.deleteObserver(this);
                }
            } else {
                ppTrial.deleteObserver(this);
            }
            ParaProf.decrementNumWindows();
            
        } catch (Exception e) {
            // do nothing
        }
        dispose();
    }

    
    public int getUnits() {
        return units;
    }
    
    public void setUnits(int units) {
        this.units = units;
        this.setHeader();
        model.reloadData();
        panel.repaint();
    }

    public boolean isPhaseDisplay() {
        return phaseDisplay;
    }

    public ParaProfTrial getPpTrial() {
        return ppTrial;
    }

    public Thread getThread() {
        return ppThread.getThread();
    }

    public void keyPressed(KeyEvent e) {
        if (e.isControlDown() && e.getKeyCode() == KeyEvent.VK_F) {
            showSearchPanel(true);
        }
    }

    public void keyReleased(KeyEvent e) {
    }

    public void keyTyped(KeyEvent e) {
    }

    public Function getPhase() {
        return phase;
    }

    
    public void setDescendingOrder(boolean order) {
        descendingOrderCheckBox.setSelected(order);
        sortLocalData();
        panel.repaint();
    }

    public boolean getDescendingOrder() {
        return descendingOrderCheckBox.isSelected();
    }

    public void setSortByName(boolean order) {
        sortByNameCheckBox.setSelected(order);
        sortLocalData();
        panel.repaint();
    }
    public boolean getSortByName() {
        return sortByNameCheckBox.isSelected();
    }
    
    public void setSortByNCT(boolean order) {
        sortByNCTCheckbox.setSelected(order);
    }
    public boolean getSortByNCT() {
        return sortByNCTCheckbox.isSelected();
    }
    
    
    
}