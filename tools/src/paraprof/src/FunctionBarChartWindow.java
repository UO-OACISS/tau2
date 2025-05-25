package edu.uoregon.tau.paraprof;

import java.awt.Component;
import java.awt.Dimension;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.awt.Rectangle;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.util.List;
import java.util.Observable;
import java.util.Observer;

import javax.swing.JCheckBoxMenuItem;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JPanel;
import javax.swing.JScrollBar;
import javax.swing.JScrollPane;
import javax.swing.JSeparator;
import javax.swing.JSlider;
import javax.swing.JTextArea;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import javax.swing.event.MenuEvent;
import javax.swing.event.MenuListener;

import edu.uoregon.tau.common.Utility;
import edu.uoregon.tau.paraprof.barchart.BarChartModel;
import edu.uoregon.tau.paraprof.barchart.BarChartPanel;
import edu.uoregon.tau.paraprof.barchart.ComparisonBarChartModel;
import edu.uoregon.tau.paraprof.barchart.FunctionBarChartModel;
import edu.uoregon.tau.paraprof.barchart.LegendPanel;
import edu.uoregon.tau.paraprof.barchart.ThreadBarChartModel;
import edu.uoregon.tau.paraprof.enums.SortType;
import edu.uoregon.tau.paraprof.enums.ValueType;
import edu.uoregon.tau.paraprof.interfaces.ParaProfWindow;
import edu.uoregon.tau.paraprof.interfaces.SearchableOwner;
import edu.uoregon.tau.paraprof.interfaces.SortListener;
import edu.uoregon.tau.paraprof.interfaces.UnitListener;
import edu.uoregon.tau.perfdmf.Function;
import edu.uoregon.tau.perfdmf.Metric;
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
 * <P>CVS $Id: FunctionBarChartWindow.java,v 1.25 2010/01/14 01:31:39 amorris Exp $</P>
 * @author  Robert Bell, Alan Morris
 * @version $Revision: 1.25 $
 * @see     FunctionBarChartModel
 * @see     ThreadBarChartModel
 */
public class FunctionBarChartWindow extends JFrame implements KeyListener, SearchableOwner, ActionListener, MenuListener,
        Observer, ChangeListener, ParaProfWindow, UnitListener, SortListener {

    /**
	 *
	 */
	private static final long serialVersionUID = 3984568716635957738L;
	private ParaProfTrial ppTrial;
    private DataSorter dataSorter;

    private Function function;

    private JMenu optionsMenu;
    private JMenu unitsSubMenu;

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

    private boolean defaultPercentValue;
    private String phaseString;
    // we keep these around to speed things up
    private JTextArea jTextArea;
    private JScrollPane jTextAreaScrollPane;
    private Component headerView;

    private FunctionBarChartWindow() {
    // disable default constructor
    }

    // Initializes Chart as a single function across threads
    public FunctionBarChartWindow(ParaProfTrial ppTrial, Function function, Component parent) {
        this(ppTrial, function, parent, false, null);
    }

    public FunctionBarChartWindow(ParaProfTrial ppTrial, Function function, Component parent, boolean isPhase, Thread thread) {
        this.ppTrial = ppTrial;
        this.function = function;
        dataSorter = new DataSorter(ppTrial);
        this.phaseDisplay = isPhase;

        if (phaseDisplay) {
            this.ppThread = new PPThread(thread, ppTrial);
            // in case we were opened on "main => foo", switch to foo
            this.function = ppTrial.getDataSource().getFunction(UtilFncs.getLeftSide(function.getName()));

            // we're no longer in a phase
            this.phase = null;
            //sortByNCTCheckbox.setSelected(false);
            //optionsMenu.remove(sortByNCTCheckbox);
            //descendingOrderCheckBox.setSelected(true);

            //sortLocalData();
            setTitle("TAU: ParaProf: " + ParaProfUtils.getThreadLabel(thread) + " - Function Data: "
                    + ppTrial.getTrialIdentifier(ParaProf.preferences.getShowPathTitleInReverse()));
        } else {
            setTitle("TAU: ParaProf: Function Data Window: "
                    + ppTrial.getTrialIdentifier(ParaProf.preferences.getShowPathTitleInReverse()));
        }
        ParaProfUtils.setFrameIcon(this);

        model = new FunctionBarChartModel(this, dataSorter, function);
        panel = new BarChartPanel(model, null);
        initialize(parent);

    }

    public static FunctionBarChartWindow CreateFunctionsOverPhaseDisplay(ParaProfTrial ppTrial, Function function, Thread thread,
            Component parent) {
        FunctionBarChartWindow fbcw = new FunctionBarChartWindow(ppTrial, function, parent, true, thread);
        return fbcw;
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

        phaseString = "";
        if (phase != null) {
            phaseString = " Phase: " + phase.getName();
        }

        this.setTitle("TAU: ParaProf: " + ParaProfUtils.getThreadLabel(thread) + " - "
                + ppTrial.getTrialIdentifier(ParaProf.preferences.getShowPathTitleInReverse()) + phaseString);
        ParaProfUtils.setFrameIcon(this);

    }

    public static FunctionBarChartWindow CreateComparisonWindow(ParaProfTrial ppTrial, Thread thread, Component invoker) {
        FunctionBarChartWindow window = new FunctionBarChartWindow();

        window.setTitle("ParaProf: Comparison Window");
        ParaProfUtils.setFrameIcon(window);

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

        defaultPercentValue = ParaProf.preferences.getShowValuesAsPercent();
        ppTrial.addObserver(this);

        panel.getBarChart().setBarLength(barLengthSlider.getValue());

        int windowWidth = 650;
        int windowHeight = 550;
        setSize(ParaProfUtils.checkSize(new java.awt.Dimension(windowWidth, windowHeight)));
        setLocation(WindowPlacer.getNewLocation(this, parent));

        addWindowListener(new java.awt.event.WindowAdapter() {
            @Override
			public void windowClosing(java.awt.event.WindowEvent evt) {
                thisWindowClosing(evt);
            }
        });

        // Set up the data sorter
        dataSorter.setSelectedMetric(ppTrial.getDefaultMetric());
        dataSorter.setSortMetric(ppTrial.getDefaultMetric());
        dataSorter.setSortByVisible(true);
        dataSorter.setSortType(SortType.VALUE);
        //dataSorter.setValueType(ValueType.EXCLUSIVE_PERCENT);

        //Set the help window text if required.
        if (ParaProf.getHelpWindow().isVisible()) {
            this.help(false);
        }

        //Setting up the layout system for the main window.
        getContentPane().setLayout(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(2, 2, 2, 2);

        //BarChart barChart = panel.getBarChart();

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

    public void setMetric(Metric metric) {
        dataSorter.setSelectedMetric(metric);
        sortLocalData();
        panel.repaint();
    }

    public Metric getMetric() {
        return dataSorter.getSelectedMetric();
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
        JCheckBoxMenuItem box;

        optionsMenu = new JMenu("Options");

        box = new JCheckBoxMenuItem("Show Width Slider", false);
        box.addActionListener(this);
        //optionsMenu.add(box);

        showFindPanelBox = new JCheckBoxMenuItem("Show Find Panel", false);
        showFindPanelBox.addActionListener(this);
        optionsMenu.add(showFindPanelBox);

        showMetaData = new JCheckBoxMenuItem("Show Meta Data in Panel", true);
        showMetaData.addActionListener(this);
        optionsMenu.add(showMetaData);

        optionsMenu.add(new JSeparator());

        ActionListener sortData = new ActionListener() {
            @Override
			public void actionPerformed(ActionEvent evt) {
                sortLocalData();
                panel.repaint();
            }
        };

        descendingOrderCheckBox = new JCheckBoxMenuItem("Descending Order", dataSorter.getDescendingOrder());
        descendingOrderCheckBox.addActionListener(sortData);
        optionsMenu.add(descendingOrderCheckBox);

        if (showValuesAsPercent == null) {
        showValuesAsPercent = new JCheckBoxMenuItem("Show Values as Percent", defaultPercentValue);
        }
        showValuesAsPercent.addActionListener(sortData);
        optionsMenu.add(showValuesAsPercent);

        unitsSubMenu = ParaProfUtils.createUnitsMenu(this, units, true);
        optionsMenu.add(unitsSubMenu);

        boolean nct = false;
        // if this is a function display (one function across all threads), we want 'sort by nct'
        if (function != null) {
            nct = true;
        }
        // if this is a phase display (all functions across one phase), we want 'sort by name'
        if (phaseDisplay) {
            nct = false;
        }

        optionsMenu.add(ParaProfUtils.createMetricSelectionMenu(ppTrial, "Select Metric...", false, nct, dataSorter, this, true));
        optionsMenu.add(ParaProfUtils.createMetricSelectionMenu(ppTrial, "Sort by...", true, nct, dataSorter, this, true));

        optionsMenu.addMenuListener(this);

        //Now, add all the menus to the main menu.
        mainMenu.add(ParaProfUtils.createFileMenu(this, panel, panel));

        mainMenu.add(optionsMenu);
        //mainMenu.add(ParaProfUtils.createTrialMenu(ppTrial, this));
        mainMenu.add(ParaProfUtils.createWindowsMenu(ppTrial, this));
        mainMenu.add(ParaProfUtils.createHelpMenu(this, this));

        setJMenuBar(mainMenu);
    }

    @Override
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

    @Override
	public void stateChanged(ChangeEvent event) {
        try {
            panel.getBarChart().setBarLength(barLengthSlider.getValue());
            panel.repaint();
        } catch (Exception e) {
            ParaProfUtils.handleException(e);
        }
    }

    @Override
	public void menuSelected(MenuEvent evt) {
        try {

            if (dataSorter.getValueType() == ValueType.EXCLUSIVE || dataSorter.getValueType() == ValueType.INCLUSIVE
                    || dataSorter.getValueType() == ValueType.EXCLUSIVE_PERCENT
                    || dataSorter.getValueType() == ValueType.INCLUSIVE_PERCENT) {
                showValuesAsPercent.setEnabled(true);

                if (showValuesAsPercent.isSelected()) {
                    unitsSubMenu.setEnabled(false);
                } else {
                    String metricName = dataSorter.getSelectedMetric().getName();
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

    @Override
	public void menuDeselected(MenuEvent evt) {}

    @Override
	public void menuCanceled(MenuEvent evt) {}

    @Override
	public void update(Observable o, Object arg) {
        String tmpString = (String) arg;
        if (tmpString.equals("prefEvent")) {
        	if(!this.comparisonChart&&this.ppThread!=null){
        	this.setTitle("TAU: ParaProf: " + ParaProfUtils.getThreadLabel(ppThread.getThread()) + " - "
                    + ppTrial.getTrialIdentifier(ParaProf.preferences.getShowPathTitleInReverse()) + phaseString);
        	}


            this.setHeader();
            panel.repaint();



        } else if (tmpString.equals("colorEvent")) {
            panel.repaint();
        } else if (tmpString.equals("dataEvent")) {
            dataSorter.setSelectedMetric(ppTrial.getDefaultMetric());
            setupMenus();
            validate(); // must call validate or the new JMenuBar won't work
            sortLocalData();
            panel.repaint();
        } else if (tmpString.equals("subWindowCloseEvent")) {
            closeThisWindow();
        }
    }

    @Override
	public void help(boolean display) {
        //Show the ParaProf help window.
        ParaProf.getHelpWindow().clearText();
        if (display) {
            ParaProf.getHelpWindow().setVisible(true);
        }

        if (function != null) {
            ParaProf.getHelpWindow().writeText("This is the function data window for:");
            ParaProf.getHelpWindow().writeText(ParaProfUtils.getDisplayName(function));
            ParaProf.getHelpWindow().writeText("");
            ParaProf.getHelpWindow().writeText("This window shows you this function's statistics across all the threads.");
            ParaProf.getHelpWindow().writeText("");
            ParaProf.getHelpWindow().writeText("Use the options menu to select different ways of displaying the data.");
            ParaProf.getHelpWindow().writeText("");
            ParaProf.getHelpWindow().writeText("Right click anywhere within this window to bring up a popup");
            ParaProf.getHelpWindow().writeText("menu. In this menu you can change or reset the default color");
            ParaProf.getHelpWindow().writeText("for this function.");
        } else {
            ParaProf.getHelpWindow().writeText("This is the thread data window");
            ParaProf.getHelpWindow().writeText("");
            ParaProf.getHelpWindow().writeText("This window shows you the values for all functions on this thread.");
            ParaProf.getHelpWindow().writeText("");
            ParaProf.getHelpWindow().writeText("Use the options menu to select different ways of displaying the data.");
            ParaProf.getHelpWindow().writeText("");
            ParaProf.getHelpWindow().writeText("Right click on any function within this window to bring up a popup");
            ParaProf.getHelpWindow().writeText("menu. In this menu you can change or reset the default color");
            ParaProf.getHelpWindow().writeText("for the function, or to show more details about the function.");
            ParaProf.getHelpWindow().writeText("You can also left click any function to highlight it in the system.");
        }
    }

    public DataSorter getDataSorter() {
        return dataSorter;
    }

    public void sortLocalData() {
        dataSorter.setDescendingOrder(descendingOrderCheckBox.isSelected());

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
        if ((showValuesAsPercent.isEnabled() && showValuesAsPercent.isSelected()) || !dataSorter.isTimeMetric()) {
            // we don't do units for non-time metrics
            return 0;
        }
        if (dataSorter.getValueType() == ValueType.NUMCALLS || dataSorter.getValueType() == ValueType.NUMSUBR) {
            return 0;
        }
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

            if (jTextArea == null) {
                jTextArea = new JTextArea();
                jTextArea.setLineWrap(true);
                jTextArea.setWrapStyleWord(true);
                jTextArea.setEditable(false);
                jTextArea.setFont(ParaProf.preferencesWindow.getFont());
                jTextArea.addKeyListener(this);
                jTextArea.setMargin(new Insets(3, 3, 3, 3));
                
                jTextAreaScrollPane = new JScrollPane(jTextArea);
                jTextAreaScrollPane.setBorder(null);
                jTextAreaScrollPane.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED);
                jTextAreaScrollPane.setHorizontalScrollBarPolicy(JScrollPane.HORIZONTAL_SCROLLBAR_NEVER);
                jTextAreaScrollPane.setPreferredSize(new Dimension(100,100));
            }
            String headerString=this.getHeaderString();
            
            jTextArea.setText(headerString);

            if (comparisonChart) {
                jTextArea.setSize(new Dimension(250, 200));
                JPanel legendPanel = new LegendPanel(((ComparisonBarChartModel) model).getLegendModel());
                JPanel holder = new JPanel();
                holder.setLayout(new GridBagLayout());
                GridBagConstraints gbc = new GridBagConstraints();
                gbc.fill = GridBagConstraints.BOTH;
                gbc.anchor = GridBagConstraints.WEST;
                gbc.weightx = 0;
                gbc.weighty = 1;
                Utility.addCompItem(holder, jTextArea, gbc, 0, 0, 1, 1);
                gbc.weightx = 1;
                gbc.weighty = 1;
                gbc.anchor = GridBagConstraints.EAST;
                Utility.addCompItem(holder, legendPanel, gbc, 1, 0, 1, 1);

                panel.setColumnHeaderView(holder);
                headerView = holder;
            } else {
                if (headerView != jTextAreaScrollPane) {
                    panel.setColumnHeaderView(jTextAreaScrollPane);
                    headerView = jTextAreaScrollPane;
                }
            }
        } else {
            panel.setColumnHeaderView(null);
            headerView = null;
        }
    }

    public String getHeaderString() {
        if (function != null) {
            StringBuilder header = new StringBuilder();
            if (ppTrial.getDataSource().getPhasesPresent() && function.isCallPathFunction()) {
                header.append("Phase: ").append(UtilFncs.getLeftSide(function.getName())).append("\nName: ").append(UtilFncs.getRightSide(function.getName()));
            } else {
                header.append("Name: ").append(function.getName());
            }

            header.append("\nMetric Name: ").append(dataSorter.getSelectedMetric().getName()).append("\nValue: ").append(dataSorter.getValueType());

            if ((dataSorter.getValueType() == ValueType.NUMCALLS || dataSorter.getValueType() == ValueType.NUMSUBR)
                    || showValuesAsPercent.isSelected()) {
                // nothing
            } else {
                header.append("\nUnits: ").append(UtilFncs.getUnitsString(units, dataSorter.isTimeMetric(), dataSorter.isDerivedMetric(),dataSorter.getSelectedMetric().getName()));
            }
            return header.append("\n").toString();
        } else {
            String starter;

            if (phase != null) {
                starter = "Phase: " + phase + "\nMetric: " + dataSorter.getSelectedMetric().getName() + "\n" + "Value: "
                        + dataSorter.getValueType();
            } else {
                starter = "Metric: " + dataSorter.getSelectedMetric().getName() + "\n" + "Value: " + dataSorter.getValueType();
            }

            if ((dataSorter.getValueType() == ValueType.NUMCALLS || dataSorter.getValueType() == ValueType.NUMSUBR)
                    || showValuesAsPercent.isSelected()) {
                starter += "\n";
            } else {
                starter += "\nUnits: "
                        + UtilFncs.getUnitsString(units, dataSorter.isTimeMetric(), dataSorter.isDerivedMetric(),dataSorter.getSelectedMetric().getName()) + "\n";
            }

            if(dataSorter.getSortType()==SortType.NAME){
        		starter+="Sorted By: Name";
        	}
        	else if(!dataSorter.getSortByVisible()){
        		starter+="Sorted By: "+dataSorter.getSortValueType().toString();
        		if(dataSorter.getSortMetric()!=dataSorter.getSelectedMetric()){
        			starter+=" ("+dataSorter.getSortMetric()+")";
        		}
        	}

            return starter;
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

        //Now call validate so that these component changes are displayed.
        validate();
    }

    @Override
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

    @Override
	public void closeThisWindow() {
        try {
            setVisible(false);

            if (comparisonChart) {
                ParaProf.theComparisonWindow = null;
                List<ParaProfTrial> trialList = ((ComparisonBarChartModel) model).getPpTrials();
                for (ParaProfTrial trial : trialList) {
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

    @Override
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

    @Override
	public void keyPressed(KeyEvent e) {
        if (e.isControlDown() && e.getKeyCode() == KeyEvent.VK_F) {
            showSearchPanel(true);
        }
    }

    @Override
	public void keyReleased(KeyEvent e) {}

    @Override
	public void keyTyped(KeyEvent e) {}

    public Function getPhase() {
        return phase;
    }

    public Function getFunction() {
        return function;
    }

    public void setDescendingOrder(boolean order) {
        descendingOrderCheckBox.setSelected(order);
        sortLocalData();
        panel.repaint();
    }

    public boolean getDescendingOrder() {
        return descendingOrderCheckBox.isSelected();
    }

    @Override
	public void resort() {
        sortLocalData();
        panel.repaint();
    }

    public BarChartPanel getPanel() {
        return panel;
    }

    @Override
	public JFrame getFrame() {
        return this;
    }

}