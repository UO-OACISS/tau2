package edu.uoregon.tau.paraprof;

import java.util.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.event.*;
import java.awt.print.*;
import edu.uoregon.tau.dms.dss.*;
import edu.uoregon.tau.paraprof.enums.*;

/**
 * FunctionDataWindow
 * This is the FunctionDataWindow.
 *  
 * <P>CVS $Id: FunctionDataWindow.java,v 1.14 2005/03/09 18:07:50 amorris Exp $</P>
 * @author	Robert Bell, Alan Morris
 * @version	$Revision: 1.14 $
 * @see		FunctionDataWindowPanel
 */
public class FunctionDataWindow extends JFrame implements ActionListener, MenuListener, Observer,
        ChangeListener {

    public FunctionDataWindow(ParaProfTrial trial, Function function) {
        this.ppTrial = trial;

        this.function = function;
        int windowWidth = 650;
        int windowHeight = 550;
        setSize(new java.awt.Dimension(windowWidth, windowHeight));

        //Now set the title.
        this.setTitle("Function Data Window: " + trial.getTrialIdentifier(true));

        //Add some window listener code
        addWindowListener(new java.awt.event.WindowAdapter() {
            public void windowClosing(java.awt.event.WindowEvent evt) {
                thisWindowClosing(evt);
            }
        });

        // Set up the data sorter
        dataSorter = new DataSorter(trial);
        dataSorter.setSelectedMetricID(trial.getDefaultMetricID());
        dataSorter.setValueType(ValueType.EXCLUSIVE_PERCENT);

        //Set the help window text if required.
        if (ParaProf.helpWindow.isVisible()) {
            this.help(false);
        }

        setupMenus();

        //Setting up the layout system for the main window.
        getContentPane().setLayout(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);

        //Panel and ScrollPane definition.
        panel = new FunctionDataWindowPanel(trial, function, this);
        sp = new JScrollPane(panel);
        JScrollBar vScollBar = sp.getVerticalScrollBar();
        vScollBar.setUnitIncrement(35);

        this.setHeader();

        sortLocalData();

        //Slider setup.
        //Do the slider stuff, but don't add. By default, sliders are off.
        String sliderMultipleStrings[] = { "1.00", "0.75", "0.50", "0.25", "0.10" };
        sliderMultiple = new JComboBox(sliderMultipleStrings);
        sliderMultiple.addActionListener(this);

        barLengthSlider.setPaintTicks(true);
        barLengthSlider.setMajorTickSpacing(5);
        barLengthSlider.setMinorTickSpacing(1);
        barLengthSlider.setPaintLabels(true);
        barLengthSlider.setSnapToTicks(true);
        barLengthSlider.addChangeListener(this);

        // add the scrollpane
        gbc.fill = GridBagConstraints.BOTH;
        gbc.anchor = GridBagConstraints.CENTER;
        gbc.weightx = 100;
        gbc.weighty = 100;
        addCompItem(sp, gbc, 0, 0, 1, 1);
        ParaProf.incrementNumWindows();
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
            return button;
        } else {
            JMenu subSubMenu = new JMenu(valueType.toString() + "...");
            for (int i = 0; i < ppTrial.getNumberOfMetrics(); i++) {

                if (i == dataSorter.getSelectedMetricID() && enabled) {
                    button = new JRadioButtonMenuItem(ppTrial.getMetric(i).getName(), true);
                } else {
                    button = new JRadioButtonMenuItem(ppTrial.getMetric(i).getName());
                }
                final int m = i;

                button.addActionListener(new ActionListener() {
                    final int metric = m;

                    public void actionPerformed(ActionEvent evt) {
                        dataSorter.setSelectedMetricID(metric);
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

    private void setupMenus() {
        JMenuBar mainMenu = new JMenuBar();
        JMenu subMenu = null;
        JMenuItem menuItem = null;

        JMenu fileMenu = new JMenu("File");

        subMenu = new JMenu("Save ...");

        menuItem = new JMenuItem("Save Image");
        menuItem.addActionListener(this);
        subMenu.add(menuItem);

        fileMenu.add(subMenu);

        menuItem = new JMenuItem("Preferences...");
        menuItem.addActionListener(this);
        fileMenu.add(menuItem);

        menuItem = new JMenuItem("Print");
        menuItem.addActionListener(this);
        fileMenu.add(menuItem);

        menuItem = new JMenuItem("Close This Window");
        menuItem.addActionListener(this);
        fileMenu.add(menuItem);

        menuItem = new JMenuItem("Exit ParaProf!");
        menuItem.addActionListener(this);
        fileMenu.add(menuItem);

        fileMenu.addMenuListener(this);

        optionsMenu = new JMenu("Options");

        JCheckBoxMenuItem box = null;
        ButtonGroup group = null;
        JRadioButtonMenuItem button = null;

        ActionListener sortData = new ActionListener() {
            public void actionPerformed(ActionEvent evt) {
                sortLocalData();
                panel.repaint();
            }
        };

        sortByNCTCheckbox = new JCheckBoxMenuItem("Sort By N,C,T", true);
        sortByNCTCheckbox.addActionListener(sortData);
        optionsMenu.add(sortByNCTCheckbox);

        descendingOrderCheckBox = new JCheckBoxMenuItem("Descending Order", false);
        descendingOrderCheckBox.addActionListener(sortData);
        optionsMenu.add(descendingOrderCheckBox);

        showValuesAsPercent = new JCheckBoxMenuItem("Show Values as Percent", true);
        showValuesAsPercent.addActionListener(sortData);
        optionsMenu.add(showValuesAsPercent);

        unitsSubMenu = new JMenu("Select Units");
        group = new ButtonGroup();

        button = new JRadioButtonMenuItem("hr:min:sec", false);
        button.addActionListener(this);
        group.add(button);
        unitsSubMenu.add(button);

        button = new JRadioButtonMenuItem("Seconds", false);
        button.addActionListener(this);
        group.add(button);
        unitsSubMenu.add(button);

        button = new JRadioButtonMenuItem("Milliseconds", false);
        button.addActionListener(this);
        group.add(button);
        unitsSubMenu.add(button);

        button = new JRadioButtonMenuItem("Microseconds", true);
        button.addActionListener(this);
        group.add(button);
        unitsSubMenu.add(button);

        optionsMenu.add(unitsSubMenu);

        //Set the value type options.
        subMenu = new JMenu("Select Metric...");
        group = new ButtonGroup();
        
        subMenu.add(createMetricMenu(ValueType.EXCLUSIVE, dataSorter.getValueType() == ValueType.EXCLUSIVE
                || dataSorter.getValueType() == ValueType.EXCLUSIVE_PERCENT, group));
        subMenu.add(createMetricMenu(ValueType.INCLUSIVE, dataSorter.getValueType() == ValueType.INCLUSIVE
                || dataSorter.getValueType() == ValueType.INCLUSIVE_PERCENT, group));
        subMenu.add(createMetricMenu(ValueType.INCLUSIVE_PER_CALL, dataSorter.getValueType() == ValueType.INCLUSIVE_PER_CALL, group));
        subMenu.add(createMetricMenu(ValueType.EXCLUSIVE_PER_CALL, dataSorter.getValueType() == ValueType.EXCLUSIVE_PER_CALL, group));

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

        button = new JRadioButtonMenuItem("Number of Child Calls",
                dataSorter.getValueType() == ValueType.NUMSUBR);
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


        box = new JCheckBoxMenuItem("Display Sliders", false);
        box.addActionListener(this);
        optionsMenu.add(box);

        showPathTitleInReverse = new JCheckBoxMenuItem("Show Path Title in Reverse", true);
        showPathTitleInReverse.addActionListener(this);
        optionsMenu.add(showPathTitleInReverse);

        showMetaData = new JCheckBoxMenuItem("Show Meta Data in Panel", true);
        showMetaData.addActionListener(this);
        optionsMenu.add(showMetaData);

        optionsMenu.addMenuListener(this);

        // windows menu
        windowsMenu = new JMenu("Windows");

        menuItem = new JMenuItem("Show ParaProf Manager");
        menuItem.addActionListener(this);
        windowsMenu.add(menuItem);

        menuItem = new JMenuItem("Show Function Ledger");
        menuItem.addActionListener(this);
        windowsMenu.add(menuItem);

        menuItem = new JMenuItem("Show Group Ledger");
        menuItem.addActionListener(this);
        windowsMenu.add(menuItem);

        menuItem = new JMenuItem("Show User Event Ledger");
        menuItem.addActionListener(this);
        windowsMenu.add(menuItem);

        menuItem = new JMenuItem("Show Call Path Relations");
        menuItem.addActionListener(this);
        windowsMenu.add(menuItem);

        menuItem = new JMenuItem("Show Histogram");
        menuItem.addActionListener(this);
        windowsMenu.add(menuItem);

        menuItem = new JMenuItem("Close All Sub-Windows");
        menuItem.addActionListener(this);
        windowsMenu.add(menuItem);

        windowsMenu.addMenuListener(this);

        //Help menu.
        JMenu helpMenu = new JMenu("Help");

        menuItem = new JMenuItem("Show Help Window");
        menuItem.addActionListener(this);
        helpMenu.add(menuItem);

        menuItem = new JMenuItem("About ParaProf");
        menuItem.addActionListener(this);
        helpMenu.add(menuItem);

        helpMenu.addMenuListener(this);

        //Now, add all the menus to the main menu.
        mainMenu.add(fileMenu);
        mainMenu.add(optionsMenu);
        mainMenu.add(windowsMenu);
        mainMenu.add(helpMenu);

        setJMenuBar(mainMenu);
    }

    

    public void actionPerformed(ActionEvent evt) {
        try {
            Object EventSrc = evt.getSource();

            if (EventSrc instanceof JMenuItem) {
                String arg = evt.getActionCommand();
                if (arg.equals("Print")) {
                    ParaProfUtils.print(panel);
                } else if (arg.equals("Preferences...")) {
                    ppTrial.getPreferencesWindow().showPreferencesWindow();
                } else if (arg.equals("Save Image")) {
                    ParaProfImageOutput imageOutput = new ParaProfImageOutput();
                    imageOutput.saveImage((ParaProfImageInterface) panel);
                } else if (arg.equals("Close This Window")) {
                    closeThisWindow();
                } else if (arg.equals("Exit ParaProf!")) {
                    setVisible(false);
                    dispose();
                    ParaProf.exitParaProf(0);

                } else if (arg.equals("Microseconds")) {
                    units = 0;
                    this.setHeader();
                    panel.repaint();
                } else if (arg.equals("Milliseconds")) {
                    units = 1;
                    this.setHeader();
                    panel.repaint();
                } else if (arg.equals("Seconds")) {
                    units = 2;
                    this.setHeader();
                    panel.repaint();
                } else if (arg.equals("hr:min:sec")) {
                    units = 3;
                    this.setHeader();
                    panel.repaint();
                } else if (arg.equals("Display Sliders")) {
                    if (((JCheckBoxMenuItem) optionsMenu.getItem(5)).isSelected())
                        displaySliders(true);
                    else
                        displaySliders(false);

                } else if (arg.equals("Show ParaProf Manager")) {
                    (new ParaProfManagerWindow()).show();
                } else if (arg.equals("Show Path Title in Reverse"))
                    this.setTitle("Function Data Window: "
                            + ppTrial.getTrialIdentifier(showPathTitleInReverse.isSelected()));
                else if (arg.equals("Show Meta Data in Panel"))
                    this.setHeader();
                else if (arg.equals("Show Function Ledger")) {
                    (new LedgerWindow(ppTrial, 0)).show();
                } else if (arg.equals("Show Group Ledger")) {
                    (new LedgerWindow(ppTrial, 1)).show();
                } else if (arg.equals("Show User Event Ledger")) {
                    (new LedgerWindow(ppTrial, 2)).show();
                } else if (arg.equals("Show Call Path Relations")) {
                    CallPathTextWindow tmpRef = new CallPathTextWindow(ppTrial, -1, -1, -1,
                            this.getDataSorter(), 2);
                    ppTrial.getSystemEvents().addObserver(tmpRef);
                    tmpRef.show();
                } else if (arg.equals("Show Histogram")) {

                    HistogramWindow tmpRef = new HistogramWindow(ppTrial, function);
                    ppTrial.getSystemEvents().addObserver(tmpRef);
                    tmpRef.show();
                } else if (arg.equals("Close All Sub-Windows")) {
                    ppTrial.getSystemEvents().updateRegisteredObjects("subWindowCloseEvent");
                } else if (arg.equals("About ParaProf")) {
                    JOptionPane.showMessageDialog(this, ParaProf.getInfoString());
                } else if (arg.equals("Show Help Window")) {
                    this.help(true);
                } else {
                    throw new ParaProfException("Menu system not implemented properly: " + arg);
                }
            } else if (EventSrc == sliderMultiple) {
                panel.changeInMultiples();
            }
        } catch (Exception e) {
            ParaProfUtils.handleException(e);
        }
    }

    public void stateChanged(ChangeEvent event) {
        try {
            panel.changeInMultiples();
        } catch (Exception e) {
            ParaProfUtils.handleException(e);
        }
    }

    public void menuSelected(MenuEvent evt) {
        try {

            if (dataSorter.getValueType() == ValueType.EXCLUSIVE
                    || dataSorter.getValueType() == ValueType.INCLUSIVE
                    || dataSorter.getValueType() == ValueType.EXCLUSIVE_PERCENT
                    || dataSorter.getValueType() == ValueType.INCLUSIVE_PERCENT) {
                showValuesAsPercent.setEnabled(true);

                if (showValuesAsPercent.isSelected()) {
                    unitsSubMenu.setEnabled(false);
                } else {

                    String metricName = ppTrial.getMetricName(dataSorter.getSelectedMetricID());
                    metricName = metricName.toUpperCase();
                    if (dataSorter.isTimeMetric())
                        unitsSubMenu.setEnabled(true);
                    else
                        unitsSubMenu.setEnabled(false);
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

            if (ppTrial.groupNamesPresent())
                ((JMenuItem) windowsMenu.getItem(2)).setEnabled(true);
            else
                ((JMenuItem) windowsMenu.getItem(2)).setEnabled(false);

            if (ppTrial.userEventsPresent())
                ((JMenuItem) windowsMenu.getItem(3)).setEnabled(true);
            else
                ((JMenuItem) windowsMenu.getItem(3)).setEnabled(false);

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

    private void help(boolean display) {
        //Show the ParaProf help window.
        ParaProf.helpWindow.clearText();
        if (display)
            ParaProf.helpWindow.show();
        ParaProf.helpWindow.writeText("This is the function data window for:");
        ParaProf.helpWindow.writeText(function.getName());
        ParaProf.helpWindow.writeText("");
        ParaProf.helpWindow.writeText("This window shows you this function's statistics across all the threads.");
        ParaProf.helpWindow.writeText("");
        ParaProf.helpWindow.writeText("Use the options menu to select different ways of displaying the data.");
        ParaProf.helpWindow.writeText("");
        ParaProf.helpWindow.writeText("Right click anywhere within this window to bring up a popup");
        ParaProf.helpWindow.writeText("menu. In this menu you can change or reset the default color");
        ParaProf.helpWindow.writeText("for this function.");
    }

    public DataSorter getDataSorter() {
        return dataSorter;
    }

    public void sortLocalData() {

        if (sortByNCTCheckbox.isSelected()) {
            dataSorter.setSortType(SortType.NCT);
        } else {
            dataSorter.setSortType(SortType.VALUE);
        }

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

        list = dataSorter.getFunctionData(function, true);

        maxValue = 0;
        for (Iterator it = list.iterator(); it.hasNext();) {
            PPFunctionProfile ppFunctionProfile = (PPFunctionProfile) it.next();
            double value = ppFunctionProfile.getValue();
            //            double value = ParaProfUtils.getValue(ppFunctionProfile, this.getValueType(), this.isPercent());
            maxValue = Math.max(maxValue, value);
        }
    }

    public double getMaxValue() {
        return maxValue;
    }

    public Vector getData() {
        return list;
    }

    public boolean isPercent() {
        return showValuesAsPercent.isSelected();
    }

    public int units() {
        if (showValuesAsPercent.isSelected())
            return 0;

        if (!dataSorter.isTimeMetric()) // we don't do units for non-time metrics
            return 0;

        if (dataSorter.getValueType() == ValueType.NUMCALLS || dataSorter.getValueType() == ValueType.NUMSUBR)
            return 0;

        return units;
    }

    public Dimension getViewportSize() {
        return sp.getViewport().getExtentSize();
    }

    public Rectangle getViewRect() {
        return sp.getViewport().getViewRect();
    }

    public void setVerticalScrollBarPosition(int position) {
        JScrollBar scrollBar = sp.getVerticalScrollBar();
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
            PreferencesWindow p = ppTrial.getPreferencesWindow();
            jTextArea.setFont(new Font(p.getParaProfFont(), p.getFontStyle(), p.getFontSize()));
            jTextArea.append(this.getHeaderString());
            sp.setColumnHeaderView(jTextArea);
        } else
            sp.setColumnHeaderView(null);
    }

    public String getHeaderString() {
        if ((dataSorter.getValueType() == ValueType.NUMCALLS || dataSorter.getValueType() == ValueType.NUMSUBR)
                || showValuesAsPercent.isSelected())
            return "Metric Name: " + (ppTrial.getMetricName(dataSorter.getSelectedMetricID())) + "\n"
                    + "Name: " + function.getName() + "\n" + "Value Type: " + dataSorter.getValueType() + "\n";
        else
            return "Metric Name: " + (ppTrial.getMetricName(dataSorter.getSelectedMetricID())) + "\n"
                    + "Name: " + function.getName() + "\n" + "Value Type: " + dataSorter.getValueType() + "\n"
                    + "Units: "
                    + UtilFncs.getUnitsString(units, dataSorter.isTimeMetric(), dataSorter.isDerivedMetric())
                    + "\n";
    }

    public int getSliderValue() {
        int tmpInt = -1;
        tmpInt = barLengthSlider.getValue();
        return tmpInt;
    }

    public double getSliderMultiple() {
        String tmpString = null;
        tmpString = (String) sliderMultiple.getSelectedItem();
        return Double.parseDouble(tmpString);
    }

    private void displaySliders(boolean displaySliders) {
        GridBagConstraints gbc = new GridBagConstraints();
        if (displaySliders) {
            getContentPane().remove(sp);

            gbc.fill = GridBagConstraints.NONE;
            gbc.anchor = GridBagConstraints.EAST;
            gbc.weightx = 0;
            gbc.weighty = 0;
            addCompItem(sliderMultipleLabel, gbc, 0, 0, 1, 1);

            gbc.fill = GridBagConstraints.NONE;
            gbc.anchor = GridBagConstraints.WEST;
            gbc.weightx = 100;
            gbc.weighty = 0;
            addCompItem(sliderMultiple, gbc, 1, 0, 1, 1);

            gbc.fill = GridBagConstraints.NONE;
            gbc.anchor = GridBagConstraints.EAST;
            gbc.weightx = 0;
            gbc.weighty = 0;
            addCompItem(barLengthLabel, gbc, 2, 0, 1, 1);

            gbc.fill = GridBagConstraints.HORIZONTAL;
            gbc.anchor = GridBagConstraints.WEST;
            gbc.weightx = 100;
            gbc.weighty = 0;
            addCompItem(barLengthSlider, gbc, 3, 0, 1, 1);

            gbc.fill = GridBagConstraints.BOTH;
            gbc.anchor = GridBagConstraints.CENTER;
            gbc.weightx = 100;
            gbc.weighty = 100;
            addCompItem(sp, gbc, 0, 1, 4, 1);
        } else {
            getContentPane().remove(sliderMultipleLabel);
            getContentPane().remove(sliderMultiple);
            getContentPane().remove(barLengthLabel);
            getContentPane().remove(barLengthSlider);
            getContentPane().remove(sp);

            gbc.fill = GridBagConstraints.BOTH;
            gbc.anchor = GridBagConstraints.CENTER;
            gbc.weightx = 100;
            gbc.weighty = 100;
            addCompItem(sp, gbc, 0, 0, 1, 1);
        }

        //Now call validate so that these componant changes are displayed.
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
            ppTrial.getSystemEvents().deleteObserver(this);
            ParaProf.decrementNumWindows();
        } catch (Exception e) {
            // do nothing
        }
        dispose();
    }

    private ParaProfTrial ppTrial = null;
    private DataSorter dataSorter = null;

    private Function function = null;

    private JMenu optionsMenu = null;
    private JMenu windowsMenu = null;
    private JMenu unitsSubMenu = null;

    private JCheckBoxMenuItem sortByNCTCheckbox = null;
    private JCheckBoxMenuItem descendingOrderCheckBox = null;
    private JCheckBoxMenuItem showValuesAsPercent = null;
    private JCheckBoxMenuItem displaySliders = null;
    private JCheckBoxMenuItem showPathTitleInReverse = null;
    private JCheckBoxMenuItem showMetaData = null;

    private JLabel sliderMultipleLabel = new JLabel("Slider Multiple");
    private JComboBox sliderMultiple;
    private JLabel barLengthLabel = new JLabel("Bar Multiple");
    private JSlider barLengthSlider = new JSlider(0, 40, 1);

    private FunctionDataWindowPanel panel = null;
    private JScrollPane sp = null;

    private Vector list = null;

    private double maxValue;

    private int units = 0; //0-microseconds,1-milliseconds,2-seconds.
}