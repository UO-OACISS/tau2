package edu.uoregon.tau.paraprof;

import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.*;

import javax.swing.*;
import javax.swing.event.*;

import edu.uoregon.tau.dms.dss.Function;
import edu.uoregon.tau.dms.dss.UtilFncs;
import edu.uoregon.tau.paraprof.enums.SortType;
import edu.uoregon.tau.paraprof.enums.ValueType;
import edu.uoregon.tau.paraprof.interfaces.ParaProfWindow;
import edu.uoregon.tau.paraprof.interfaces.UnitListener;

/**
 * FunctionDataWindow
 * This is the FunctionDataWindow.
 *  
 * <P>CVS $Id: FunctionDataWindow.java,v 1.21 2005/05/10 01:48:37 amorris Exp $</P>
 * @author	Robert Bell, Alan Morris
 * @version	$Revision: 1.21 $
 * @see		FunctionDataWindowPanel
 */
public class FunctionDataWindow extends JFrame implements ActionListener, MenuListener, Observer,
        ChangeListener, ParaProfWindow, UnitListener {


    private ParaProfTrial ppTrial = null;
    private DataSorter dataSorter = null;

    private Function function = null;

    private JMenu optionsMenu = null;
    private JMenu unitsSubMenu = null;

    private JCheckBoxMenuItem sortByNCTCheckbox = null;
    private JCheckBoxMenuItem descendingOrderCheckBox = null;
    private JCheckBoxMenuItem showValuesAsPercent = null;
    private JCheckBoxMenuItem showPathTitleInReverse = null;
    private JCheckBoxMenuItem showMetaData = null;

    private JLabel barLengthLabel = new JLabel("Bar Width");
    private JSlider barLengthSlider = new JSlider(0, 2000, 250);

    private FunctionDataWindowPanel panel = null;
    private JScrollPane sp = null;

    private Vector list = new Vector();

    private double maxValue;
    private int units = ParaProf.preferences.getUnits();

    public FunctionDataWindow(ParaProfTrial trial, Function function) {
        this.ppTrial = trial;

        this.function = function;
        int windowWidth = 650;
        int windowHeight = 550;
        setSize(new java.awt.Dimension(windowWidth, windowHeight));

        //Now set the title.
        this.setTitle("Function Data Window: " + trial.getTrialIdentifier(ParaProf.preferences.getShowPathTitleInReverse()));

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


        //Setting up the layout system for the main window.
        getContentPane().setLayout(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);

        //Panel and ScrollPane definition.
        panel = new FunctionDataWindowPanel(trial, function, this);
        sp = new JScrollPane(panel);
        JScrollBar vScrollBar = sp.getVerticalScrollBar();
        vScrollBar.setUnitIncrement(35);

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

    private void setupMenus() {
        JMenuBar mainMenu = new JMenuBar();
        JMenu subMenu = null;
        JMenuItem menuItem = null;

        

        optionsMenu = new JMenu("Options");

        JCheckBoxMenuItem box = null;
        ButtonGroup group = null;
        JRadioButtonMenuItem button = null;

        box = new JCheckBoxMenuItem("Show Width Slider", false);
        box.addActionListener(this);
        optionsMenu.add(box);

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

        sortByNCTCheckbox = new JCheckBoxMenuItem("Sort By N,C,T", true);
        sortByNCTCheckbox.addActionListener(sortData);
        optionsMenu.add(sortByNCTCheckbox);

        descendingOrderCheckBox = new JCheckBoxMenuItem("Descending Order", false);
        descendingOrderCheckBox.addActionListener(sortData);
        optionsMenu.add(descendingOrderCheckBox);

        showValuesAsPercent = new JCheckBoxMenuItem("Show Values as Percent", ParaProf.preferences.getShowValuesAsPercent());
        showValuesAsPercent.addActionListener(sortData);
        optionsMenu.add(showValuesAsPercent);

        unitsSubMenu = ParaProfUtils.createUnitsMenu(this, units);
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
                    if (((JCheckBoxMenuItem) EventSrc).isSelected())
                        showWidthSlider(true);
                    else
                        showWidthSlider(false);
                } else if (arg.equals("Show Meta Data in Panel")) {
                    this.setHeader();
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
            panel.setBarLength(barLengthSlider.getValue());
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
        if (display)
            ParaProf.helpWindow.show();
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
        if (showValuesAsPercent.isEnabled() && showValuesAsPercent.isSelected())
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
                    + "Name: " + ParaProfUtils.getFunctionName(function) + "\n" + "Value Type: " + dataSorter.getValueType() + "\n";
        else
            return "Metric Name: " + (ppTrial.getMetricName(dataSorter.getSelectedMetricID())) + "\n"
                    + "Name: " + ParaProfUtils.getFunctionName(function) + "\n" + "Value Type: " + dataSorter.getValueType() + "\n"
                    + "Units: "
                    + UtilFncs.getUnitsString(units, dataSorter.isTimeMetric(), dataSorter.isDerivedMetric())
                    + "\n";
    }


    private void showWidthSlider(boolean displaySliders) {
        GridBagConstraints gbc = new GridBagConstraints();
        if (displaySliders) {
            getContentPane().remove(sp);


            gbc.insets = new Insets(5,5,5,5);
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

            gbc.insets = new Insets(0,0,0,0);
            gbc.fill = GridBagConstraints.BOTH;
            gbc.anchor = GridBagConstraints.CENTER;
            gbc.weightx = 1.0;
            gbc.weighty = 0.99;
            addCompItem(sp, gbc, 0, 1, 2, 1);
        } else {
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

    public void setUnits(int units) {
        this.units = units;
        this.setHeader();
        panel.repaint();
    }
}