/* 
 ThreadDataWindow.java

 Title:      ParaProf
 Author:     Robert Bell
 Description:  The container for the FunctionDataWindowPanel.
 */

/*
 To do: 
 1) Change the name of this class to reflect the fact that it handles more than
 just thread displays.

 2) Update the help text for this window.
 
 3) Add some comments to some of the code.
 */

package edu.uoregon.tau.paraprof;

import java.awt.*;
import java.awt.event.*;
import java.util.Observable;
import java.util.Observer;
import java.util.Vector;

import javax.swing.*;
import javax.swing.event.*;

import edu.uoregon.tau.dms.dss.UtilFncs;
import edu.uoregon.tau.paraprof.enums.SortType;
import edu.uoregon.tau.paraprof.enums.ValueType;
import edu.uoregon.tau.paraprof.interfaces.*;
import edu.uoregon.tau.paraprof.interfaces.ParaProfWindow;
import edu.uoregon.tau.paraprof.interfaces.ScrollBarController;
import edu.uoregon.tau.paraprof.interfaces.SearchableOwner;

public class ThreadDataWindow extends JFrame implements ActionListener, MenuListener, Observer, ChangeListener,
        KeyListener, SearchableOwner, ScrollBarController, ParaProfWindow, UnitListener {

    
    private PPThread ppThread;
    private ParaProfTrial ppTrial = null;
    private DataSorter dataSorter = null;

    private JMenu optionsMenu = null;
    private JMenu unitsSubMenu = null;

    private JCheckBoxMenuItem sortByName = null;
    private JCheckBoxMenuItem descendingOrder = null;
    private JCheckBoxMenuItem showValuesAsPercent = null;
    private JCheckBoxMenuItem showPathTitleInReverse = null;
    private JCheckBoxMenuItem showMetaData = null;
    private JCheckBoxMenuItem showFindPanelBox;
    
    private JLabel barLengthLabel = new JLabel("Bar Width");
    private JSlider barLengthSlider = new JSlider(0, 2000, 250);

    private Container contentPane = null;
    private GridBagLayout gbl = null;
    private GridBagConstraints gbc = null;

    private JScrollPane jScrollPane;
    private ThreadDataWindowPanel panel;

    private Vector list = new Vector();

    private boolean percent = ParaProf.preferences.getShowValuesAsPercent();
    private int units = ParaProf.preferences.getUnits();

    private SearchPanel searchPanel;

    // for derived metrics the exclusive could be higher than the inclusive, so the percent
    // will be higher than 100.  This may confuse users so we disable showing percentages if one
    // goes over 100.
    private boolean exclusivePercentOver100 = false;

    
    public ThreadDataWindow(ParaProfTrial trial, int nodeID, int contextID, int threadID) {

        this.ppTrial = trial;
        dataSorter = new DataSorter(trial);
        dataSorter.setSelectedMetricID(trial.getDefaultMetricID());
        dataSorter.setValueType(ValueType.EXCLUSIVE_PERCENT);

        if (nodeID == -1) { // if this is a 'mean' window
            edu.uoregon.tau.dms.dss.Thread thread;
            thread = trial.getDataSource().getMeanData();
            ppThread = new PPThread(thread, trial);
        } else {
            edu.uoregon.tau.dms.dss.Thread thread;
            thread = trial.getDataSource().getThread(nodeID, contextID, threadID);
            ppThread = new PPThread(thread, trial);
        }

        //setLocation(new java.awt.Point(300, 200));
        setSize(new java.awt.Dimension(700, 450));

        //Now set the title.
        if (nodeID == -1)
            this.setTitle("Mean Data Window: " + trial.getTrialIdentifier(ParaProf.preferences.getShowPathTitleInReverse()));
        else
            this.setTitle("n,c,t, " + nodeID + "," + contextID + "," + threadID + " - "
                    + trial.getTrialIdentifier(ParaProf.preferences.getShowPathTitleInReverse()));

        //Add some window listener code
        addWindowListener(new java.awt.event.WindowAdapter() {
            public void windowClosing(java.awt.event.WindowEvent evt) {
                thisWindowClosing(evt);
            }
        });

        //Set the help window text if required.new JPanel()
        if (ParaProf.helpWindow.isVisible()) {
            this.help(false);
        }

        setupMenus();

        contentPane = getContentPane();
        contentPane.setLayout(new GridBagLayout());
        gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 0, 5);

        panel = new ThreadDataWindowPanel(trial, nodeID, contextID, threadID, this);
        this.addKeyListener(this);

        jScrollPane = new JScrollPane(panel);

        jScrollPane.setAutoscrolls(false);
        JScrollBar vScrollBar = jScrollPane.getVerticalScrollBar();
        vScrollBar.setUnitIncrement(35);

        this.setHeader();

        requestFocus();

        barLengthSlider.setPaintTicks(true);
        barLengthSlider.setMajorTickSpacing(400);
        barLengthSlider.setMinorTickSpacing(50);
        barLengthSlider.setPaintLabels(true);
        barLengthSlider.setSnapToTicks(false);
        barLengthSlider.addChangeListener(this);

        gbc.fill = GridBagConstraints.BOTH;
        gbc.anchor = GridBagConstraints.CENTER;
        gbc.weightx = 0.95;
        gbc.weighty = 0.98;
        addCompItem(jScrollPane, gbc, 0, 1, 2, 1);

        sortLocalData();

        ParaProf.incrementNumWindows();
    }

    // this is copied from FunctionDataWindow
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
        mainMenu.addKeyListener(this);
        JMenu subMenu = null;
        JMenuItem menuItem = null;


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
        
        sortByName = new JCheckBoxMenuItem("Sort By Name", false);
        sortByName.addActionListener(this);
        optionsMenu.add(sortByName);

        descendingOrder = new JCheckBoxMenuItem("Descending Order", true);
        descendingOrder.addActionListener(this);
        optionsMenu.add(descendingOrder);

        showValuesAsPercent = new JCheckBoxMenuItem("Show Values as Percent", ParaProf.preferences.getShowValuesAsPercent());
        showValuesAsPercent.addActionListener(this);
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
        subMenu.add(createMetricMenu(ValueType.INCLUSIVE_PER_CALL,
                dataSorter.getValueType() == ValueType.INCLUSIVE_PER_CALL, group));
        subMenu.add(createMetricMenu(ValueType.EXCLUSIVE_PER_CALL,
                dataSorter.getValueType() == ValueType.EXCLUSIVE_PER_CALL, group));

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
        mainMenu.add(ParaProfUtils.createTrialMenu(ppTrial, this));
        mainMenu.add(ParaProfUtils.createWindowsMenu(ppTrial, this));
        mainMenu.add(ParaProfUtils.createHelpMenu(this, this));

        setJMenuBar(mainMenu);
        //####################################
        //End - Code to generate the menus.
        //####################################

    }

    //####################################
    //Interface code.
    //####################################

    //######
    //ActionListener.
    //######
    public void actionPerformed(ActionEvent evt) {
        try {
            Object EventSrc = evt.getSource();

            if (EventSrc instanceof JMenuItem) {
                String arg = evt.getActionCommand();
                if (arg.equals("Sort By Name")) {
                    sortLocalData();
                    panel.repaint();
                } else if (arg.equals("Descending Order")) {
                    sortLocalData();
                    panel.repaint();
                } else if (arg.equals("Show Values as Percent")) {
                    if (showValuesAsPercent.isSelected()) {
                        percent = true;
                        //units = 0;
                    } else
                        percent = false;
                    this.setHeader();
                    sortLocalData();
                    panel.repaint();
                } else if (arg.equals("Exclusive")) {

                    // turn off showing percentages when exclusive will go over 100%
                    if (exclusivePercentOver100) {
                        percent = false;
                        showValuesAsPercent.setEnabled(false);
                        showValuesAsPercent.setSelected(false);
                    }

                    dataSorter.setValueType(ValueType.EXCLUSIVE);
                    this.setHeader();
                    sortLocalData();
                    panel.repaint();
                } else if (arg.equals("Inclusive")) {
                    dataSorter.setValueType(ValueType.INCLUSIVE);
                    this.setHeader();
                    sortLocalData();
                    panel.repaint();
                } else if (arg.equals("Number of Calls")) {
                    dataSorter.setValueType(ValueType.NUMCALLS);
                    this.setHeader();
                    sortLocalData();
                    panel.repaint();
                } else if (arg.equals("Number of Child Calls")) {
                    dataSorter.setValueType(ValueType.NUMSUBR);
                    this.setHeader();
                    sortLocalData();
                    panel.repaint();
                } else if (arg.equals("Inclusive per Call")) {
                    dataSorter.setValueType(ValueType.INCLUSIVE_PER_CALL);
                    this.setHeader();
                    sortLocalData();
                    panel.repaint();
                } else if (arg.equals("Exclusive per Call")) {
                    dataSorter.setValueType(ValueType.EXCLUSIVE_PER_CALL);
                    this.setHeader();
                    sortLocalData();
                    panel.repaint();
                } else if (arg.equals("Show Width Slider")) {
                    if (((JCheckBoxMenuItem) EventSrc).isSelected())
                        showWidthSlider(true);
                    else
                        showWidthSlider(false);
                } else if (arg.equals("Show Find Panel")) {
                    if (showFindPanelBox.isSelected())
                        showSearchPanel(true);
                    else
                        showSearchPanel(false);

                } else if (arg.equals("Show Path Title in Reverse")) {
                    if (ppThread.getNodeID() == -1)
                        this.setTitle("Mean Data Window: "
                                + ppTrial.getTrialIdentifier(showPathTitleInReverse.isSelected()));
                    else
                        this.setTitle("n,c,t, " + ppThread.getNodeID() + "," + ppThread.getContextID() + ","
                                + ppThread.getThreadID() + " - "
                                + ppTrial.getTrialIdentifier(showPathTitleInReverse.isSelected()));
                } else if (arg.equals("Show Meta Data in Panel")) {
                    this.setHeader();
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

            // turn off showing percentages when exclusive will go over 100%
            if (dataSorter.getValueType() == ValueType.EXCLUSIVE) {
                if (exclusivePercentOver100) {
                    percent = false;
                    showValuesAsPercent.setEnabled(false);
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
            sortLocalData();
            if (!(ppTrial.isTimeMetric()))
                units = 0;
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

    public DataSorter getDataSorter() {
        return dataSorter;
    }

    //Updates this window's data copy.
    private void sortLocalData() {
        //The name selection behaves slightly differently. Thus the check for it.
        //        if (name) {
        //            list = dataSorter.getFunctionProfiles(nodeID, contextID, threadID, order);
        //        } else {
        //            list = dataSorter.getFunctionProfiles(nodeID, contextID, threadID, valueType + order);
        //        }

        dataSorter.setDescendingOrder(descendingOrder.isSelected());

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

        if (sortByName.isSelected()) {
            dataSorter.setSortType(SortType.NAME);
        } else {
            dataSorter.setSortType(SortType.VALUE);
        }
        

        list = ppThread.getSortedFunctionProfiles(this.dataSorter, false);

        if (ppThread.getMaxExclusivePercent() > 100) {
            exclusivePercentOver100 = true;
            percent = false;
        }

        panel.resetStringSize();

    }

    public Vector getData() {
        return list;
    }

    public boolean isPercent() {
        return percent;
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
        return jScrollPane.getViewport().getExtentSize();
    }

    public Rectangle getViewRect() {
        return jScrollPane.getViewport().getViewRect();
    }

    public void setVerticalScrollBarPosition(int position) {
        JScrollBar scrollBar = jScrollPane.getVerticalScrollBar();
        scrollBar.setValue(position);
    }

    public void setHorizontalScrollBarPosition(int position) {
        JScrollBar scrollBar = jScrollPane.getHorizontalScrollBar();
        scrollBar.setValue(position);
    }

    
    //######
    //Panel header.
    //######
    //This process is separated into two functionProfiles to provide the option
    //of obtaining the current header string being used for the panel
    //without resetting the actual header. Printing and image generation
    //use this functionality for example.
    public void setHeader() {
        if (showMetaData.isSelected()) {
            JTextArea jTextArea = new JTextArea();
            jTextArea.setLineWrap(true);
            jTextArea.setWrapStyleWord(true);
            jTextArea.setEditable(false);
            PreferencesWindow p = ppTrial.getPreferencesWindow();
            jTextArea.setFont(new Font(p.getParaProfFont(), p.getFontStyle(), p.getFontSize()));
            jTextArea.append(this.getHeaderString());
            jScrollPane.setColumnHeaderView(jTextArea);

            jTextArea.addKeyListener(this);
        } else
            jScrollPane.setColumnHeaderView(null);
    }

    public String getHeaderString() {
        if ((dataSorter.getValueType() == ValueType.NUMCALLS || dataSorter.getValueType() == ValueType.NUMSUBR)
                || showValuesAsPercent.isSelected())
            return "Metric Name: " + (ppTrial.getMetricName(dataSorter.getSelectedMetricID())) + "\n"
                    + "Value Type: " + dataSorter.getValueType() + "\n";
        else
            return "Metric Name: " + (ppTrial.getMetricName(dataSorter.getSelectedMetricID())) + "\n"
                    + "Value Type: " + dataSorter.getValueType() + "\n" + "Units: "
                    + UtilFncs.getUnitsString(units, dataSorter.isTimeMetric(), dataSorter.isDerivedMetric())
                    + "\n";
    }

    public void showSearchPanel(boolean show) {
        if (show) {
            if (searchPanel == null) {
                searchPanel = new SearchPanel(this, panel.getSearcher());
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

    
    private void showWidthSlider(boolean displaySliders) {
        GridBagConstraints gbc = new GridBagConstraints();
        if (displaySliders) {

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

        } else {
            getContentPane().remove(barLengthLabel);
            getContentPane().remove(barLengthSlider);
        }

        //Now call validate so that these component changes are displayed.
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

    //Instance data.

    public PPThread getPPThread() {
        return ppThread;
    }


    public JScrollPane getJScrollPane() {
        return jScrollPane;
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

    public Dimension getThisViewportSize() {
        return getViewportSize();
    }

    public void setUnits(int units) {
        this.units = units;
        this.setHeader();
        panel.repaint();
    }
}