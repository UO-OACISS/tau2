/*
 * StatWindow.java
 * 
 * Title: ParaProf Author: Robert Bell Description:
 */

package edu.uoregon.tau.paraprof;

import java.util.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.event.*;
import java.awt.print.*;
import edu.uoregon.tau.dms.dss.*;

public class StatWindow extends JFrame implements ActionListener, MenuListener, Observer {

    public StatWindow(ParaProfTrial trial, int nodeID, int contextID, int threadID, DataSorter dataSorter,
            boolean userEventWindow) {
        this.trial = trial;
        this.dataSorter = dataSorter;
        this.nodeID = nodeID;
        this.contextID = contextID;
        this.threadID = threadID;
        this.userEventWindow = userEventWindow;

        setLocation(new java.awt.Point(0, 0));
        setSize(new java.awt.Dimension(1000, 600));

        if (nodeID == -1 && userEventWindow) {
            // There is no User Event data for mean
            throw new ParaProfException("There is no User Event data for mean");
        }

        //Now set the title.
        if (nodeID == -1)
            this.setTitle("Mean Data Statistics: " + trial.getTrialIdentifier(true));
        else {
            if (userEventWindow)
                valueType = 12;
            this.setTitle("n,c,t, " + nodeID + "," + contextID + "," + threadID + " - "
                    + trial.getTrialIdentifier(true));
        }

        //Add some window listener code
        addWindowListener(new java.awt.event.WindowAdapter() {
            public void windowClosing(java.awt.event.WindowEvent evt) {
                thisWindowClosing(evt);
            }
        });

        //Set the help window text if required.
        if (ParaProf.helpWindow.isVisible()) {
            this.help(false);
        }

        //Sort the local data.
        sortLocalData();

        //####################################
        //Code to generate the menus.
        //####################################
        JMenuBar mainMenu = new JMenuBar();
        JMenu subMenu = null;
        JMenuItem menuItem = null;

        //######
        //File menu.
        //######
        JMenu fileMenu = new JMenu("File");

        //Save menu.
        subMenu = new JMenu("Save ...");

        /*
         * menuItem = new JMenuItem("ParaProf Preferences");
         * menuItem.addActionListener(this); subMenu.add(menuItem);
         */

        menuItem = new JMenuItem("Save Image");
        menuItem.addActionListener(this);
        subMenu.add(menuItem);

        fileMenu.add(subMenu);
        //End - Save menu.

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
        //######
        //End - File menu.
        //######

        //######
        //Options menu.
        //######
        optionsMenu = new JMenu("Options");

        JCheckBoxMenuItem box = null;
        ButtonGroup group = null;
        JRadioButtonMenuItem button = null;

        sortByName = new JCheckBoxMenuItem("Sort By Name", false);
        sortByName.addActionListener(this);
        optionsMenu.add(sortByName);

        descendingOrder = new JCheckBoxMenuItem("Descending Order", true);
        descendingOrder.addActionListener(this);
        optionsMenu.add(descendingOrder);

        //Units submenu.
        unitsSubMenu = new JMenu("Select Units");
        if (!userEventWindow) {
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
        }
        optionsMenu.add(unitsSubMenu);
        //End - Units submenu.

        //Set the value type options.
        subMenu = new JMenu("Sort By");
        group = new ButtonGroup();

        if (userEventWindow) {
            button = new JRadioButtonMenuItem("Number of Userevents", true);
            button.addActionListener(this);
            group.add(button);
            subMenu.add(button);

            button = new JRadioButtonMenuItem("Min. Value", false);
            button.addActionListener(this);
            group.add(button);
            subMenu.add(button);

            button = new JRadioButtonMenuItem("Max. Value", false);
            button.addActionListener(this);
            group.add(button);
            subMenu.add(button);

            button = new JRadioButtonMenuItem("Mean Value", false);
            button.addActionListener(this);
            group.add(button);
            subMenu.add(button);
        } else {
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

            button = new JRadioButtonMenuItem("Number of Subroutines", false);
            button.addActionListener(this);
            group.add(button);
            subMenu.add(button);

            button = new JRadioButtonMenuItem("Per Call Value", false);
            button.addActionListener(this);
            group.add(button);
        }
        subMenu.add(button);
        optionsMenu.add(subMenu);
        //End - Set the value type options.

        showPathTitleInReverse = new JCheckBoxMenuItem("Show Path Title in Reverse", true);
        showPathTitleInReverse.addActionListener(this);
        optionsMenu.add(showPathTitleInReverse);

        showMetaData = new JCheckBoxMenuItem("Show Meta Data in Panel", true);
        showMetaData.addActionListener(this);
        optionsMenu.add(showMetaData);

        optionsMenu.addMenuListener(this);
        //######
        //End - Options menu.
        //######

        //######
        //Windows menu
        //######
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

        menuItem = new JMenuItem("Close All Sub-Windows");
        menuItem.addActionListener(this);
        windowsMenu.add(menuItem);

        windowsMenu.addMenuListener(this);
        //######
        //End - Windows menu
        //######

        //######
        //Help menu.
        //######
        JMenu helpMenu = new JMenu("Help");

        menuItem = new JMenuItem("Show Help Window");
        menuItem.addActionListener(this);
        helpMenu.add(menuItem);

        menuItem = new JMenuItem("About ParaProf");
        menuItem.addActionListener(this);
        helpMenu.add(menuItem);

        helpMenu.addMenuListener(this);
        //######
        //End - Help menu.
        //######

        //Now, add all the menus to the main menu.
        mainMenu.add(fileMenu);
        mainMenu.add(optionsMenu);
        mainMenu.add(windowsMenu);
        mainMenu.add(helpMenu);

        setJMenuBar(mainMenu);
        //####################################
        //End - Code to generate the menus.
        //####################################

        //####################################
        //Create and add the components.
        //####################################
        //Setting up the layout system for the main window.
        Container contentPane = getContentPane();
        GridBagLayout gbl = new GridBagLayout();
        contentPane.setLayout(gbl);
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);

        //######
        //Panel and ScrollPane definition.
        //######
        panel = new StatWindowPanel(trial, nodeID, contextID, threadID, this, userEventWindow);
        sp = new JScrollPane(panel);
        
        JScrollBar jScrollBar = sp.getVerticalScrollBar();
        jScrollBar.setUnitIncrement(35);

        this.setHeader();
        //######
        //End - Panel and ScrollPane definition.
        //######

        //Now add the components to the main screen.
        gbc.fill = GridBagConstraints.BOTH;
        gbc.anchor = GridBagConstraints.CENTER;
        gbc.weightx = 1;
        gbc.weighty = 1;
        addCompItem(sp, gbc, 0, 0, 1, 1);
        //####################################
        //End - Create and add the components.
        //####################################
        ParaProf.incrementNumWindows();
    }

    public void actionPerformed(ActionEvent evt) {
        try {
            Object EventSrc = evt.getSource();

            if (EventSrc instanceof JMenuItem) {
                String arg = evt.getActionCommand();
                if (arg.equals("Print")) {
                    ParaProfUtils.print(panel);
                } else if (arg.equals("Preferences...")) {
                    trial.getPreferences().showPreferencesWindow();
                } else if (arg.equals("Save Image")) {
                    ParaProfImageOutput imageOutput = new ParaProfImageOutput();
                    imageOutput.saveImage((ParaProfImageInterface) panel);
                } else if (arg.equals("Close This Window")) {
                    closeThisWindow();
                } else if (arg.equals("Exit ParaProf!")) {
                    setVisible(false);
                    dispose();
                    ParaProf.exitParaProf(0);
                } else if (arg.equals("Sort By Name")) {
                    if (sortByName.isSelected())
                        name = true;
                    else
                        name = false;
                    sortLocalData();
                    panel.repaint();
                } else if (arg.equals("Descending Order")) {
                    if (descendingOrder.isSelected())
                        order = 0;
                    else
                        order = 1;
                    sortLocalData();
                    panel.repaint();
                } else if (arg.equals("Exclusive")) {
                    valueType = 2;
                    this.setHeader();
                    sortLocalData();
                    panel.repaint();
                } else if (arg.equals("Inclusive")) {
                    valueType = 4;
                    this.setHeader();
                    sortLocalData();
                    panel.repaint();
                } else if (arg.equals("Number of Calls")) {
                    valueType = 6;
                    this.setHeader();
                    sortLocalData();
                    panel.repaint();
                } else if (arg.equals("Number of Subroutines")) {
                    valueType = 8;
                    this.setHeader();
                    sortLocalData();
                    panel.repaint();
                } else if (arg.equals("Per Call Value")) {
                    valueType = 10;
                    this.setHeader();
                    sortLocalData();
                    panel.repaint();
                } else if (arg.equals("Number of Userevents")) {
                    valueType = 12;
                    this.setHeader();
                    sortLocalData();
                    panel.repaint();
                } else if (arg.equals("Min. Value")) {
                    valueType = 14;
                    this.setHeader();
                    sortLocalData();
                    panel.repaint();
                } else if (arg.equals("Max. Value")) {
                    valueType = 16;
                    this.setHeader();
                    sortLocalData();
                    panel.repaint();
                } else if (arg.equals("Mean Value")) {
                    valueType = 18;
                    this.setHeader();
                    sortLocalData();
                    panel.repaint();
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
                } else if (arg.equals("Show Path Title in Reverse")) {
                    if (nodeID == -1)
                        this.setTitle("Mean Data Window Total: "
                                + trial.getTrialIdentifier(showPathTitleInReverse.isSelected()));
                    else
                        this.setTitle("Total " + "n,c,t, " + nodeID + "," + contextID + "," + threadID + " - "
                                + trial.getTrialIdentifier(showPathTitleInReverse.isSelected()));
                } else if (arg.equals("Show Meta Data in Panel")) {
                    this.setHeader(); 
                } else if (arg.equals("Show ParaProf Manager")) {
                    (new ParaProfManagerWindow()).show();
                } else if (arg.equals("Show Function Ledger")) {
                    (new LedgerWindow(trial, 0)).show();
                } else if (arg.equals("Show Group Ledger")) {
                    (new LedgerWindow(trial, 1)).show();
                } else if (arg.equals("Show User Event Ledger")) {
                    (new LedgerWindow(trial, 2)).show();
                } else if (arg.equals("Show Call Path Relations")) {
                    CallPathTextWindow tmpRef = new CallPathTextWindow(trial, -1, -1, -1, this.getDataSorter(),
                            2);
                    trial.getSystemEvents().addObserver(tmpRef);
                    tmpRef.show();
                } else if (arg.equals("Close All Sub-Windows")) {
                    trial.getSystemEvents().updateRegisteredObjects("subWindowCloseEvent");
                } else if (arg.equals("About ParaProf")) {
                    JOptionPane.showMessageDialog(this, ParaProf.getInfoString());
                } else if (arg.equals("Show Help Window")) {
                    this.help(true);
                }
            }
        } catch (Exception e) {
            ParaProfUtils.handleException(e);
        }
    }

    public void menuSelected(MenuEvent evt) {
        try {
            if (trial.isTimeMetric() && !userEventWindow)
                unitsSubMenu.setEnabled(true);
            else
                unitsSubMenu.setEnabled(false);

            if (trial.groupNamesPresent())
                ((JMenuItem) windowsMenu.getItem(2)).setEnabled(true);
            else
                ((JMenuItem) windowsMenu.getItem(2)).setEnabled(false);

            if (trial.userEventsPresent())
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
        }
        if (tmpString.equals("colorEvent")) {
            //Just need to call a repaint on the ThreadDataWindowPanel.
            panel.repaint();
        } else if (tmpString.equals("dataEvent")) {
            if (!(trial.isTimeMetric()))
                units = 0;
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
        ParaProf.helpWindow.writeText("This is the statistics window");
        ParaProf.helpWindow.writeText("");
        ParaProf.helpWindow.writeText("This window shows you textual statistics.");
        ParaProf.helpWindow.writeText("");
        ParaProf.helpWindow.writeText("Use the options menu to select different ways of displaying the data.");
        ParaProf.helpWindow.writeText("");
        ParaProf.helpWindow.writeText("Right click on any line within this window to bring up a popup");
        ParaProf.helpWindow.writeText("menu. In this menu you can change or reset the default color");
        ParaProf.helpWindow.writeText(", or to show more details about the Function / User Event.");
        ParaProf.helpWindow.writeText("You can also left click any line to highlight it in the system.");
    }

    //Helper functionProfiles.
    private void addCompItem(Component c, GridBagConstraints gbc, int x, int y, int w, int h) {
        gbc.gridx = x;
        gbc.gridy = y;
        gbc.gridwidth = w;
        gbc.gridheight = h;
        getContentPane().add(c, gbc);
    }

    public DataSorter getDataSorter() {
        return dataSorter;
    }

    //Updates this window's data copy.
    private void sortLocalData() {
        if (name) { // if sort by name
            if (userEventWindow) {
                list = dataSorter.getUserEventProfiles(nodeID, contextID, threadID, order);
            } else {
                list = dataSorter.getFunctionProfiles(nodeID, contextID, threadID, order);
            }
        } else {

            if (userEventWindow) {
                list = dataSorter.getUserEventProfiles(nodeID, contextID, threadID, valueType + order);
            } else {
                list = dataSorter.getFunctionProfiles(nodeID, contextID, threadID, valueType + order);
            }
        }
    }

    public Vector getData() {
        return list;
    }

    public int getValueType() {
        return valueType;
    }

    public int units() {
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
            Preferences p = trial.getPreferences();
            jTextArea.setFont(new Font(p.getParaProfFont(), p.getFontStyle(), p.getFontSize()));
            jTextArea.append(this.getHeaderString());
            sp.setColumnHeaderView(jTextArea);
        } else
            sp.setColumnHeaderView(null);
    }

    public String getHeaderString() {
        if (userEventWindow)
            return "Sorted By: " + UtilFncs.getValueTypeString(valueType) + "\n";
        else
            return "Metric Name: " + (trial.getMetricName(trial.getSelectedMetricID())) + "\n" + "Sorted By: "
                    + UtilFncs.getValueTypeString(valueType) + "\n" + "Units: "
                    + UtilFncs.getUnitsString(units, trial.isTimeMetric(), trial.isDerivedMetric()) + "\n";
    }

    //######
    //End - Panel header.
    //######

    //Respond correctly when this window is closed.
    void thisWindowClosing(java.awt.event.WindowEvent e) {
        closeThisWindow();
    }

    void closeThisWindow() {
        try {
            setVisible(false);
            trial.getSystemEvents().deleteObserver(this);
            ParaProf.decrementNumWindows();
        } catch (Exception e) {
            // do nothing
        }
        dispose();
    }

    //Instance data.
    private ParaProfTrial trial = null;
    private DataSorter dataSorter;
    private int nodeID = -1;
    private int contextID = -1;
    private int threadID = -1;
    private boolean userEventWindow;

    private JMenu optionsMenu = null;
    private JMenu windowsMenu = null;
    private JMenu unitsSubMenu = null;

    private JCheckBoxMenuItem sortByName = null;
    private JCheckBoxMenuItem descendingOrder = null;
    private JCheckBoxMenuItem showPathTitleInReverse = null;
    private JCheckBoxMenuItem showMetaData = null;

    private JScrollPane sp = null;
    private StatWindowPanel panel = null;

    Vector list = null;

    private boolean name = false; //true: sort by name,false: sort by value.
    private int order = 0; //0: descending order,1: ascending order.
    private int valueType = 2; //2-exclusive,4-inclusive,6-number of
    // calls,8-number of subroutines,10-per call
    // value.
    private int units = 0; //0-microseconds,1-milliseconds,2-seconds.
}