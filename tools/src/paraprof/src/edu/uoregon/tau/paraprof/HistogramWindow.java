package edu.uoregon.tau.paraprof;

import java.util.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.event.*;
import java.awt.print.*;
import edu.uoregon.tau.dms.dss.*;

/**
 * HistogramWindow
 * This is the histogram window
 *  
 * <P>CVS $Id: HistogramWindow.java,v 1.6 2005/01/04 01:16:26 amorris Exp $</P>
 * @author	Robert Bell, Alan Morris
 * @version	$Revision: 1.6 $
 * @see		HistogramWindowPanel
 */
public class HistogramWindow extends JFrame implements ActionListener, MenuListener, Observer, ChangeListener {

    private void setupMenus() {
        JMenuBar mainMenu = new JMenuBar();
        JMenu subMenu = null;
        JMenuItem menuItem = null;

        //File menu.
        JMenu fileMenu = new JMenu("File");

        //Save menu.
        subMenu = new JMenu("Save ...");

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

        // options menu
        optionsMenu = new JMenu("Options");

        slidersCheckBox = new JCheckBoxMenuItem("Show Number of Bins Slider", false);
        slidersCheckBox.addActionListener(this);
        optionsMenu.add(slidersCheckBox);

        JCheckBoxMenuItem box = null;
        ButtonGroup group = null;
        JRadioButtonMenuItem button = null;

        // units submenu
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

        button = new JRadioButtonMenuItem("Number of Subroutines", false);
        button.addActionListener(this);
        group.add(button);
        subMenu.add(button);

        button = new JRadioButtonMenuItem("Inclusive Per Call", false);
        button.addActionListener(this);
        group.add(button);
        subMenu.add(button);

        optionsMenu.add(subMenu);

        optionsMenu.addMenuListener(this);

        //Windows menu
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

    public HistogramWindow(ParaProfTrial ppTrial, DataSorter dataSorter, Function function) {
        this.ppTrial = ppTrial;
        this.dataSorter = dataSorter;
        this.function = function;

        setLocation(new java.awt.Point(300, 200));
        setSize(new java.awt.Dimension(670, 630));
        //Now set the title.
        setTitle("Histogram: " + ppTrial.getTrialIdentifier(true));

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

        setupMenus();

        numBinsSlider.setPaintTicks(true);
        numBinsSlider.setMajorTickSpacing(50);
        numBinsSlider.setMinorTickSpacing(10);
        numBinsSlider.setPaintLabels(true);
        numBinsSlider.setSnapToTicks(false);
        numBinsSlider.addChangeListener(this);

        //Setting up the layout system for the main window.
        contentPane = getContentPane();
        gbl = new GridBagLayout();
        contentPane.setLayout(gbl);
        gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);

        //Panel and ScrollPane definition.
        panel = new HistogramWindowPanel(ppTrial, this, function);
        sp = new JScrollPane(panel);
        this.setHeader();

        gbc.fill = GridBagConstraints.BOTH;
        gbc.anchor = GridBagConstraints.CENTER;
        gbc.weightx = 0.95;
        gbc.weighty = 0.98;
        addCompItem(sp, gbc, 0, 0, 1, 1);

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
                    ppTrial.getPreferences().showPreferencesWindow();
                } else if (arg.equals("Save Image")) {
                    ParaProfImageOutput imageOutput = new ParaProfImageOutput();
                    imageOutput.saveImage((ParaProfImageInterface) panel);
                } else if (arg.equals("Close This Window")) {
                    closeThisWindow();
                } else if (arg.equals("Exit ParaProf!")) {
                    setVisible(false);
                    dispose();
                    ParaProf.exitParaProf(0);

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
                } else if (arg.equals("Inclusive Per Call")) {
                    valueType = 10;
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
                } else if (arg.equals("Show Number of Bins Slider")) {
                    if (slidersCheckBox.isSelected())
                        displaySliders(true);
                    else
                        displaySliders(false);
                } else if (arg.equals("Show ParaProf Manager")) {
                    (new ParaProfManagerWindow()).show();
                } else if (arg.equals("Show Function Ledger")) {
                    (new LedgerWindow(ppTrial, 0)).show();
                } else if (arg.equals("Show Group Ledger")) {
                    (new LedgerWindow(ppTrial, 1)).show();
                } else if (arg.equals("Show User Event Ledger")) {
                    (new LedgerWindow(ppTrial, 2)).show();
                } else if (arg.equals("Show Call Path Relations")) {
                    CallPathTextWindow tmpRef = new CallPathTextWindow(ppTrial, -1, -1, -1, this.dataSorter, 2);
                    ppTrial.getSystemEvents().addObserver(tmpRef);
                    tmpRef.show();
                } else if (arg.equals("Close All Sub-Windows")) {
                    ppTrial.getSystemEvents().updateRegisteredObjects("subWindowCloseEvent");
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

    private void displaySliders(boolean displaySliders) {
        Container contentPane = this.getContentPane();
        GridBagConstraints gbc = new GridBagConstraints();
        if (displaySliders) {
            contentPane.remove(sp);

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
            addCompItem(sp, gbc, 0, 2, 2, 1);
        } else {
            contentPane.remove(numBinsLabel);
            contentPane.remove(numBinsSlider);
            contentPane.remove(sp);

            gbc.fill = GridBagConstraints.BOTH;
            gbc.anchor = GridBagConstraints.CENTER;
            gbc.weightx = 1;
            gbc.weighty = 1;
            addCompItem(sp, gbc, 0, 1, 1, 1);
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

    // listener for the boxWidthSlider
    public void stateChanged(ChangeEvent event) {
        try {
            numBins = numBinsSlider.getValue();
            if (numBins < 1)
                numBins = 1;
            panel.repaint();
        } catch (Exception e) {
            ParaProfUtils.handleException(e);
        }

    }

    public void update(Observable o, Object arg) {
        String tmpString = (String) arg;
        if (tmpString.equals("prefEvent")) {
            this.setHeader();
            //Just need to call a repaint on the ThreadDataWindowPanel.
            panel.repaint();
        } else if (tmpString.equals("colorEvent")) {
            //Just need to call a repaint on the ThreadDataWindowPanel.
            panel.repaint();
        } else if (tmpString.equals("dataEvent")) {
            sortLocalData();
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
        ParaProf.helpWindow.writeText("This is the histogram window");
        ParaProf.helpWindow.writeText("");
        ParaProf.helpWindow.writeText("This window shows you a histogram of all of the values for this function.");
        ParaProf.helpWindow.writeText("");
        ParaProf.helpWindow.writeText("Use the options menu to select different types of data to display.");
        ParaProf.helpWindow.writeText("");
    }

    //Updates this window's data copy.
    private void sortLocalData() {
        list = dataSorter.getFunctionData(function, 0, false);
    }

    // This process is separated into two functions to provide the option of obtaining the current 
    // header string being used for the panel without resetting the actual header. 
    // Printing and image generation use this functionality for example.
    public void setHeader() {
        JTextArea jTextArea = new JTextArea();
        jTextArea.setLineWrap(true);
        jTextArea.setWrapStyleWord(true);
        jTextArea.setEditable(false);
        Preferences p = ppTrial.getPreferences();
        jTextArea.setFont(new Font(p.getParaProfFont(), p.getFontStyle(), p.getFontSize()));
        jTextArea.append(this.getHeaderString());
        sp.setColumnHeaderView(jTextArea);
    }

    public String getHeaderString() {
        if (valueType > 5)
            return "Metric Name: " + (ppTrial.getMetricName(ppTrial.getSelectedMetricID())) + "\n" + "Name: "
                    + function.getName() + "\n" + "Value Type: " + UtilFncs.getValueTypeString(valueType)
                    + "\n";
        else
            return "Metric Name: " + (ppTrial.getMetricName(ppTrial.getSelectedMetricID())) + "\n" + "Name: "
                    + function.getName() + "\n" + "Value Type: " + UtilFncs.getValueTypeString(valueType)
                    + "\n" + "Units: "
                    + UtilFncs.getUnitsString(units, ppTrial.isTimeMetric(), ppTrial.isDerivedMetric()) + "\n";
    }

    public Vector getData() {
        return list;
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

    void closeThisWindow() {
        try {
            setVisible(false);
            ppTrial.getSystemEvents().deleteObserver(this);
            ParaProf.decrementNumWindows();
        } catch (Exception e) {
            // do nothing
        }
        dispose();
    }

    public int getValueType() {
        return valueType;
    }

    public int units() {
        if (valueType > 5)
            return 0;
        return units;
    }

    public int getNumBins() {
        return numBins;
    }

    //Instance data.
    private ParaProfTrial ppTrial = null;
    private DataSorter dataSorter = null;
    private Function function = null;

    private JMenu optionsMenu = null;
    private JMenu windowsMenu = null;
    private JMenu unitsSubMenu = null;

    private Container contentPane = null;
    private GridBagLayout gbl = null;
    private GridBagConstraints gbc = null;

    private JScrollPane sp = null;
    private HistogramWindowPanel panel = null;

    private Vector list = null;

    private int valueType = 2; //2-exclusive,4-inclusive,6-number of calls,8-number of subroutines,10-per call value.
    private int units = 0; //0-microseconds,1-milliseconds,2-seconds.

    private int numBins = 10;

    private JCheckBoxMenuItem slidersCheckBox = null;
    private JLabel numBinsLabel = new JLabel("Number of Bins");
    private JSlider numBinsSlider = new JSlider(0, 100, 10);

}