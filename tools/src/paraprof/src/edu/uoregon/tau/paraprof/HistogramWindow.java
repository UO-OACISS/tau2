package edu.uoregon.tau.paraprof;

import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.Observable;
import java.util.Observer;
import java.util.Vector;

import javax.swing.*;
import javax.swing.event.*;

import edu.uoregon.tau.dms.dss.Function;
import edu.uoregon.tau.dms.dss.UtilFncs;
import edu.uoregon.tau.paraprof.enums.ValueType;
import edu.uoregon.tau.paraprof.interfaces.ParaProfWindow;
import edu.uoregon.tau.paraprof.interfaces.UnitListener;


/**
 * HistogramWindow
 * This is the histogram window
 *  
 * <P>CVS $Id: HistogramWindow.java,v 1.12 2005/05/10 01:48:37 amorris Exp $</P>
 * @author	Robert Bell, Alan Morris
 * @version	$Revision: 1.12 $
 * @see		HistogramWindowPanel
 */
public class HistogramWindow extends JFrame implements ActionListener, MenuListener, Observer, ChangeListener, ParaProfWindow, UnitListener {

    // instance data
    private ParaProfTrial ppTrial = null;
    private DataSorter dataSorter = null;
    private Function function = null;

    // hold on to these two for 'menuSelected'
    private JMenu unitsSubMenu = null;


    private JScrollPane sp = null;
    private HistogramWindowPanel panel = null;

    private Vector data = null;

    private int units = ParaProf.preferences.getUnits();

    private JCheckBoxMenuItem slidersCheckBox = null;
    private JLabel numBinsLabel = new JLabel("Number of Bins");
    private JSlider numBinsSlider = new JSlider(0, 100, 10);
    private int numBins = 10;

    

    public HistogramWindow(ParaProfTrial ppTrial, Function function) {
        this.ppTrial = ppTrial;
        this.dataSorter = new DataSorter(ppTrial);
        this.function = function;

        setTitle("Histogram: " + ppTrial.getTrialIdentifier(ParaProf.preferences.getShowPathTitleInReverse()));
        setLocation(new java.awt.Point(300, 200));
        setSize(new java.awt.Dimension(670, 630));

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

        // panel and scrollpane definition
        panel = new HistogramWindowPanel(ppTrial, this);
        sp = new JScrollPane(panel);
        this.setHeader();

        gbc.fill = GridBagConstraints.BOTH;
        gbc.anchor = GridBagConstraints.CENTER;
        gbc.weightx = 0.95;
        gbc.weighty = 0.98;
        addCompItem(sp, gbc, 0, 0, 1, 1);

        setupMenus();

        ParaProf.incrementNumWindows();
    }


    private void setupMenus() {
        JMenuBar mainMenu = new JMenuBar();
        JMenu subMenu = null;
        JMenuItem menuItem = null;

        

        // options menu
        JMenu optionsMenu = new JMenu("Options");

        slidersCheckBox = new JCheckBoxMenuItem("Show Number of Bins Slider", false);
        slidersCheckBox.addActionListener(this);
        optionsMenu.add(slidersCheckBox);

        JCheckBoxMenuItem box = null;
        ButtonGroup group = null;
        JRadioButtonMenuItem button = null;

        // units submenu
        unitsSubMenu = ParaProfUtils.createUnitsMenu(this, units);
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
                if (arg.equals("Exclusive")) {
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
                } else if (arg.equals("Show Number of Bins Slider")) {
                    if (slidersCheckBox.isSelected())
                        displaySliders(true);
                    else
                        displaySliders(false);
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
        } catch (Exception e) {
            ParaProfUtils.handleException(e);
        }
    }

    public void menuDeselected(MenuEvent evt) {
    }

    public void menuCanceled(MenuEvent evt) {
    }

    // listener for the numBinsSlider
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
            panel.repaint();
        } else if (tmpString.equals("colorEvent")) {
            panel.repaint();
        } else if (tmpString.equals("dataEvent")) {
            dataSorter.setSelectedMetricID(ppTrial.getDefaultMetricID());
            sortLocalData();
            this.setHeader();
            panel.repaint();
        } else if (tmpString.equals("subWindowCloseEvent")) {
            closeThisWindow();
        }
    }

    public void help(boolean display) {
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

    private void sortLocalData() {
        data = dataSorter.getFunctionData(function, false);
    }

    // This process is separated into two functions to provide the option of obtaining the current 
    // header string being used for the panel without resetting the actual header. 
    // Printing and image generation use this functionality for example.
    public void setHeader() {
        JTextArea jTextArea = new JTextArea();
        jTextArea.setLineWrap(true);
        jTextArea.setWrapStyleWord(true);
        jTextArea.setEditable(false);
        PreferencesWindow p = ppTrial.getPreferencesWindow();
        jTextArea.setFont(new Font(p.getParaProfFont(), p.getFontStyle(), p.getFontSize()));
        jTextArea.append(this.getHeaderString());
        sp.setColumnHeaderView(jTextArea);
    }

    public String getHeaderString() {
        if (dataSorter.getValueType() == ValueType.NUMCALLS || dataSorter.getValueType() == ValueType.NUMSUBR)
            return "Metric Name: " + (ppTrial.getMetricName(ppTrial.getDefaultMetricID())) + "\n" + "Name: "
                    + ParaProfUtils.getFunctionName(function) + "\n" + "Value Type: " + dataSorter.getValueType()
                    + "\n";
        else
            return "Metric Name: " + (ppTrial.getMetricName(ppTrial.getDefaultMetricID())) + "\n" + "Name: "
                    + ParaProfUtils.getFunctionName(function) + "\n" + "Value Type: " + dataSorter.getValueType()
                    + "\n" + "Units: "
                    + UtilFncs.getUnitsString(units, ppTrial.isTimeMetric(), ppTrial.isDerivedMetric()) + "\n";
    }

    public Vector getData() {
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
            ppTrial.getSystemEvents().deleteObserver(this);
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
        this.numBins =  numBins;
        panel.repaint();
    }

    public int getNumBins() {
        return numBins;
    }


    public void setUnits(int units) {
        this.units = units;
        this.setHeader();
        panel.repaint();
    }

}