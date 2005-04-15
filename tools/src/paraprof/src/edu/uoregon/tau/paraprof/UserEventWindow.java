/*
 * UserEventWindow.java
 * 
 * Title: ParaProf 
 * Author: Robert Bell 
 * Description: The container for the UserEventWindowPanel.
 */

package edu.uoregon.tau.paraprof;

import java.util.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.event.*;
import java.awt.print.*;
import edu.uoregon.tau.dms.dss.*;

public class UserEventWindow extends JFrame implements ActionListener, MenuListener, Observer, ChangeListener {

    public UserEventWindow(ParaProfTrial trial, UserEvent userEvent, DataSorter dataSorter) {
        this.userEvent = userEvent;
        this.trial = trial;
        this.dataSorter = dataSorter;

        int windowWidth = 650;
        int windowHeight = 550;
        //Grab the screen size.
        Toolkit tk = Toolkit.getDefaultToolkit();
        Dimension screenDimension = tk.getScreenSize();
        int screenHeight = screenDimension.height;
        int screenWidth = screenDimension.width;
        if (windowWidth > screenWidth)
            windowWidth = screenWidth;
        if (windowHeight > screenHeight)
            windowHeight = screenHeight;
        //Set the window to come up in the center of the screen.
        int xPosition = (screenWidth - windowWidth) / 2;
        int yPosition = (screenHeight - windowHeight) / 2;
        setSize(new java.awt.Dimension(windowWidth, windowHeight));
        setLocation(xPosition, yPosition);

        //Now set the title.
        this.setTitle("User Event Window: " + trial.getTrialIdentifier(true));

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

        ButtonGroup group = null;
        JCheckBoxMenuItem box = null;
        JRadioButtonMenuItem button = null;

        descendingOrder = new JCheckBoxMenuItem("Descending Order", true);
        descendingOrder.addActionListener(this);
        optionsMenu.add(descendingOrder);

        //Set the value type options.
        subMenu = new JMenu("Select Value Type");
        group = new ButtonGroup();

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

        button = new JRadioButtonMenuItem("Standard Deviation", false);
        button.addActionListener(this);
        group.add(button);
        subMenu.add(button);

        optionsMenu.add(subMenu);

        box = new JCheckBoxMenuItem("Show Width Slider", false);
        box.addActionListener(this);
        optionsMenu.add(box);

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
        contentPane = getContentPane();
        gbl = new GridBagLayout();
        contentPane.setLayout(gbl);
        gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);

        //######
        //Panel and ScrollPane definition.
        //######
        panel = new UserEventWindowPanel(trial, userEvent, this);
        //The scroll panes into which the list shall be placed.
        sp = new JScrollPane(panel);
        this.setHeader();
        //######
        //Panel and ScrollPane definition.
        //######

        //######
        //Slider setup.
        //Do the slider stuff, but don't add. By default, sliders are off.
        //######
        String sliderMultipleStrings[] = { "1.00", "0.75", "0.50", "0.25", "0.10" };
        sliderMultiple = new JComboBox(sliderMultipleStrings);
        sliderMultiple.addActionListener(this);

        barLengthSlider.setPaintTicks(true);
        barLengthSlider.setMajorTickSpacing(5);
        barLengthSlider.setMinorTickSpacing(1);
        barLengthSlider.setPaintLabels(true);
        barLengthSlider.setSnapToTicks(true);
        barLengthSlider.addChangeListener(this);
        //######
        //End - Slider setup.
        //Do the slider stuff, but don't add. By default, sliders are off.
        //######
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
                    trial.getPreferencesWindow().showPreferencesWindow();
                } else if (arg.equals("Save Image")) {
                    ParaProfImageOutput.saveImage(panel);
                }
                if (arg.equals("Close This Window")) {
                    closeThisWindow();
                } else if (arg.equals("Exit ParaProf!")) {
                    setVisible(false);
                    dispose();
                    ParaProf.exitParaProf(0);
                } else if (arg.equals("Number of Userevents")) {
                    valueType = 12;
                    this.setHeader();
                    panel.repaint();
                } else if (arg.equals("Min. Value")) {
                    valueType = 14;
                    this.setHeader();
                    panel.repaint();
                } else if (arg.equals("Max. Value")) {
                    valueType = 16;
                    this.setHeader();
                    panel.repaint();
                } else if (arg.equals("Mean Value")) {
                    valueType = 18;
                    this.setHeader();
                    panel.repaint();
                } else if (arg.equals("Standard Deviation")) {
                    valueType = 20;
                    this.setHeader();
                    panel.repaint();
                } else if (arg.equals("Descending Order")) {
                    if (((JCheckBoxMenuItem) optionsMenu.getItem(0)).isSelected())
                        order = 0;
                    else
                        order = 1;
                    sortLocalData();
                    panel.repaint();
                } else if (arg.equals("Show Width Slider")) {
                    if (((JCheckBoxMenuItem) optionsMenu.getItem(2)).isSelected())
                        displaySiders(true);
                    else
                        displaySiders(false);
                } else if (arg.equals("Show Path Title in Reverse")) {
                    this.setTitle("User Event Window: "
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
                            1);
                    trial.getSystemEvents().addObserver(tmpRef);
                    tmpRef.show();
                } else if (arg.equals("Close All Sub-Windows")) {
                    trial.getSystemEvents().updateRegisteredObjects("subWindowCloseEvent");
                } else if (arg.equals("About ParaProf")) {
                    JOptionPane.showMessageDialog(this, ParaProf.getInfoString());
                } else if (arg.equals("Show Help Window")) {
                    this.help(true);
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
            panel.repaint();
        } else if (tmpString.equals("colorEvent")) {
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
        ParaProf.helpWindow.writeText("This is the userevent data window for:");
        ParaProf.helpWindow.writeText(userEvent.getName());
        ParaProf.helpWindow.writeText("");
        ParaProf.helpWindow.writeText("This window shows you this userevent's statistics across all the threads.");
        ParaProf.helpWindow.writeText("");
        ParaProf.helpWindow.writeText("Use the options menu to select different ways of displaying the data.");
        ParaProf.helpWindow.writeText("");
        ParaProf.helpWindow.writeText("Right click anywhere within this window to bring up a popup");
        ParaProf.helpWindow.writeText("menu. In this menu you can change or reset the default color");
        ParaProf.helpWindow.writeText("for this userevent.");
    }

    public DataSorter getDataSorter() {
        return dataSorter;
    }

    public void sortLocalData() {
        list = dataSorter.getUserEventData(userEvent);
    }

    public Vector getData() {
        return list;
    }

    public int getValueType() {
        return valueType;
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
    //This process is separated into two functions to provide the option
    //of obtaining the current header string being used for the panel
    //without resetting the actual header. Printing and image generation
    //use this functionality for example.
    public void setHeader() {
        if (showMetaData.isSelected()) {
            JTextArea jTextArea = new JTextArea();
            jTextArea.setLineWrap(true);
            jTextArea.setWrapStyleWord(true);
            jTextArea.setEditable(false);
            jTextArea.append(this.getHeaderString());
            sp.setColumnHeaderView(jTextArea);
        } else
            sp.setColumnHeaderView(null);
    }

    public String getHeaderString() {
        return "Name: " + userEvent.getName() + "\n" + "Value Type: " + UtilFncs.getValueTypeString(valueType)
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

    private void displaySiders(boolean displaySliders) {
        if (displaySliders) {
            contentPane.remove(sp);

            gbc.fill = GridBagConstraints.NONE;
            gbc.anchor = GridBagConstraints.EAST;
            gbc.weightx = 0.10;
            gbc.weighty = 0.01;
            addCompItem(sliderMultipleLabel, gbc, 0, 0, 1, 1);

            gbc.fill = GridBagConstraints.NONE;
            gbc.anchor = GridBagConstraints.WEST;
            gbc.weightx = 0.10;
            gbc.weighty = 0.01;
            addCompItem(sliderMultiple, gbc, 1, 0, 1, 1);

            gbc.fill = GridBagConstraints.NONE;
            gbc.anchor = GridBagConstraints.EAST;
            gbc.weightx = 0.10;
            gbc.weighty = 0.01;
            addCompItem(barLengthLabel, gbc, 2, 0, 1, 1);

            gbc.fill = GridBagConstraints.HORIZONTAL;
            gbc.anchor = GridBagConstraints.WEST;
            gbc.weightx = 0.70;
            gbc.weighty = 0.01;
            addCompItem(barLengthSlider, gbc, 3, 0, 1, 1);

            gbc.fill = GridBagConstraints.BOTH;
            gbc.anchor = GridBagConstraints.CENTER;
            gbc.weightx = 0.95;
            gbc.weighty = 0.98;
            addCompItem(sp, gbc, 0, 1, 4, 1);
        } else {
            contentPane.remove(sliderMultipleLabel);
            contentPane.remove(sliderMultiple);
            contentPane.remove(barLengthLabel);
            contentPane.remove(barLengthSlider);
            contentPane.remove(sp);

            gbc.fill = GridBagConstraints.BOTH;
            gbc.anchor = GridBagConstraints.CENTER;
            gbc.weightx = 0.95;
            gbc.weighty = 0.98;
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
    private DataSorter dataSorter = null;

    UserEvent userEvent = null;

    private JMenu optionsMenu = null;
    private JMenu windowsMenu = null;

    private JCheckBoxMenuItem descendingOrder = null;
    private JCheckBoxMenuItem showPathTitleInReverse = null;
    private JCheckBoxMenuItem showMetaData = null;

    private JLabel sliderMultipleLabel = new JLabel("Slider Multiple");
    private JComboBox sliderMultiple;
    private JLabel barLengthLabel = new JLabel("Bar Multiple");
    private JSlider barLengthSlider = new JSlider(0, 40, 1);

    private Container contentPane = null;
    private GridBagLayout gbl = null;
    private GridBagConstraints gbc = null;

    UserEventWindowPanel panel = null;
    JScrollPane sp = null;
    JLabel label = null;

    private Vector list = null;

    private int order = 0; //0: descending order,1: ascending order.
    private int valueType = 12; //12-number of
    // userEvents,14-min,16-max,18-mean.

}