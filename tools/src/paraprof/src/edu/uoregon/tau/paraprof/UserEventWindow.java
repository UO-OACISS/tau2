/*
 * UserEventWindow.java
 * 
 * Title: ParaProf 
 * Author: Robert Bell 
 * Description: The container for the UserEventWindowPanel.
 */

package edu.uoregon.tau.paraprof;

import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.*;
import java.util.List;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import javax.swing.event.MenuEvent;

import edu.uoregon.tau.dms.dss.UserEvent;
import edu.uoregon.tau.paraprof.enums.UserEventValueType;
import edu.uoregon.tau.paraprof.interfaces.ParaProfWindow;

public class UserEventWindow extends JFrame implements ActionListener, Observer, ChangeListener, ParaProfWindow {

    
    //Instance data.
    private ParaProfTrial trial = null;
    private DataSorter dataSorter = null;

    UserEvent userEvent = null;

    private JMenu optionsMenu = null;

    private JCheckBoxMenuItem descendingOrder = null;
    private JCheckBoxMenuItem showPathTitleInReverse = null;
    private JCheckBoxMenuItem showMetaData = null;

    private JLabel barLengthLabel = new JLabel("Bar Width");
    private JSlider barLengthSlider = new JSlider(0, 2000, 250);

    private Container contentPane = null;
    private GridBagLayout gbl = null;
    private GridBagConstraints gbc = null;

    UserEventWindowPanel panel = null;
    JScrollPane sp = null;
    JLabel label = null;

    private List list = new ArrayList();

    UserEventValueType userEventValueType = UserEventValueType.NUMSAMPLES;
    

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
        this.setTitle("User Event Window: " + trial.getTrialIdentifier(ParaProf.preferences.getShowPathTitleInReverse()));

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
        panel = new UserEventWindowPanel(trial, userEvent, this);
        
        //####################################
        //Code to generate the menus.
        //####################################
        JMenuBar mainMenu = new JMenuBar();
        JMenu subMenu = null;
        JMenuItem menuItem = null;

       


        //Options menu.
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

        button = new JRadioButtonMenuItem("Number of Samples", true);
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


        showMetaData = new JCheckBoxMenuItem("Show Meta Data in Panel", true);
        showMetaData.addActionListener(this);
        optionsMenu.add(showMetaData);


       
        //Now, add all the menus to the main menu.
        mainMenu.add(ParaProfUtils.createFileMenu(this, panel, panel));
        mainMenu.add(optionsMenu);
        //mainMenu.add(ParaProfUtils.createTrialMenu(trial, this));
        mainMenu.add(ParaProfUtils.createWindowsMenu(trial, this));
        mainMenu.add(ParaProfUtils.createHelpMenu(this, this));

        setJMenuBar(mainMenu);

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
        addCompItem(sp, gbc, 0, 1, 2, 1);

        ParaProf.incrementNumWindows();
    }

    public void actionPerformed(ActionEvent evt) {
        try {
            Object EventSrc = evt.getSource();

            if (EventSrc instanceof JMenuItem) {
                String arg = evt.getActionCommand();

                if (arg.equals("Number of Samples")) {
                    userEventValueType = UserEventValueType.NUMSAMPLES;
                    sortLocalData();
                    this.setHeader();
                    panel.repaint();
                } else if (arg.equals("Min. Value")) {
                    userEventValueType = UserEventValueType.MIN;
                    sortLocalData();
                    this.setHeader();
                    panel.repaint();
                } else if (arg.equals("Max. Value")) {
                    userEventValueType = UserEventValueType.MAX;
                    sortLocalData();
                    this.setHeader();
                    panel.repaint();
                } else if (arg.equals("Mean Value")) {
                    userEventValueType = UserEventValueType.MEAN;
                    sortLocalData();
                    this.setHeader();
                    panel.repaint();
                } else if (arg.equals("Standard Deviation")) {
                    userEventValueType = UserEventValueType.STDDEV;
                    sortLocalData();
                    this.setHeader();
                    panel.repaint();
                } else if (arg.equals("Descending Order")) {
                    dataSorter.setDescendingOrder(((JCheckBoxMenuItem) optionsMenu.getItem(0)).isSelected());
                    sortLocalData();
                    panel.repaint();
                } else if (arg.equals("Show Width Slider")) {
                    if (((JCheckBoxMenuItem) optionsMenu.getItem(2)).isSelected())
                        displaySiders(true);
                    else
                        displaySiders(false);
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

    public void help(boolean display) {
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

        dataSorter.setUserEventValueType(userEventValueType);
        list = dataSorter.getUserEventData(userEvent);
    }

    public List getData() {
        return list;
    }

    public UserEventValueType getValueType() {
        return userEventValueType;
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
        return "Name: " + userEvent.getName() + "\n" + "Value Type: " + userEventValueType.toString()
                + "\n";
    }


    private void displaySiders(boolean displaySliders) {
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
            trial.getSystemEvents().deleteObserver(this);
            ParaProf.decrementNumWindows();
        } catch (Exception e) {
            // do nothing
        }
        dispose();
    }

 

}