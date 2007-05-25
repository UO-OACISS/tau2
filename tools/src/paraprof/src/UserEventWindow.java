package edu.uoregon.tau.paraprof;

import java.awt.*;
import java.awt.event.*;
import java.util.*;
import java.util.List;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import javax.swing.event.MenuEvent;

import edu.uoregon.tau.paraprof.barchart.*;
import edu.uoregon.tau.paraprof.enums.SortType;
import edu.uoregon.tau.paraprof.enums.UserEventValueType;
import edu.uoregon.tau.paraprof.interfaces.ParaProfWindow;
import edu.uoregon.tau.paraprof.interfaces.SearchableOwner;
import edu.uoregon.tau.perfdmf.Thread;
import edu.uoregon.tau.perfdmf.UserEvent;

/**
 * The UserEventWindow shows one User Event over all threads.
 * 
 * <P>CVS $Id: UserEventWindow.java,v 1.28 2007/05/25 21:55:45 amorris Exp $</P>
 * @author  Alan Morris, Robert Bell
 * @version $Revision: 1.28 $
 * @see GlobalBarChartModel
 */
public class UserEventWindow extends JFrame implements ActionListener, KeyListener, Observer, ChangeListener, ParaProfWindow, SearchableOwner {

    private ParaProfTrial ppTrial;
    private DataSorter dataSorter;
    private UserEvent userEvent;
    private Thread thread;

    private JMenu optionsMenu;

    private JCheckBoxMenuItem descendingOrder;
    private JCheckBoxMenuItem sortByNCT;
    private JCheckBoxMenuItem sortByName;
    private JCheckBoxMenuItem showMetaData;
    private JCheckBoxMenuItem displayWidthSlider;
    private JCheckBoxMenuItem showFindPanelBox;

    private JLabel barLengthLabel = new JLabel("Bar Width");
    private JSlider barLengthSlider = new JSlider(0, 2000, 400);

    private GridBagLayout gbl;
    private GridBagConstraints gbc;

    private BarChartPanel panel;
    private BarChartModel model;

    private SearchPanel searchPanel;

    private List list = new ArrayList();

    private UserEventValueType userEventValueType = UserEventValueType.NUMSAMPLES;

    public UserEventWindow(ParaProfTrial ppTrial, Thread thread, Component invoker) {
        this.thread = thread;

        this.ppTrial = ppTrial;
        ppTrial.addObserver(this);

        this.dataSorter = new DataSorter(ppTrial);
        
        int windowWidth = 750;
        int windowHeight = 650;

        setSize(ParaProfUtils.checkSize(new java.awt.Dimension(windowWidth, windowHeight)));
        setLocation(WindowPlacer.getNewLocation(this, invoker));

        //Now set the title.
        this.setTitle("TAU: ParaProf: User Event Window: " + ppTrial.getTrialIdentifier(ParaProf.preferences.getShowPathTitleInReverse()));
        ParaProfUtils.setFrameIcon(this);


        //Add some window listener code
        addWindowListener(new java.awt.event.WindowAdapter() {
            public void windowClosing(java.awt.event.WindowEvent evt) {
                thisWindowClosing(evt);
            }
        });

        //Set the help window text if required.
        if (ParaProf.getHelpWindow().isVisible()) {
            this.help(false);
        }

        model = new UserEventThreadBarChartModel(this, dataSorter, thread);


        panel = new BarChartPanel(model, null);

        panel.getBarChart().setBarLength(barLengthSlider.getValue());

        setupMenus();

        //Setting up the layout system for the main window.
        gbl = new GridBagLayout();
        getContentPane().setLayout(gbl);
        gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);

        this.setHeader();

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
        addCompItem(panel, gbc, 0, 1, 2, 1);
        sortLocalData();

        ParaProf.incrementNumWindows();

    }

    public UserEventWindow(ParaProfTrial ppTrial, UserEvent userEvent, Component invoker) {
        this.userEvent = userEvent;
        this.ppTrial = ppTrial;
        ppTrial.addObserver(this);

        this.dataSorter = new DataSorter(ppTrial);

        int windowWidth = 650;
        int windowHeight = 550;

        setSize(ParaProfUtils.checkSize(new java.awt.Dimension(windowWidth, windowHeight)));
        setLocation(WindowPlacer.getNewLocation(this, invoker));

        //Now set the title.
        this.setTitle("User Event Window: " + ppTrial.getTrialIdentifier(ParaProf.preferences.getShowPathTitleInReverse()));

        //Add some window listener code
        addWindowListener(new java.awt.event.WindowAdapter() {
            public void windowClosing(java.awt.event.WindowEvent evt) {
                thisWindowClosing(evt);
            }
        });

        //Set the help window text if required.
        if (ParaProf.getHelpWindow().isVisible()) {
            this.help(false);
        }

        model = new UserEventBarChartModel(this, dataSorter, userEvent);

        this.addKeyListener(this);


        panel = new BarChartPanel(model, null);

        panel.getBarChart().setBarLength(barLengthSlider.getValue());
        setupMenus();


        //Setting up the layout system for the main window.
        gbl = new GridBagLayout();
        getContentPane().setLayout(gbl);
        gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);

        this.setHeader();

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
        addCompItem(panel, gbc, 0, 1, 2, 1);

        sortLocalData();
        ParaProf.incrementNumWindows();
    }

    private void setupMenus() {
        JMenuBar mainMenu = new JMenuBar();
        mainMenu.addKeyListener(this);

        JMenu subMenu = null;

        optionsMenu = new JMenu("Options");

        ButtonGroup group = null;
        JRadioButtonMenuItem button = null;

        displayWidthSlider = new JCheckBoxMenuItem("Show Width Slider", false);
        displayWidthSlider.addActionListener(this);
        //optionsMenu.add(displayWidthSlider);

        showFindPanelBox = new JCheckBoxMenuItem("Show Find Panel", false);
        showFindPanelBox.addActionListener(this);
        optionsMenu.add(showFindPanelBox);

        showMetaData = new JCheckBoxMenuItem("Show Meta Data in Panel", true);
        showMetaData.addActionListener(this);
        optionsMenu.add(showMetaData);

        optionsMenu.add(new JSeparator());

        descendingOrder = new JCheckBoxMenuItem("Descending Order", true);
        descendingOrder.addActionListener(this);
        optionsMenu.add(descendingOrder);

        if (thread == null) {
            sortByNCT = new JCheckBoxMenuItem("Sort By N,C,T", false);
            sortByNCT.addActionListener(this);
            optionsMenu.add(sortByNCT);
        } else {
            sortByName = new JCheckBoxMenuItem("Sort By Name", false);
            sortByName.addActionListener(this);
            optionsMenu.add(sortByName);
        }
        
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

        //Now, add all the menus to the main menu.
        mainMenu.add(ParaProfUtils.createFileMenu(this, panel, panel));
        mainMenu.add(optionsMenu);
        //mainMenu.add(ParaProfUtils.createTrialMenu(trial, this));
        mainMenu.add(ParaProfUtils.createWindowsMenu(ppTrial, this));
        mainMenu.add(ParaProfUtils.createHelpMenu(this, this));

        setJMenuBar(mainMenu);
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
                    sortLocalData();
                    panel.repaint();
                } else if (arg.equals("Sort By N,C,T")) {
                    sortLocalData();
                    panel.repaint();
                } else if (arg.equals("Sort By Name")) {
                    sortLocalData();
                    panel.repaint();
                } else if (arg.equals("Show Width Slider")) {
                    if (displayWidthSlider.isSelected()) {
                        displaySiders(true);
                    } else {
                        displaySiders(false);
                    }
                } else if (arg.equals("Show Find Panel")) {
                    if (showFindPanelBox.isSelected()) {
                        showSearchPanel(true);
                    } else {
                        showSearchPanel(false);
                    }
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
            panel.getBarChart().setBarLength(barLengthSlider.getValue());
            panel.repaint();
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
        } else if (tmpString.equals("dataEvent")) {
            sortLocalData();
        } else if (tmpString.equals("colorEvent")) {
            panel.repaint();
        } else if (tmpString.equals("subWindowCloseEvent")) {
            closeThisWindow();
        }
    }

    public void help(boolean display) {
        //Show the ParaProf help window.
        ParaProf.getHelpWindow().clearText();
        if (display)
            ParaProf.getHelpWindow().show();
        ParaProf.getHelpWindow().writeText("This is the userevent data window for:");
        ParaProf.getHelpWindow().writeText(userEvent.getName());
        ParaProf.getHelpWindow().writeText("");
        ParaProf.getHelpWindow().writeText("This window shows you this userevent's statistics across all the threads.");
        ParaProf.getHelpWindow().writeText("");
        ParaProf.getHelpWindow().writeText("Use the options menu to select different ways of displaying the data.");
        ParaProf.getHelpWindow().writeText("");
        ParaProf.getHelpWindow().writeText("Right click anywhere within this window to bring up a popup");
        ParaProf.getHelpWindow().writeText("menu. In this menu you can change or reset the default color");
        ParaProf.getHelpWindow().writeText("for this userevent.");
    }

    public DataSorter getDataSorter() {
        return dataSorter;
    }

    public void sortLocalData() {
        dataSorter.setDescendingOrder(descendingOrder.isSelected());
        
        if (sortByNCT != null && sortByNCT.isSelected()) {
            dataSorter.setSortType(SortType.NCT);
        } else {
            dataSorter.setSortType(SortType.VALUE);
        }

        if (sortByName != null && sortByName.isSelected()) {
            dataSorter.setSortType(SortType.NAME);
        }

        
        dataSorter.setUserEventValueType(userEventValueType);
        model.reloadData();
    }

    public List getData() {
        return list;
    }

    public UserEventValueType getValueType() {
        return userEventValueType;
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
            jTextArea.addKeyListener(this);
            jTextArea.setLineWrap(true);
            jTextArea.setWrapStyleWord(true);
            jTextArea.setMargin(new Insets(3, 3, 3, 3));
            jTextArea.setEditable(false);
            jTextArea.append(this.getHeaderString());
            panel.setColumnHeaderView(jTextArea);
        } else
            panel.setColumnHeaderView(null);
    }

    public String getHeaderString() {
        if (userEvent != null) {
            return "Name: " + userEvent.getName() + "\n" + "Value Type: " + userEventValueType.toString() + "\n";
        } else {
            return "Thread: " + ParaProfUtils.getThreadIdentifier(thread) + "\n" + "Value Type: " + userEventValueType.toString()
                    + "\n";
        }
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
            ppTrial.deleteObserver(this);
            ParaProf.decrementNumWindows();
        } catch (Exception e) {
            // do nothing
        }
        dispose();
    }

    public ParaProfTrial getPpTrial() {
        return ppTrial;
    }

    public void showSearchPanel(boolean show) {
        System.out.println("showSearchPanel = " + show);
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
    
    public void keyPressed(KeyEvent e) {
        if (e.isControlDown() && e.getKeyCode() == KeyEvent.VK_F) {
            showSearchPanel(true);
        }
    }

    public void keyReleased(KeyEvent e) {
        // TODO Auto-generated method stub
        
    }

    public void keyTyped(KeyEvent e) {
        // TODO Auto-generated method stub
        
    }

}