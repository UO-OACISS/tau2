package edu.uoregon.tau.paraprof;

import java.awt.*;
import java.awt.event.*;
import java.util.*;

import javax.swing.*;
import javax.swing.event.MenuEvent;
import javax.swing.event.MenuListener;

import edu.uoregon.tau.dms.dss.DssIterator;
import edu.uoregon.tau.dms.dss.UtilFncs;
import edu.uoregon.tau.paraprof.enums.SortType;
import edu.uoregon.tau.paraprof.enums.ValueType;
import edu.uoregon.tau.paraprof.interfaces.*;
import edu.uoregon.tau.paraprof.interfaces.ParaProfWindow;
import edu.uoregon.tau.paraprof.interfaces.ScrollBarController;
import edu.uoregon.tau.paraprof.interfaces.SearchableOwner;

/**
 * CallPathTextWindow: This window displays callpath data in a text format
 *   
 * <P>CVS $Id: CallPathTextWindow.java,v 1.24 2005/05/10 01:48:37 amorris Exp $</P>
 * @author	Robert Bell, Alan Morris
 * @version	$Revision: 1.24 $
 * @see		CallPathDrawObject
 * @see		CallPathTextWindowPanel
 */
public class CallPathTextWindow extends JFrame implements ActionListener, MenuListener, Observer, SearchableOwner,
        ScrollBarController, KeyListener, ParaProfWindow, UnitListener {

    private ParaProfTrial ppTrial = null;
    private int nodeID = -1;
    private int contextID = -1;
    private int threadID = -1;
    private DataSorter dataSorter = null;
    private int windowType = 0; //0: mean data,1: function data, 2: global relations.

    private JMenu optionsMenu = null;
    private JMenu unitsSubMenu = null;

    private boolean sortByName;
    private JCheckBoxMenuItem descendingOrder = null;
    private JCheckBoxMenuItem collapsedView = null;
    private JCheckBoxMenuItem showPathTitleInReverse = null;
    private JCheckBoxMenuItem showMetaData = null;
    private JCheckBoxMenuItem showFindPanelBox;

    private JScrollPane sp = null;
    private CallPathTextWindowPanel panel = null;

    private Vector list = new Vector();

    private SearchPanel searchPanel;

    private int order = 0;
    private int units = ParaProf.preferences.getUnits();

    public CallPathTextWindow(ParaProfTrial ppTrial, int nodeID, int contextID, int threadID, DataSorter dataSorter,
            int windowType) {

        this.ppTrial = ppTrial;
        this.nodeID = nodeID;
        this.contextID = contextID;
        this.threadID = threadID;
        this.dataSorter = new DataSorter(ppTrial);
        this.windowType = windowType;

        setLocation(0, 0);
        setSize(800, 600);

        addKeyListener(this);

        //Now set the title.
        if (windowType == 0) {
            this.setTitle("Mean Call Path Data - " + ppTrial.getTrialIdentifier(ParaProf.preferences.getShowPathTitleInReverse()));
        } else if (windowType == 1) {
            this.setTitle("Call Path Data " + "n,c,t, " + nodeID + "," + contextID + "," + threadID + " - "
                    + ppTrial.getTrialIdentifier(true));
            //CallPathUtilFuncs.trimCallPathData(trial.getDataSource(), trial.getDataSource().getThread(nodeID, contextID, threadID));
        } else
            this.setTitle("Call Path Data Relations - " + ppTrial.getTrialIdentifier(ParaProf.preferences.getShowPathTitleInReverse()));

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

       

        //####################################
        //Create and add the components.
        //####################################
        //Setting up the layout system for the main window.
        Container contentPane = getContentPane();
        GridBagLayout gbl = new GridBagLayout();
        contentPane.setLayout(gbl);
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);

        edu.uoregon.tau.dms.dss.Thread thread = ppTrial.getDataSource().getThread(nodeID, contextID, threadID);

        if (windowType == 0 || windowType == 2) {
            thread = ppTrial.getDataSource().getMeanData();
        }

        panel = new CallPathTextWindowPanel(ppTrial, thread, this, windowType);
        //The scroll panes into which the list shall be placed.

        setupMenus();
        sp = new JScrollPane(panel);

        JScrollBar vScrollBar = sp.getVerticalScrollBar();
        vScrollBar.setUnitIncrement(35);

        //Now add the componants to the main screen.
        gbc.fill = GridBagConstraints.BOTH;
        gbc.anchor = GridBagConstraints.CENTER;
        gbc.weightx = 1;
        gbc.weighty = 1;
        addCompItem(sp, gbc, 0, 0, 1, 1);

        sortLocalData();

        ParaProf.incrementNumWindows();

    }

    private void setupMenus() {
        JMenuBar mainMenu = new JMenuBar();
        mainMenu.addKeyListener(this);
        JMenu subMenu = null;
        JMenuItem menuItem = null;



        //Options menu.
        optionsMenu = new JMenu("Options");

        JCheckBoxMenuItem box = null;
        ButtonGroup group = null;
        JRadioButtonMenuItem button = null;

        showFindPanelBox = new JCheckBoxMenuItem("Show Find Panel", false);
        showFindPanelBox.addActionListener(this);
        optionsMenu.add(showFindPanelBox);

        showMetaData = new JCheckBoxMenuItem("Show Meta Data in Panel", true);
        showMetaData.addActionListener(this);
        optionsMenu.add(showMetaData);

        optionsMenu.add(new JSeparator());

        collapsedView = new JCheckBoxMenuItem("Collapsible View", false);
        collapsedView.addActionListener(this);
        optionsMenu.add(collapsedView);

        unitsSubMenu = ParaProfUtils.createUnitsMenu(this, units);
        optionsMenu.add(unitsSubMenu);
        //End - Units submenu.

        //Set the value type options.
        subMenu = new JMenu("Sort By");
        group = new ButtonGroup();

        button = new JRadioButtonMenuItem("Name", false);
        button.addActionListener(this);
        group.add(button);
        subMenu.add(button);

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
        //End - Set the value type options.

        descendingOrder = new JCheckBoxMenuItem("Descending Order", true);
        descendingOrder.addActionListener(this);
        optionsMenu.add(descendingOrder);


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
                if (arg.equals("Print")) {
                    ParaProfUtils.print(panel);
                } else if (arg.equals("Preferences...")) {
                    ppTrial.getPreferencesWindow().showPreferencesWindow();
                } else if (arg.equals("Save Image")) {
                    ParaProfImageOutput.saveImage(panel);
                } else if (arg.equals("Close This Window")) {
                    closeThisWindow();
                } else if (arg.equals("Exit ParaProf!")) {
                    setVisible(false);
                    dispose();
                    ParaProf.exitParaProf(0);
                } else if (arg.equals("Name")) {
                    sortByName = true;
                    sortLocalData();
                    panel.resetAllDrawObjects();
                    panel.repaint();
                } else if (arg.equals("Descending Order")) {
                    if (descendingOrder.isSelected())
                        order = 0;
                    else
                        order = 1;
                    sortLocalData();
                    panel.resetAllDrawObjects();
                    panel.repaint();
                } else if (arg.equals("Exclusive")) {
                    sortByName = false;
                    dataSorter.setValueType(ValueType.EXCLUSIVE);
                    sortLocalData();
                    panel.resetAllDrawObjects();
                    panel.repaint();
                } else if (arg.equals("Inclusive")) {
                    sortByName = false;
                    dataSorter.setValueType(ValueType.INCLUSIVE);
                    sortLocalData();
                    panel.resetAllDrawObjects();
                    panel.repaint();
                } else if (arg.equals("Number of Calls")) {
                    sortByName = false;
                    dataSorter.setValueType(ValueType.NUMCALLS);
                    sortLocalData();
                    panel.resetAllDrawObjects();
                    panel.repaint();
                } else if (arg.equals("Number of Child Calls")) {
                    sortByName = false;
                    dataSorter.setValueType(ValueType.NUMSUBR);
                    sortLocalData();
                    panel.resetAllDrawObjects();
                    panel.repaint();
                } else if (arg.equals("Inclusive per Call")) {
                    sortByName = false;
                    dataSorter.setValueType(ValueType.INCLUSIVE_PER_CALL);
                    sortLocalData();
                    panel.resetAllDrawObjects();
                    panel.repaint();
                } else if (arg.equals("Exclusive per Call")) {
                    sortByName = false;
                    dataSorter.setValueType(ValueType.EXCLUSIVE_PER_CALL);
                    sortLocalData();
                    panel.resetAllDrawObjects();
                    panel.repaint();
                } else if (arg.equals("Collapsible View")) {
                    panel.resetAllDrawObjects();
                    panel.repaint();
                } else if (arg.equals("Show Meta Data in Panel")) {
                    this.setHeader();
                } else if (arg.equals("Show Find Panel")) {
                    if (showFindPanelBox.isSelected())
                        showSearchPanel(true);
                    else
                        showSearchPanel(false);
                }
            }
        } catch (Exception e) {
            ParaProfUtils.handleException(e);
        }
    }

    public void menuSelected(MenuEvent evt) {
        try {
            if (ppTrial.isTimeMetric())
                unitsSubMenu.setEnabled(true);
            else
                unitsSubMenu.setEnabled(false);

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
            this.setHeader();
            sortLocalData();
            panel.resetAllDrawObjects();
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
        ParaProf.helpWindow.writeText("Call path text window.");
        ParaProf.helpWindow.writeText("");
        ParaProf.helpWindow.writeText("This window displays call path relationships in two ways:");
        ParaProf.helpWindow.writeText("1- If this window has been invoked from the \"windows\" menu of");
        ParaProf.helpWindow.writeText("ParaProf, the information displayed is all call path relations found.");
        ParaProf.helpWindow.writeText("That is, all the parent/child relationships.");
        ParaProf.helpWindow.writeText("Thus, in this case, given the parallel nature of ParaProf, this information");
        ParaProf.helpWindow.writeText("might not be valid for a particular thread. It is however useful to observe");
        ParaProf.helpWindow.writeText("all the realtionships that exist in the data.");
        ParaProf.helpWindow.writeText("");
        ParaProf.helpWindow.writeText("2- If this window has been invoked from the popup menu to the left of a thread bar");
        ParaProf.helpWindow.writeText("in the main ParaProf window, the information dispayed will be specific to this thread,");
        ParaProf.helpWindow.writeText("and will thus contain both parent/child relations and the data relating to those");
        ParaProf.helpWindow.writeText("relationships.");
    }

    public DataSorter getDataSorter() {
        return dataSorter;
    }

    //Updates this window's data copy.
    private void sortLocalData() {
        //The name selection behaves slightly differently. Thus the check for it.

        if (sortByName)
            dataSorter.setSortType(SortType.NAME);
        else
            dataSorter.setSortType(SortType.VALUE);

        dataSorter.setDescendingOrder(descendingOrder.isSelected());
        dataSorter.setSelectedMetricID(ppTrial.getDefaultMetricID());

        this.setHeader();

        if (sortByName) {
            if (windowType == 0 || windowType == 1) {
                list = dataSorter.getFunctionProfiles(nodeID, contextID, threadID);
            } else {
                list = new Vector();
                for (Iterator it = ppTrial.getDataSource().getFunctions(); it.hasNext();)
                    list.add(it.next());
            }
        } else {
            if (windowType == 0 || windowType == 1) {
                list = dataSorter.getFunctionProfiles(nodeID, contextID, threadID);
            } else {
                list = new Vector();
                for (Iterator it = ppTrial.getDataSource().getFunctions(); it.hasNext();)
                    list.add(it.next());
                Collections.sort(list);
            }
        }

    }

    public Vector getData() {
        return list;
    }

    public ListIterator getDataIterator() {
        return new DssIterator(this.getData());
    }

    public int getWindowType() {
        return windowType;
    }

    public int units() {
        if (ppTrial.isTimeMetric())
            return units;
        return 0;
    }

    public boolean showCollapsedView() {
        return collapsedView.isSelected();
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
            jTextArea.addKeyListener(this);
            PreferencesWindow p = ppTrial.getPreferencesWindow();
            jTextArea.setFont(new Font(p.getParaProfFont(), p.getFontStyle(), p.getFontSize()));
            jTextArea.append(this.getHeaderString());
            sp.setColumnHeaderView(jTextArea);
        } else
            sp.setColumnHeaderView(null);
    }

    public String getHeaderString() {
        return "Metric Name: " + (ppTrial.getMetricName(dataSorter.getSelectedMetricID())) + "\n" + "Sorted By: "
                + dataSorter.getValueType() + "\n" + "Units: "
                + UtilFncs.getUnitsString(units, dataSorter.isTimeMetric(), dataSorter.isDerivedMetric()) + "\n";
    }

    //######
    //End - Panel header.
    //######

    public void closeThisWindow() {
        setVisible(false);
        ppTrial.getSystemEvents().deleteObserver(this);
        ParaProf.decrementNumWindows();
        dispose();
    }

    public void showSearchPanel(boolean show) {
        // TODO Auto-generated method stub
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

    public void setHorizontalScrollBarPosition(int position) {
        JScrollBar scrollBar = sp.getHorizontalScrollBar();
        scrollBar.setValue(position);
    }

    public Dimension getThisViewportSize() {
        return this.getViewportSize();
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

    public void setUnits(int units) {
        this.units = units; 
        this.setHeader();
        panel.repaint();
        
    }
}