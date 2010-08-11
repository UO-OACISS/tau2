package edu.uoregon.tau.paraprof;

import java.awt.Component;
import java.awt.Container;
import java.awt.Dimension;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.awt.Rectangle;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.util.List;
import java.util.ListIterator;
import java.util.Observable;
import java.util.Observer;

import javax.swing.ButtonGroup;
import javax.swing.JCheckBoxMenuItem;
import javax.swing.JFrame;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JRadioButtonMenuItem;
import javax.swing.JScrollBar;
import javax.swing.JScrollPane;
import javax.swing.JSeparator;
import javax.swing.JTextArea;
import javax.swing.event.MenuEvent;
import javax.swing.event.MenuListener;

import edu.uoregon.tau.paraprof.enums.SortType;
import edu.uoregon.tau.paraprof.enums.ValueType;
import edu.uoregon.tau.paraprof.interfaces.ParaProfWindow;
import edu.uoregon.tau.paraprof.interfaces.ScrollBarController;
import edu.uoregon.tau.paraprof.interfaces.SearchableOwner;
import edu.uoregon.tau.paraprof.interfaces.UnitListener;
import edu.uoregon.tau.perfdmf.Thread;
import edu.uoregon.tau.perfdmf.UtilFncs;

/**
 * CallPathTextWindow: This window displays callpath data in a text format
 *   
 * <P>CVS $Id: CallPathTextWindow.java,v 1.32 2009/09/10 00:13:44 amorris Exp $</P>
 * @author	Robert Bell, Alan Morris
 * @version	$Revision: 1.32 $
 * @see		CallPathDrawObject
 * @see		CallPathTextWindowPanel
 */
public class CallPathTextWindow extends JFrame implements ActionListener, MenuListener, Observer, SearchableOwner,
        ScrollBarController, KeyListener, ParaProfWindow, UnitListener {

    /**
	 * 
	 */
	private static final long serialVersionUID = -8482671835939047124L;
	private ParaProfTrial ppTrial = null;
    private DataSorter dataSorter = null;

    private JMenu optionsMenu = null;
    private JMenu unitsSubMenu = null;

    private boolean sortByName;
    private JCheckBoxMenuItem descendingOrder = null;
    private JCheckBoxMenuItem collapsedView = null;
    //private JCheckBoxMenuItem showPathTitleInReverse = null;
    private JCheckBoxMenuItem showMetaData = null;
    private JCheckBoxMenuItem showFindPanelBox;

    private JScrollPane sp;
    private CallPathTextWindowPanel panel;

    private List<PPFunctionProfile> list;

    private SearchPanel searchPanel;

    //private int order = 0;
    private int units = ParaProf.preferences.getUnits();

    private Thread thread;

    public CallPathTextWindow(ParaProfTrial ppTrial, Thread thread, Component invoker) {
        this.ppTrial = ppTrial;
        ppTrial.addObserver(this);
        this.dataSorter = new DataSorter(ppTrial);
        this.thread = thread;

        setSize(ParaProfUtils.checkSize(new java.awt.Dimension(800, 600)));
        setLocation(WindowPlacer.getNewLocation(this, invoker));

        addKeyListener(this);

        //Now set the title.
        //if (windowType == 0) {
        if (thread.getNodeID() == -1) {
            this.setTitle("TAU: ParaProf: Mean Call Path Data - "
                    + ppTrial.getTrialIdentifier(ParaProf.preferences.getShowPathTitleInReverse()));
        } else if (thread.getNodeID() == -3) {
            this.setTitle("TAU: ParaProf: Standard Deviation Call Path Data - "
                    + ppTrial.getTrialIdentifier(ParaProf.preferences.getShowPathTitleInReverse()));
        } else {
            this.setTitle("TAU: ParaProf: Call Path Data " + "n,c,t, " + thread.getNodeID() + "," + thread.getContextID() + ","
                    + thread.getThreadID() + " - " + ppTrial.getTrialIdentifier(ParaProf.preferences.getShowPathTitleInReverse()));
        }
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

        //Setting up the layout system for the main window.
        Container contentPane = getContentPane();
        GridBagLayout gbl = new GridBagLayout();
        contentPane.setLayout(gbl);
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);

        panel = new CallPathTextWindowPanel(ppTrial, this.thread, this);
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

        //Options menu.
        optionsMenu = new JMenu("Options");

        //JCheckBoxMenuItem box = null;
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

        unitsSubMenu = ParaProfUtils.createUnitsMenu(this, units, true);
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
                if (arg.equals("Name")) {
                    sortByName = true;
                    sortLocalData();
                    panel.resetAllDrawObjects();
                    panel.repaint();
                } else if (arg.equals("Descending Order")) {
                    //if (descendingOrder.isSelected())
                        //order = 0;
                    //else
                        //order = 1;
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

    public void menuDeselected(MenuEvent evt) {}

    public void menuCanceled(MenuEvent evt) {}

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
        ParaProf.getHelpWindow().clearText();
        if (display) {
            ParaProf.getHelpWindow().setVisible(true);
        }
        ParaProf.getHelpWindow().writeText("Call path text window.");
        ParaProf.getHelpWindow().writeText("");
        ParaProf.getHelpWindow().writeText("This window displays call path relationships in two ways:");
        ParaProf.getHelpWindow().writeText("1- If this window has been invoked from the \"windows\" menu of");
        ParaProf.getHelpWindow().writeText("ParaProf, the information displayed is all call path relations found.");
        ParaProf.getHelpWindow().writeText("That is, all the parent/child relationships.");
        ParaProf.getHelpWindow().writeText("Thus, in this case, given the parallel nature of ParaProf, this information");
        ParaProf.getHelpWindow().writeText("might not be valid for a particular thread. It is however useful to observe");
        ParaProf.getHelpWindow().writeText("all the realtionships that exist in the data.");
        ParaProf.getHelpWindow().writeText("");
        ParaProf.getHelpWindow().writeText("2- If this window has been invoked from the popup menu to the left of a thread bar");
        ParaProf.getHelpWindow().writeText(
                "in the main ParaProf window, the information dispayed will be specific to this thread,");
        ParaProf.getHelpWindow().writeText("and will thus contain both parent/child relations and the data relating to those");
        ParaProf.getHelpWindow().writeText("relationships.");
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
        dataSorter.setSelectedMetric(ppTrial.getDefaultMetric());

        this.setHeader();

        list = dataSorter.getFunctionProfiles(thread);
    }

    public List<PPFunctionProfile> getData() {
        return list;
    }

    public ListIterator<PPFunctionProfile> getDataIterator() {
        return list.listIterator();
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
            jTextArea.setFont(ParaProf.preferencesWindow.getFont());
            jTextArea.append(this.getHeaderString());
            jTextArea.setMargin(new Insets(3, 3, 3, 3));

            sp.setColumnHeaderView(jTextArea);
        } else
            sp.setColumnHeaderView(null);
    }

    public String getHeaderString() {
        return "Metric Name: " + (dataSorter.getSelectedMetric().getName()) + "\n" + "Sorted By: "
                + dataSorter.getValueType() + "\n" + "Units: "
                + UtilFncs.getUnitsString(units, dataSorter.isTimeMetric(), dataSorter.isDerivedMetric()) + "\n";
    }

    public void closeThisWindow() {
        setVisible(false);
        ppTrial.deleteObserver(this);
        ParaProf.decrementNumWindows();
        dispose();
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

    public void keyReleased(KeyEvent e) {}

    public void keyTyped(KeyEvent e) {}

    public void setUnits(int units) {
        this.units = units;
        this.setHeader();
        panel.repaint();

    }

    public JScrollPane getScrollPane() {
        return sp;
    }
    public JFrame getFrame() {
        return this;
    }
}