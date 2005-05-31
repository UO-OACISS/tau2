/*
 * StaticMainWindow.java
 * 
 * Title: ParaProf 
 * Author: Robert Bell 
 * Description:
 */

package edu.uoregon.tau.paraprof;

import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.List;
import java.util.Observable;
import java.util.Observer;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import javax.swing.event.MenuEvent;

import edu.uoregon.tau.dms.dss.UtilFncs;
import edu.uoregon.tau.paraprof.enums.SortType;
import edu.uoregon.tau.paraprof.enums.ValueType;
import edu.uoregon.tau.paraprof.interfaces.ParaProfWindow;

public class FullDataWindow extends JFrame implements ActionListener, Observer, ChangeListener, ParaProfWindow {

    private int userEventLedgerIndex;
    private int groupLedgerIndex;

    //Instance data.
    private ParaProfTrial ppTrial = null;

    //Create a file chooser to allow the user to select files for loading data.
    private JFileChooser fileChooser = new JFileChooser();

    //References for some of the components for this frame.
    private FullDataWindowPanel panel = null;
    private DataSorter dataSorter = null;

    private JMenu optionsMenu = null;
    private JCheckBoxMenuItem nameCheckBox = null;
    private JCheckBoxMenuItem normalizeCheckBox = null;
    private JCheckBoxMenuItem stackBarsCheckBox = null;
    private JCheckBoxMenuItem orderByMeanCheckBox = null;
    private JCheckBoxMenuItem orderCheckBox = null;
    private JCheckBoxMenuItem slidersCheckBox = null;
    private JCheckBoxMenuItem pathTitleCheckBox = null;
    private JCheckBoxMenuItem metaDataCheckBox = null;

    private JLabel barLengthLabel = new JLabel("Bar Width");
    private JSlider barLengthSlider = new JSlider(0, 2000, 500);

    private Container contentPane = null;
    private GridBagLayout gbl = null;
    private GridBagConstraints gbc = null;
    private JScrollPane jScrollPane;

    private boolean normalizeBars = true;
    private boolean stackBars = true;
    private boolean displaySliders = false;

    private List list;

    public FullDataWindow(ParaProfTrial ppTrial) {
        //This window needs to maintain a reference to its trial.
        this.ppTrial = ppTrial;

        //Window Stuff.

        setTitle("ParaProf: " + ppTrial.getTrialIdentifier(ParaProf.preferences.getShowPathTitleInReverse()));

        int windowWidth = 750;
        int windowHeight = 400;
        setSize(new java.awt.Dimension(windowWidth, windowHeight));

        dataSorter = new DataSorter(ppTrial);

        //Add some window listener code
        addWindowListener(new java.awt.event.WindowAdapter() {
            public void windowClosing(java.awt.event.WindowEvent evt) {
                thisWindowClosing(evt);
            }
        });

        int xPosition = ParaProf.paraProfManager.getLocation().x;
        int yPosition = ParaProf.paraProfManager.getLocation().y;
        setLocation(xPosition + 75, yPosition + 110);

        //Setting up the layout system for the main window.
        contentPane = getContentPane();
        gbl = new GridBagLayout();
        contentPane.setLayout(gbl);
        gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);

        //Panel and ScrollPane definition.
        panel = new FullDataWindowPanel(ppTrial, this);
        jScrollPane = new JScrollPane(panel);

        setupMenus();

        JScrollBar jScrollBar = jScrollPane.getVerticalScrollBar();
        jScrollBar.setUnitIncrement(35);

        this.setHeader();

        barLengthSlider.setPaintTicks(true);
        barLengthSlider.setMajorTickSpacing(400);
        barLengthSlider.setMinorTickSpacing(50);
        barLengthSlider.setPaintLabels(true);
        barLengthSlider.addChangeListener(this);

        gbc.fill = GridBagConstraints.BOTH;
        gbc.anchor = GridBagConstraints.CENTER;
        gbc.weightx = 1;
        gbc.weighty = 1;
        addCompItem(jScrollPane, gbc, 0, 0, 1, 1);

        dataSorter = new DataSorter(ppTrial);

        sortLocalData();

        panel.repaint();

    }

    private void setupMenus() {
        JMenuBar mainMenu = new JMenuBar();

        JMenu subMenu = null;
        JMenuItem menuItem = null;

        //Options menu.
        optionsMenu = new JMenu("Options");

        slidersCheckBox = new JCheckBoxMenuItem("Show Width Slider", false);
        slidersCheckBox.addActionListener(this);
        optionsMenu.add(slidersCheckBox);

        metaDataCheckBox = new JCheckBoxMenuItem("Show Meta Data in Panel", true);
        metaDataCheckBox.addActionListener(this);
        optionsMenu.add(metaDataCheckBox);

        optionsMenu.add(new JSeparator());

        nameCheckBox = new JCheckBoxMenuItem("Sort By Name", false);
        nameCheckBox.addActionListener(this);
        optionsMenu.add(nameCheckBox);

        normalizeCheckBox = new JCheckBoxMenuItem("Normalize Bars", true);
        normalizeCheckBox.addActionListener(this);
        optionsMenu.add(normalizeCheckBox);

        orderByMeanCheckBox = new JCheckBoxMenuItem("Order By Mean", true);
        orderByMeanCheckBox.addActionListener(this);
        optionsMenu.add(orderByMeanCheckBox);

        orderCheckBox = new JCheckBoxMenuItem("Descending Order", true);
        orderCheckBox.addActionListener(this);
        optionsMenu.add(orderCheckBox);

        stackBarsCheckBox = new JCheckBoxMenuItem("Stack Bars Together", true);
        stackBarsCheckBox.addActionListener(this);
        optionsMenu.add(stackBarsCheckBox);

        //Now, add all the menus to the main menu.
        mainMenu.add(ParaProfUtils.createFileMenu(this, panel, panel));
        mainMenu.add(optionsMenu);
        //mainMenu.add(ParaProfUtils.createTrialMenu(ppTrial, this));

        //mainMenu.add(ParaProfUtils.createThreadMenu(ppTrial, this, null));
        //mainMenu.add(ParaProfUtils.createFunctionMenu(ppTrial, this, null));
        mainMenu.add(ParaProfUtils.createWindowsMenu(ppTrial, this));
        mainMenu.add(ParaProfUtils.createHelpMenu(this, this));

        setJMenuBar(mainMenu);
    }

    public void actionPerformed(ActionEvent evt) {

        try {
            Object EventSrc = evt.getSource();

            if (EventSrc instanceof JMenuItem) {
                String arg = evt.getActionCommand();

                if (arg.equals("Sort By Name")) {
                    sortLocalData();
                    panel.repaint();
                } else if (arg.equals("Normalize Bars")) {
                    if (normalizeCheckBox.isSelected())
                        normalizeBars = true;
                    else
                        normalizeBars = false;
                    panel.repaint();

                } else if (arg.equals("Stack Bars Together")) {
                    if (stackBarsCheckBox.isSelected()) {

                        normalizeCheckBox.setEnabled(true);
                        orderByMeanCheckBox.setEnabled(true);

                        stackBars = true;
                    } else {
                        stackBars = false;

                        normalizeCheckBox.setSelected(false);
                        normalizeCheckBox.setEnabled(false);
                        normalizeBars = false;
                        orderByMeanCheckBox.setSelected(true);
                        orderByMeanCheckBox.setEnabled(false);
                    }
                    panel.repaint();
                } else if (arg.equals("Order By Mean")) {
                    sortLocalData();
                    panel.repaint();
                } else if (arg.equals("Descending Order")) {
                    sortLocalData();
                    panel.repaint();
                } else if (arg.equals("Show Width Slider")) {
                    if (slidersCheckBox.isSelected())
                        showWidthSlider(true);
                    else
                        showWidthSlider(false);
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
            this.setHeader();
            panel.repaint();
        } else if (tmpString.equals("colorEvent")) {
            panel.repaint();
        } else if (tmpString.equals("dataEvent")) {
            sortLocalData();
            this.setHeader();
            panel.repaint();
        }
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

    //######
    //Panel header.
    //######
    //This process is separated into two functionProfiles to provide the option
    //of obtaining the current header string being used for the panel
    //without resetting the actual header. Printing and image generation
    //use this functionality for example.
    public void setHeader() {
        if (metaDataCheckBox.isSelected()) {
            JTextArea jTextArea = new JTextArea();
            jTextArea.setLineWrap(true);
            jTextArea.setWrapStyleWord(true);
            jTextArea.setEditable(false);
            PreferencesWindow p = ppTrial.getPreferencesWindow();
            jTextArea.setFont(new Font(p.getParaProfFont(), p.getFontStyle(), p.getFontSize()));
            jTextArea.append(this.getHeaderString());
            jScrollPane.setColumnHeaderView(jTextArea);
        } else
            jScrollPane.setColumnHeaderView(null);
    }

    public String getHeaderString() {
        return "Metric Name: " + (ppTrial.getMetricName(ppTrial.getDefaultMetricID())) + "\n" + "Value Type: "
                + UtilFncs.getValueTypeString(2) + "\n";
    }

    private void showWidthSlider(boolean displaySliders) {
        if (displaySliders) {
            contentPane.remove(jScrollPane);

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

            gbc.insets = new Insets(0, 0, 0, 0);
            gbc.fill = GridBagConstraints.BOTH;
            gbc.anchor = GridBagConstraints.CENTER;
            gbc.weightx = 1.0;
            gbc.weighty = 0.99;
            addCompItem(jScrollPane, gbc, 0, 1, 2, 1);
        } else {
            contentPane.remove(barLengthLabel);
            contentPane.remove(barLengthSlider);
            contentPane.remove(jScrollPane);

            gbc.fill = GridBagConstraints.BOTH;
            gbc.anchor = GridBagConstraints.CENTER;
            gbc.weightx = 1;
            gbc.weighty = 1;
            addCompItem(jScrollPane, gbc, 0, 0, 1, 1);
        }

        //Now call validate so that these component changes are displayed.
        validate();
    }

    private void addCompItem(Component c, GridBagConstraints gbc, int x, int y, int w, int h) {
        gbc.gridx = x;
        gbc.gridy = y;
        gbc.gridwidth = w;
        gbc.gridheight = h;
        contentPane.add(c, gbc);
    }

    public DataSorter getDataSorter() {
        return dataSorter;
    }

    //Updates the sorted lists after a change of sorting method takes place.
    private void sortLocalData() {
        dataSorter.setSelectedMetricID(ppTrial.getDefaultMetricID());
        dataSorter.setValueType(ValueType.EXCLUSIVE);

        if (nameCheckBox.isSelected()) {
            dataSorter.setSortType(SortType.NAME);
        } else {
            if (orderByMeanCheckBox.isSelected()) {
                dataSorter.setSortType(SortType.MEAN_VALUE);
            } else {
                dataSorter.setSortType(SortType.MEAN_VALUE);
            }
        }
        dataSorter.setDescendingOrder(orderCheckBox.isSelected());
        list = dataSorter.getAllFunctionProfiles();
    }

    public List getData() {
        return list;
    }

    private boolean mShown = false;

    public void addNotify() {
        super.addNotify();

        if (mShown)
            return;

        // resize frame to account for menubar
        JMenuBar jMenuBar = getJMenuBar();
        if (jMenuBar != null) {
            int jMenuBarHeight = jMenuBar.getPreferredSize().height;
            Dimension dimension = getSize();
            dimension.height += jMenuBarHeight;
            setSize(dimension);
        }

        mShown = true;
    }

    //Close the window when the close box is clicked
    void thisWindowClosing(java.awt.event.WindowEvent e) {
        closeThisWindow();
    }

    public void closeThisWindow() {
        try {
            setVisible(false);

            // don't do this!
            //trial.getSystemEvents().deleteObserver(this);
            ParaProf.decrementNumWindows();
        } catch (Exception e) {
            // do nothing
        }
        dispose();
    }

    public boolean getNormalizeBars() {
        return this.normalizeBars;
    }

    public boolean getStackBars() {
        return this.stackBars;
    }

    public void help(boolean display) {
        ParaProf.helpWindow.show();
    }

}