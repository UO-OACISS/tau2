package edu.uoregon.tau.paraprof;

import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.Observable;
import java.util.Observer;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import javax.swing.event.MenuEvent;

import edu.uoregon.tau.paraprof.barchart.AbstractBarChartModel;
import edu.uoregon.tau.paraprof.barchart.BarChartPanel;
import edu.uoregon.tau.paraprof.barchart.GlobalBarChartModel;
import edu.uoregon.tau.paraprof.enums.SortType;
import edu.uoregon.tau.paraprof.enums.ValueType;
import edu.uoregon.tau.paraprof.interfaces.ParaProfWindow;
import edu.uoregon.tau.paraprof.interfaces.SortListener;
import edu.uoregon.tau.perfdmf.Function;

/**
 * The GlobalDataWindow shows the exclusive value for all functions/all threads for a trial.
 * 
 * <P>CVS $Id: GlobalDataWindow.java,v 1.22 2008/08/05 18:45:42 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.22 $
 * @see GlobalBarChartModel
 */
public class GlobalDataWindow extends JFrame implements ActionListener, Observer, ChangeListener, ParaProfWindow, SortListener {

    private ParaProfTrial ppTrial;
    private Function phase; // null for non-phase profiles

    private BarChartPanel panel;
    //    private GlobalBarChartModel model;
    private AbstractBarChartModel model;
    private DataSorter dataSorter;

    private JMenu optionsMenu;
    private JCheckBoxMenuItem nameCheckBox = new JCheckBoxMenuItem("Sort By Name", false);
    private JCheckBoxMenuItem normalizeCheckBox = new JCheckBoxMenuItem("Normalize Bars", true);
    private JCheckBoxMenuItem orderByMeanCheckBox = new JCheckBoxMenuItem("Order By Mean", true);
    private JCheckBoxMenuItem orderCheckBox = new JCheckBoxMenuItem("Descending Order", true);
    private JCheckBoxMenuItem stackBarsCheckBox = new JCheckBoxMenuItem("Stack Bars Together", true);
    private JCheckBoxMenuItem slidersCheckBox;
    private JCheckBoxMenuItem metaDataCheckBox;

    private JLabel barLengthLabel = new JLabel("Bar Width");
    private JSlider barLengthSlider = new JSlider(0, 2000, 600);

    private boolean visible = false;

    private static int defaultWidth = 750;
    private static int defaultHeight = 410;

    
    // we keep these around to speed things up
    private JTextArea jTextArea;
    private Component headerView;


    public BarChartPanel getPanel() {
        return panel;
    }

    public GlobalDataWindow(ParaProfTrial ppTrial, Function phase) {
        this.ppTrial = ppTrial;
        this.phase = phase;
        ppTrial.addObserver(this);

        dataSorter = new DataSorter(ppTrial);
        dataSorter.setPhase(phase);

        if (phase == null) {
            setTitle("TAU: ParaProf: " + ppTrial.getTrialIdentifier(ParaProf.preferences.getShowPathTitleInReverse()));
        } else {
            setTitle("TAU: ParaProf: " + ppTrial.getTrialIdentifier(ParaProf.preferences.getShowPathTitleInReverse())
                    + " Phase: " + phase.getName());
        }
        ParaProfUtils.setFrameIcon(this);

        setSize(ParaProfUtils.checkSize(new java.awt.Dimension(defaultWidth, defaultHeight)));
        setLocation(WindowPlacer.getGlobalDataWindowPosition(this));

        addWindowListener(new java.awt.event.WindowAdapter() {
            public void windowClosing(java.awt.event.WindowEvent evt) {
                thisWindowClosing(evt);
            }
        });

        if (ParaProf.demoMode) { // for Scott's quicktime videos
            barLengthSlider.setValue(500);
        }

        getContentPane().setLayout(new GridBagLayout());

        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(2, 2, 2, 2);

        model = new GlobalBarChartModel(this, dataSorter, ppTrial);
        //model = new ThreadSnapshotBarChartModel(this, dataSorter, ppTrial);
        panel = new BarChartPanel(model);
        setupMenus();

        panel.getBarChart().setLeftJustified(true);
        panel.getBarChart().setAutoResize(true);
        //panel.getBarChart().setBarLength(barLengthSlider.getValue());

        // more sane scrollbar sensitivity
        panel.getVerticalScrollBar().setUnitIncrement(35);

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
        ParaProfUtils.addCompItem(this, panel, gbc, 0, 0, 1, 1);

        sortLocalData();

        panel.repaint();

      
        
        ParaProf.incrementNumWindows();
    }

    public AbstractBarChartModel getModel() {
        return model;
    }

    private void setupMenus() {
        JMenuBar mainMenu = new JMenuBar();

        optionsMenu = new JMenu("Options");

        slidersCheckBox = new JCheckBoxMenuItem("Show Width Slider", false);
        slidersCheckBox.addActionListener(this);
        //optionsMenu.add(slidersCheckBox);

        metaDataCheckBox = new JCheckBoxMenuItem("Show Meta Data in Panel", true);
        metaDataCheckBox.addActionListener(this);
        optionsMenu.add(metaDataCheckBox);

        optionsMenu.add(new JSeparator());

        //nameCheckBox.addActionListener(this);
        //optionsMenu.add(nameCheckBox);

        normalizeCheckBox.addActionListener(this);
        optionsMenu.add(normalizeCheckBox);

        //        orderByMeanCheckBox.addActionListener(this);
        //        optionsMenu.add(orderByMeanCheckBox);

        orderCheckBox.addActionListener(this);
        optionsMenu.add(orderCheckBox);

        stackBarsCheckBox.addActionListener(this);
        optionsMenu.add(stackBarsCheckBox);

        optionsMenu.add(ParaProfUtils.createMetricSelectionMenu(ppTrial, "Select Metric...", false, false, dataSorter, this, true));
        optionsMenu.add(ParaProfUtils.createMetricSelectionMenu(ppTrial, "Sort by...", true, false, dataSorter, this, true));

        mainMenu.add(ParaProfUtils.createFileMenu(this, panel, panel));
        mainMenu.add(optionsMenu);
        mainMenu.add(ParaProfUtils.createWindowsMenu(ppTrial, this));
        if (ParaProf.scripts.size() > 0) {
            mainMenu.add(ParaProfUtils.createScriptMenu(ppTrial, this));
        }
        mainMenu.add(ParaProfUtils.createHelpMenu(this, this));

        setJMenuBar(mainMenu);
    }

    public void actionPerformed(ActionEvent evt) {

        try {
            Object EventSrc = evt.getSource();

            if (EventSrc instanceof JMenuItem) {
                String arg = evt.getActionCommand();

                if (arg.equals("Sort By Name")) {
                    setSortByName(nameCheckBox.isSelected());
                } else if (arg.equals("Normalize Bars")) {
                    setNormalized(normalizeCheckBox.isSelected());
                } else if (arg.equals("Stack Bars Together")) {
                    setStackBars(stackBarsCheckBox.isSelected());
                } else if (arg.equals("Order By Mean")) {
                    setSortByMean(orderByMeanCheckBox.isSelected());
                } else if (arg.equals("Descending Order")) {
                    setDescendingOrder(orderCheckBox.isSelected());
                } else if (arg.equals("Show Width Slider")) {
                    showWidthSlider(slidersCheckBox.isSelected());
                } else if (arg.equals("Show Meta Data in Panel")) {
                    setShowMetaData(metaDataCheckBox.isSelected());
                }
            }
        } catch (Exception e) {
            ParaProfUtils.handleException(e);
        }
    }

    public void stateChanged(ChangeEvent event) {
        try {
            setBarLength(barLengthSlider.getValue());
        } catch (Exception e) {
            ParaProfUtils.handleException(e);
        }
    }

    public int getBarLength() {
        return barLengthSlider.getValue();
    }

    public void setBarLength(int length) {
        barLengthSlider.setValue(length);
        panel.getBarChart().setBarLength(barLengthSlider.getValue());
        panel.repaint();
    }

    public boolean getWidthSliderShown() {
        return slidersCheckBox.isSelected();
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
            dataSorter.setSelectedMetricID(ppTrial.getDefaultMetricID());
            setupMenus();
            validate(); // must call validate or the new JMenuBar won't work
            sortLocalData();
            this.setHeader();
            panel.repaint();
        } else if (tmpString.equals("subWindowCloseEvent")) {
            if (this != ppTrial.getFullDataWindow()) {
                closeThisWindow();
            }
        }
    }

    public Dimension getViewportSize() {
        return panel.getViewport().getExtentSize();
    }

    public Rectangle getViewRect() {
        return panel.getViewport().getViewRect();
    }

    public void setHeader() {
        if (metaDataCheckBox.isSelected()) {
            if (jTextArea == null) {
                jTextArea = new JTextArea();
                jTextArea.setLineWrap(true);
                jTextArea.setWrapStyleWord(true);
                jTextArea.setEditable(false);
                jTextArea.setMargin(new Insets(3, 3, 3, 3));
                jTextArea.setFont(ParaProf.preferencesWindow.getFont());
            }
            jTextArea.setText(getHeaderString());
            if (headerView != jTextArea) {
                panel.setColumnHeaderView(jTextArea);
                headerView = jTextArea;
            }
            panel.setColumnHeaderView(jTextArea);
        } else {
            panel.setColumnHeaderView(null);
            headerView = jTextArea;
        }
    }

    public String getHeaderString() {
        if (phase != null) {
            return "Phase: " + phase + "\nMetric: " + (ppTrial.getMetricName(ppTrial.getDefaultMetricID())) + "\nValue: "
                    + "Exclusive" + "\n";
        } else {
            return "Metric: " + (ppTrial.getMetricName(dataSorter.getSelectedMetricID())) + "\nValue: " + dataSorter.getValueType() + "\n";
        }
    }

    public void showWidthSlider(boolean displaySliders) {
        GridBagConstraints gbc = new GridBagConstraints();

        slidersCheckBox.setSelected(displaySliders);
        if (displaySliders) {
            getContentPane().remove(panel);

            gbc.insets = new Insets(5, 5, 5, 5);
            gbc.fill = GridBagConstraints.NONE;
            gbc.anchor = GridBagConstraints.EAST;
            gbc.weightx = 0.10;
            gbc.weighty = 0.01;
            ParaProfUtils.addCompItem(this, barLengthLabel, gbc, 0, 0, 1, 1);

            gbc.fill = GridBagConstraints.HORIZONTAL;
            gbc.anchor = GridBagConstraints.WEST;
            gbc.weightx = 0.70;
            gbc.weighty = 0.01;
            ParaProfUtils.addCompItem(this, barLengthSlider, gbc, 1, 0, 1, 1);

            gbc.insets = new Insets(0, 0, 0, 0);
            gbc.fill = GridBagConstraints.BOTH;
            gbc.anchor = GridBagConstraints.CENTER;
            gbc.weightx = 1.0;
            gbc.weighty = 0.99;
            ParaProfUtils.addCompItem(this, panel, gbc, 0, 1, 2, 1);
        } else {
            getContentPane().remove(barLengthLabel);
            getContentPane().remove(barLengthSlider);
            getContentPane().remove(panel);

            gbc.fill = GridBagConstraints.BOTH;
            gbc.anchor = GridBagConstraints.CENTER;
            gbc.weightx = 1;
            gbc.weighty = 1;
            ParaProfUtils.addCompItem(this, panel, gbc, 0, 0, 1, 1);
        }

        //Now call validate so that these component changes are displayed.
        validate();
    }

    public DataSorter getDataSorter() {
        return dataSorter;
    }

    private void sortLocalData() {
        //dataSorter.setSelectedMetricID(ppTrial.getDefaultMetricID());
        //dataSorter.setValueType(ValueType.EXCLUSIVE);

        //        if (nameCheckBox.isSelected()) {
        //            dataSorter.setSortType(SortType.NAME);
        //        } else {
        //            if (orderByMeanCheckBox.isSelected()) {
        //                dataSorter.setSortType(SortType.MEAN_VALUE);
        //            } else {
        //                dataSorter.setSortType(SortType.MEAN_VALUE);
        //            }
        //        }

        if (dataSorter.getSortType() == SortType.VALUE) {
            dataSorter.setSortType(SortType.MEAN_VALUE);
        }
        dataSorter.setDescendingOrder(orderCheckBox.isSelected());
        model.reloadData();
    }

    public void addNotify() {
        super.addNotify();

        if (visible) {
            return;
        }

        // resize frame to account for menubar
        JMenuBar jMenuBar = getJMenuBar();
        if (jMenuBar != null) {
            int jMenuBarHeight = jMenuBar.getPreferredSize().height;
            Dimension dimension = getSize();
            dimension.height += jMenuBarHeight;
            setSize(dimension);
        }

        visible = true;
    }

    void thisWindowClosing(java.awt.event.WindowEvent e) {
        closeThisWindow();
    }

    public void closeThisWindow() {
        try {
            setVisible(false);

            //if (this != ppTrial.getFullDataWindow()) {
                ppTrial.deleteObserver(this);
                dispose();
            //}
            ParaProf.decrementNumWindows();
        } catch (Exception e) {
            // do nothing
        }
    }

    public void help(boolean display) {
        ParaProf.getHelpWindow().setVisible(true);
    }

    public Function getPhase() {
        return phase;
    }

    public boolean getSortByName() {
        return nameCheckBox.isSelected();
    }

    public void setSortByName(boolean value) {
        nameCheckBox.setSelected(value);
        sortLocalData();
        panel.repaint();
    }

    public boolean getNormalized() {
        return normalizeCheckBox.isSelected();
    }

    public void setNormalized(boolean value) {
        normalizeCheckBox.setSelected(value);
        panel.getBarChart().setNormalized(normalizeCheckBox.isSelected());
        panel.repaint();
    }

    public boolean getSortByMean() {
        return orderByMeanCheckBox.isSelected();
    }

    public void setSortByMean(boolean value) {
        orderByMeanCheckBox.setSelected(value);
        sortLocalData();
        panel.repaint();
    }

    public boolean getDescendingOrder() {
        return orderCheckBox.isSelected();
    }

    public void setDescendingOrder(boolean value) {
        orderCheckBox.setSelected(value);
        sortLocalData();
        panel.repaint();
    }

    public boolean getShowMetaData() {
        return metaDataCheckBox.isSelected();
    }

    public void setShowMetaData(boolean value) {
        metaDataCheckBox.setSelected(value);
        setHeader();
    }

    public boolean getStackBars() {
        return stackBarsCheckBox.isSelected();
    }

    public void setStackBars(boolean value) {
        stackBarsCheckBox.setSelected(value);

        if (value) {
            normalizeCheckBox.setEnabled(true);
            orderByMeanCheckBox.setEnabled(true);

            panel.getBarChart().setNormalized(getNormalized());
            panel.getBarChart().setStacked(true);

        } else {
            normalizeCheckBox.setSelected(false);
            normalizeCheckBox.setEnabled(false);
            orderByMeanCheckBox.setSelected(true);
            orderByMeanCheckBox.setEnabled(false);

            panel.getBarChart().setNormalized(getNormalized());
            panel.getBarChart().setStacked(false);
        }

        sortLocalData();
        panel.repaint();
    }

    public void resort() {
        sortLocalData();
        setHeader();
        panel.repaint();
    }

    public ParaProfTrial getPpTrial() {
        return ppTrial;
    }

    public int units() {
        if (!dataSorter.isTimeMetric()) // we don't do units for non-time metrics
            return 0;
        return ParaProf.preferences.getUnits();
    }
}
