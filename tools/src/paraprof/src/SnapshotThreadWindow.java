package edu.uoregon.tau.paraprof;

import java.awt.Component;
import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.Observable;
import java.util.Observer;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import edu.uoregon.tau.paraprof.barchart.AbstractBarChartModel;
import edu.uoregon.tau.paraprof.barchart.BarChartPanel;
import edu.uoregon.tau.paraprof.barchart.ThreadSnapshotBarChartModel;
import edu.uoregon.tau.paraprof.interfaces.ParaProfWindow;
import edu.uoregon.tau.paraprof.util.ObjectFilter;
import edu.uoregon.tau.perfdmf.Thread;

public class SnapshotThreadWindow extends JFrame implements ActionListener, Observer, ChangeListener, ParaProfWindow {

    private ParaProfTrial ppTrial;
    private Thread thread;

    private BarChartPanel panel;
    private AbstractBarChartModel model;
    private DataSorter dataSorter;

    private static int defaultWidth = 750;
    private static int defaultHeight = 410;

    private JMenu optionsMenu;
    private JCheckBoxMenuItem nameCheckBox = new JCheckBoxMenuItem("Sort By Name", false);
    private JCheckBoxMenuItem normalizeCheckBox = new JCheckBoxMenuItem("Normalize Bars", true);
    private JCheckBoxMenuItem orderByMeanCheckBox = new JCheckBoxMenuItem("Order By Mean", true);
    private JCheckBoxMenuItem orderCheckBox = new JCheckBoxMenuItem("Descending Order", true);
    private JCheckBoxMenuItem stackBarsCheckBox = new JCheckBoxMenuItem("Stack Bars Together", true);
    private JCheckBoxMenuItem metaDataCheckBox;

    public SnapshotThreadWindow(ParaProfTrial ppTrial, Thread thread, Component owner) {
        this.ppTrial = ppTrial;
        this.thread = thread;

        PPThread ppThread = new PPThread(thread, ppTrial);

        this.setTitle("TAU: ParaProf: Snapshots for " + ppThread.getFullName() + " - "
                + ppTrial.getTrialIdentifier(ParaProf.preferences.getShowPathTitleInReverse()));

        dataSorter = new DataSorter(ppTrial);
        model = new ThreadSnapshotBarChartModel(this, dataSorter, ppTrial, thread);
        panel = new BarChartPanel(model);
        panel.getBarChart().setLeftJustified(true);
        panel.getBarChart().setAutoResize(true);

        panel.getVerticalScrollBar().setUnitIncrement(35);

        setSize(ParaProfUtils.checkSize(new Dimension(defaultWidth, defaultHeight)));
        setLocation(WindowPlacer.getNewLocation(this, owner));

        getContentPane().add(panel);

        setupMenus();

        ParaProfUtils.setFrameIcon(this);
    }

    private void setupMenus() {
        JMenuBar mainMenu = new JMenuBar();

        optionsMenu = new JMenu("Options");

        metaDataCheckBox = new JCheckBoxMenuItem("Show Meta Data in Panel", true);
        metaDataCheckBox.addActionListener(this);
        optionsMenu.add(metaDataCheckBox);

        optionsMenu.add(new JSeparator());

        nameCheckBox.addActionListener(this);
        optionsMenu.add(nameCheckBox);

        normalizeCheckBox.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                setNormalized(normalizeCheckBox.isSelected());
            }
        });
        optionsMenu.add(normalizeCheckBox);

        orderByMeanCheckBox.addActionListener(this);
        optionsMenu.add(orderByMeanCheckBox);

        orderCheckBox.addActionListener(this);
        optionsMenu.add(orderCheckBox);

        stackBarsCheckBox.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                setStackBars(stackBarsCheckBox.isSelected());
            }
        });
        optionsMenu.add(stackBarsCheckBox);

        mainMenu.add(ParaProfUtils.createFileMenu(this, panel, panel));
        mainMenu.add(optionsMenu);
        mainMenu.add(ParaProfUtils.createWindowsMenu(ppTrial, this));
        if (ParaProf.scripts.size() > 0) {
            mainMenu.add(ParaProfUtils.createScriptMenu(ppTrial, this));
        }
        mainMenu.add(ParaProfUtils.createHelpMenu(this, this));

        setJMenuBar(mainMenu);

    }

    public void actionPerformed(ActionEvent e) {
    // TODO Auto-generated method stub

    }

    public void update(Observable o, Object arg) {
    // TODO Auto-generated method stub

    }

    public void stateChanged(ChangeEvent e) {
    // TODO Auto-generated method stub

    }

    public void closeThisWindow() {
    // TODO Auto-generated method stub

    }

    public void help(boolean display) {
    // TODO Auto-generated method stub

    }

    public boolean getNormalized() {
        return normalizeCheckBox.isSelected();
    }

    public void setNormalized(boolean value) {
        normalizeCheckBox.setSelected(value);
        panel.getBarChart().setNormalized(normalizeCheckBox.isSelected());
        panel.repaint();
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

        panel.repaint();
    }

    public JFrame getFrame() {
        return this;
    }
}
