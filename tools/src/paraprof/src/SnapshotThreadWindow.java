package edu.uoregon.tau.paraprof;

import java.awt.Component;
import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.Observable;
import java.util.Observer;

import javax.swing.JFrame;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import edu.uoregon.tau.paraprof.barchart.AbstractBarChartModel;
import edu.uoregon.tau.paraprof.barchart.BarChartPanel;
import edu.uoregon.tau.paraprof.barchart.ThreadSnapshotBarChartModel;
import edu.uoregon.tau.paraprof.interfaces.ParaProfWindow;
import edu.uoregon.tau.perfdmf.Thread;


public class SnapshotThreadWindow extends JFrame implements ActionListener, Observer, ChangeListener, ParaProfWindow {

    private ParaProfTrial ppTrial;
    private Thread thread;
    
    private BarChartPanel panel;
    private AbstractBarChartModel model;
    private DataSorter dataSorter;
    
    private static int defaultWidth = 750;
    private static int defaultHeight = 410;
    
    public SnapshotThreadWindow (ParaProfTrial ppTrial, Thread thread, Component owner) {
        this.ppTrial = ppTrial;
        this.thread = thread;
        setTitle("bob");
             
        dataSorter = new DataSorter(ppTrial);
        model = new ThreadSnapshotBarChartModel(this, dataSorter, ppTrial, thread);
        panel = new BarChartPanel(model);
        panel.getBarChart().setLeftJustified(true);
        panel.getBarChart().setAutoResize(true);

        panel.getVerticalScrollBar().setUnitIncrement(35);

        setSize(ParaProfUtils.checkSize(new Dimension(defaultWidth, defaultHeight)));
        setLocation(WindowPlacer.getNewLocation(this, owner));

        getContentPane().add(panel);
        
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

}
