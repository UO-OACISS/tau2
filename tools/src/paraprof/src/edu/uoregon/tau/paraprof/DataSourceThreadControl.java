package edu.uoregon.tau.paraprof;

import edu.uoregon.tau.dms.dss.*;

import java.awt.*;
import java.util.*;

public class DataSourceThreadControl implements Runnable {
    public DataSourceThreadControl() {
        super();
    }


    public void initialize(DataSource dataSource, boolean graphicsEnvironment) {
        this.dataSource = dataSource;
        this.graphicsEnvironment = graphicsEnvironment;
        java.lang.Thread thread = new java.lang.Thread(this);
        thread.start();
    }

    public void run() {

        try {
            this.dataSource.load();
            loadedOk = true;
        } catch (Exception e) {
            System.err.println ("Error loading trial:");
            e.printStackTrace();
            exception = e;
        }
        
        if (graphicsEnvironment) {
            //Need to notify observers that we are done. Be careful here.
            //It is likely that they will modify swing elements. Make sure
            //to dump request onto the event dispatch thread to ensure
            //safe update of said swing elements. Remember, swing is not thread
            //safe for the most part.
            EventQueue.invokeLater(new Runnable() {
                public void run() {
                    DataSourceThreadControl.this.notifyObservers();
                }
            });
        } else
            this.notifyObservers();
    }

    public void addObserver(ParaProfObserver observer) {
        observers.add(observer);
    }

    public void removeObserver(ParaProfObserver observer) {
        observers.remove(observer);
    }

    public void notifyObservers() {
        if (this.debug()) {
            System.out.println("######");
            System.out.println("ParaProfDataSource.notifyObservers()");
            System.out.println("Listening classes ...");
            for (Enumeration e = observers.elements(); e.hasMoreElements();)
                System.out.println(e.nextElement().getClass());
            System.out.println("######");
        }
 
        if (loadedOk) {
            for (Enumeration e = observers.elements(); e.hasMoreElements();)
                ((ParaProfObserver) e.nextElement()).update(dataSource);
        } else {
            for (Enumeration e = observers.elements(); e.hasMoreElements();)
                ((ParaProfObserver) e.nextElement()).update(exception);
        }
    }

    public void setDebug(boolean debug) {
        this.debug = debug;
    }

    public boolean debug() {
        return debug;
    }

    //####################################
    //Private Section.
    //####################################
    private DataSource dataSource = null;
    private boolean graphicsEnvironment = false;

    private Vector observers = new Vector();
    private boolean debug = false;
    private boolean loadedOk = false;
    
    private Exception exception = null;
    
    //####################################
    //End - Private Section.
    //####################################
}