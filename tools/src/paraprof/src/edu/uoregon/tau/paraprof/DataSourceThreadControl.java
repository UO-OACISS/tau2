package edu.uoregon.tau.paraprof;

import edu.uoregon.tau.dms.dss.*;

import java.awt.*;
import java.util.*;

/**
 * DataSourceThreadControl.java
 * This object runs a datasource's load method, then notify's the observers when done 
 *  
 * 
 * <P>CVS $Id: DataSourceThreadControl.java,v 1.3 2005/01/04 01:16:26 amorris Exp $</P>
 * @author	Robert Bell, Alan Morris
 * @version	$Revision: 1.3 $
 */
public class DataSourceThreadControl implements Runnable {
    public DataSourceThreadControl() {
        super();
    }

    public void initialize(DataSource dataSource) {
        this.dataSource = dataSource;
        java.lang.Thread thread = new java.lang.Thread(this);
        thread.start();
    }

    public void run() {

        try {
            this.dataSource.load();
            loadedOk = true;
        } catch (Exception e) {
            exception = new DataSourceException(e);
        }

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
    }

    public void addObserver(ParaProfObserver observer) {
        observers.add(observer);
    }

    public void removeObserver(ParaProfObserver observer) {
        observers.remove(observer);
    }

    public void notifyObservers() {
        try {
            if (loadedOk) {
                for (Enumeration e = observers.elements(); e.hasMoreElements();)
                    ((ParaProfObserver) e.nextElement()).update(dataSource);
            } else {
                for (Enumeration e = observers.elements(); e.hasMoreElements();)
                    ((ParaProfObserver) e.nextElement()).update(exception);
            }
        } catch (DatabaseException e) {
            ParaProfUtils.handleException(e);
        }
    }

    private DataSource dataSource = null;
    private Vector observers = new Vector();
    private boolean loadedOk = false;
    private Exception exception = null;

}