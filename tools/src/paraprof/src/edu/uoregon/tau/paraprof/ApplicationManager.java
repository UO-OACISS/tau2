package edu.uoregon.tau.paraprof;

import java.util.*;
import edu.uoregon.tau.dms.dss.*;

/**
 * ApplicationManager
 * This controls the adding of applications to the system.
 *  
 * 
 * <P>CVS $Id: ApplicationManager.java,v 1.5 2005/01/19 02:33:25 amorris Exp $</P>
 * @author	Robert Bell
 * @version	$Revision: 1.5 $
 */
public class ApplicationManager extends Observable {

    private Vector applications = new Vector();

    public ParaProfApplication addApplication() {
        ParaProfApplication application = new ParaProfApplication();
        application.setID((applications.size()));
        applications.add(application);
        return application;
    }

    public void removeApplication(ParaProfApplication application) {
        applications.remove(application);
    }

    public Vector getApplications() {
        return applications;
    }

    public DssIterator getApplicationList() {
        return new DssIterator(applications);
    }

    public boolean isEmpty() {
        if (applications.size() == 0)
            return true;
        else
            return false;
    }

    public ParaProfApplication getApplication(int applicationID) {
        return (ParaProfApplication) applications.elementAt(applicationID);
    }

    public ParaProfExperiment getExperiment(int applicationID, int experimentID) {
        return (this.getApplication(applicationID)).getExperiment(experimentID);
    }

    public ParaProfTrial getTrial(int applicationID, int experimentID, int trialID) {
        return ((this.getApplication(applicationID)).getExperiment(experimentID)).getTrial(trialID);
    }

    public boolean isApplicationPresent(String name) {
        for (Enumeration e = applications.elements(); e.hasMoreElements();) {
            ParaProfApplication application = (ParaProfApplication) e.nextElement();
            if (name.compareTo(application.getName()) == 0)
                return true;
        }
        return false;
    }
}