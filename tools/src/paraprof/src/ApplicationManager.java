package edu.uoregon.tau.paraprof;

import java.util.*;

/**
 * ApplicationManager
 * This controls the adding of applications to the system.
 *  
 * 
 * <P>CVS $Id: ApplicationManager.java,v 1.16 2007/05/29 20:27:20 amorris Exp $</P>
 * @author	Robert Bell
 * @version	$Revision: 1.16 $
 */
public class ApplicationManager extends Observable {

    private List applications = new ArrayList();

    public ParaProfApplication addApplication() {
        ParaProfApplication application = new ParaProfApplication();
        application.setID((applications.size()));
        applications.add(application);
        return application;
    }

    public void removeApplication(ParaProfApplication application) {
        applications.remove(application);
    }

    public List getApplications() {
        return applications;
    }

    public boolean isEmpty() {
        if (applications.size() == 0)
            return true;
        else
            return false;
    }

    public ParaProfApplication getApplication(int applicationID) {
        return (ParaProfApplication) applications.get(applicationID);
    }

    public ParaProfExperiment getExperiment(int applicationID, int experimentID) {
        return (this.getApplication(applicationID)).getExperiment(experimentID);
    }

    public ParaProfTrial getTrial(int applicationID, int experimentID, int trialID) {
        return ((this.getApplication(applicationID)).getExperiment(experimentID)).getTrial(trialID);
    }

    public boolean isApplicationPresent(String name) {
        for (Iterator it = applications.iterator(); it.hasNext();) {
            ParaProfApplication application = (ParaProfApplication) it.next();
            if (name.compareTo(application.getName()) == 0)
                return true;
        }
        return false;
    }
}