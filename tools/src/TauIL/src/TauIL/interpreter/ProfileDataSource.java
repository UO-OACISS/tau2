/************************************************************
 *
 *           File : ProfileDataSource.java
 *         Author : Tyrel Datwyler
 *
 *    Description : Gateway between TauIL and ParaProf for
 *                  retrivial of profile data.
 *
 ************************************************************/

package TauIL.interpreter;

import edu.uoregon.tau.dms.dss.Function;
import edu.uoregon.tau.dms.dss.FunctionProfile;
import edu.uoregon.tau.dms.dss.Node;
import edu.uoregon.tau.dms.dss.Context;
import edu.uoregon.tau.dms.dss.Thread;
import edu.uoregon.tau.dms.dss.TauPprofDataSource;
import edu.uoregon.tau.dms.dss.DataSource;

import java.io.File;
import java.util.Iterator;
import java.util.Vector;

class ProfileDataSource extends TauIL.interpreter.DataSource  {

    private Function event;
    private edu.uoregon.tau.dms.dss.DataSource data;

    private Vector files = new Vector();
    private File [] source_file = new File[1];

    private Iterator iterator;

    private boolean time_metric = true;
    private boolean loading = false;

    /* Assume by default that profile data is coming from pprof.dat. */
    protected ProfileDataSource() {
	this("pprof.dat");
    }

    /* Supply file name as source of profile data. */
    protected ProfileDataSource(String fname) {
	setFile(fname);
    }

    /* Set the filename for profile data source. */
    protected void setFile(String fname) {
	if (fname == null || fname.equals(""))
	    source_file[0] = new File("pprof.dat");
	else
	    source_file[0] = new File(fname);
    }

    /* Load profile data into memory. */
    protected void load() {
	files.add(source_file);	

	data = new TauPprofDataSource(files);

	try {
	    data.load();
	} catch (Exception e) {
	    e.printStackTrace();
	    System.exit(-1);
	}
	iterator = data.getFunctions();
    }

    /* The following accessor methods should be self-explanatory. */
    protected boolean isTimeMetric() {
	return time_metric;
    }

    protected boolean hasNext() {
	return iterator.hasNext();
    }

    protected void next() {
	event = (Function) iterator.next();
    }

    protected void reset() {
	iterator = data.getFunctions();
    }

    protected String getEventName() {
	return event.getName();
    }

    protected double getNumCalls() {

	double maxValue = 0;
	for (Iterator it = data.getNodes(); it.hasNext();) {
            Node node = (Node) it.next();
            for (Iterator it2 = node.getContexts(); it2.hasNext();) {
                Context context = (Context) it2.next();
                for (Iterator it3 = context.getThreads(); it3.hasNext();) {
		    edu.uoregon.tau.dms.dss.Thread thread = (edu.uoregon.tau.dms.dss.Thread) it3.next();
                    FunctionProfile functionProfile = thread.getFunctionProfile(event);
                    if (functionProfile != null) {
			maxValue = Math.max(maxValue, functionProfile.getNumCalls());
                    }
                }
            }
        }


	return maxValue;
    }

    protected double getNumSubRS() {
	double maxValue = 0;
	for (Iterator it = data.getNodes(); it.hasNext();) {
            Node node = (Node) it.next();
            for (Iterator it2 = node.getContexts(); it2.hasNext();) {
                Context context = (Context) it2.next();
                for (Iterator it3 = context.getThreads(); it3.hasNext();) {
		    edu.uoregon.tau.dms.dss.Thread thread = (edu.uoregon.tau.dms.dss.Thread) it3.next();
                    FunctionProfile functionProfile = thread.getFunctionProfile(event);
                    if (functionProfile != null) {
			maxValue = Math.max(maxValue, functionProfile.getNumSubr());
                    }
                }
            }
        }


	return maxValue;
    }

    protected double getPercent() {
	return event.getMeanInclusivePercent(0);
    }

    protected double getUsec() {
	return getExclusiveValue();
    }

    protected double getCount() {
	return getExclusiveValue();
    }

    protected double getExclusiveValue() {
	return event.getTotalExclusive(0);
    }

    protected double getCumUsec() {
	return getInclusiveValue();
    }

    protected double getTotCount() {
	return getInclusiveValue();
    }

    protected double getInclusiveValue() {
	return event.getTotalInclusive(0);
    }

    protected double getStdDev() {
	return 0.0;
    }

    protected double getUsecsPerCall() {
	return getPerCall();
    }

    protected double getCountsPerCall() {
	return getPerCall();
    }

    protected double getPerCall() {
	return event.getMeanInclusivePerCall(0);
    }
}
