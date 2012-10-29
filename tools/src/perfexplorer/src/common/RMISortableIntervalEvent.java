package edu.uoregon.tau.perfexplorer.common;

import java.io.Serializable;
import java.sql.SQLException;
import java.text.DecimalFormat;
import java.text.FieldPosition;

import edu.uoregon.tau.perfdmf.DatabaseAPI;
import edu.uoregon.tau.perfdmf.Function;
import edu.uoregon.tau.perfdmf.FunctionProfile;


/**
 * This class is the RMI class which contains the tree of views to be 
 * constructed in the PerfExplorerClient.
 *
 * <P>CVS $Id: RMISortableIntervalEvent.java,v 1.3 2009/02/24 00:53:37 khuck Exp $</P>
 * @author khuck
 * @version 0.1
 * @since   0.1
 *
 */
public class RMISortableIntervalEvent implements Serializable, Comparable<RMISortableIntervalEvent> {
	/**
	 * 
	 */
	private static final long serialVersionUID = -4062837640988586241L;
	public int metricIndex;
	private Function function;
	private int trialID;
	private DatabaseAPI dataSession;
	public RMISortableIntervalEvent (Function f, int trialID, DatabaseAPI dataSession, int metricIndex) {
		super();
		this.function = f;
		this.trialID = trialID;
		this.metricIndex = metricIndex;
		this.dataSession = dataSession;
	}
	
	public FunctionProfile getMeanSummary() {
		if (this.function.getMeanProfile() != null)
			return this.function.getMeanProfile();
		else {
			FunctionProfile meanSummary = null;
            try {
				meanSummary = dataSession.getIntervalEventDetail(this.function);
			} catch (SQLException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			this.function.setMeanProfile(meanSummary);
			return meanSummary;
		}
	}

	public Function getFunction() {
		return function;
	}
	
	public int getTrialID() {
		return this.trialID;
	}
	
	public String toString() {
		StringBuffer buf = new StringBuffer();
        try {
			DecimalFormat format = new DecimalFormat("00.00");
			FieldPosition f = new FieldPosition(0);
			format.format(this.function.getMeanProfile().getExclusive(metricIndex), buf, f);
        } catch (Exception exception) {}
		buf.append(" : ");
		buf.append(this.function.getName());
		return buf.toString();
	}

    public int compareTo(RMISortableIntervalEvent e) {
        FunctionProfile ms1 = null;
        FunctionProfile ms2 = null;
        try {
            ms1 = this.getMeanSummary();
            ms2 = e.getMeanSummary();
        } catch (Exception exception) {
            return 0;
        }
		// sort in DESCENDING order!!!!
        if (ms1.getExclusive(metricIndex) < ms2.getExclusive(metricIndex)) {
            return 1;
        } else if (ms1.getExclusive(metricIndex) > ms2.getExclusive(metricIndex)) {
            return -1;
        }else
            return 0;
    }

}
