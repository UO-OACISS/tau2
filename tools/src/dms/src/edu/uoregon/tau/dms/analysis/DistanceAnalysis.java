package edu.uoregon.tau.dms.analysis;

import edu.uoregon.tau.dms.dss.*;
import edu.uoregon.tau.dms.database.*;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.Vector;
import java.util.ListIterator;

/**
 * This is the top level class for doing distance analysis.
 * This code is largely based on the work of Sherwood, Perelman, Hamerly
 * and Calder, in their paper "Automatically Characterizing Large Scale
 * Program Behavior", 2002.  In that paper, they describe the algorithm for
 * calculating distance metrics between phases of a serial executable.  In
 * this implementation, the phases are replaced by threads or events.
 * Instead of comparing blocks of code to find phases of execution, the user
 * can compare either threads or events.  See the referenced paper for more
 * details.
 *
 * <P>CVS $Id: DistanceAnalysis.java,v 1.3 2004/07/29 23:17:09 khuck Exp $</P>
 * @author	Kevin Huck
 * @version	0.1
 * @since	0.1
 */
abstract public class DistanceAnalysis {
	protected DB db = null;
	protected DistanceMatrix results = null;
	protected Trial trial = null;
	protected Metric metric = null;

/**
 * Basic constructor for the DistanceAnalysis object.
 *
 * @param	session	a reference to a PerfDMFSession database session object.
 * @param	inTrial	a reference to a PerfDMF Trial object of interest.
 * @param	inMetric	a reference to a PerfDMF Metric object of interest.
 */
	public DistanceAnalysis (PerfDMFSession session, Trial inTrial, Metric inMetric) {
		this.db = session.db();
		this.trial = inTrial;
		this.metric = inMetric;
	}

	private void getRawData() {
		// calculate the threadCount;
		int threadCount = trial.getNodeCount() * trial.getNumContextsPerNode() * trial.getNumThreadsPerContext();

		// get the event count from the database
		StringBuffer buf = new StringBuffer();
		buf.append("select count(id) from interval_event where trial = ");
		buf.append(trial.getID());
		int eventCount = Integer.parseInt(db.getDataItem(buf.toString()));
		System.out.println(threadCount + " " + eventCount);

		// initialize the matrix
		results = createTheMatrix(threadCount, eventCount);

		// get the event names from the database
		buf = new StringBuffer();
		buf.append("select name from interval_event where trial = ? order by id");
		try {
			PreparedStatement statement = db.prepareStatement(buf.toString());
			statement.setInt(1, trial.getID());
			ResultSet resultSet = statement.executeQuery();          
			int i = 0;
			while (resultSet.next() != false) {
				results.eventName[i++] = resultSet.getString(1);
				// System.out.println(results.eventName[i-1]);
        	}
        	resultSet.close(); 
		} catch (SQLException e) {
			e.printStackTrace();
			return;
		}

		// build the query to get total amounts
		buf = new StringBuffer();
		buf.append("select l.node, l.context, l.thread, sum(l.exclusive) ");
		buf.append("from interval_location_profile l ");
		buf.append("inner join interval_event e ");
		buf.append("on l.interval_event = e.id ");
		buf.append("where e.trial = ? and l.metric = ? ");
		buf.append("group by l.node, l.context, l.thread ");
		buf.append("order by l.node, l.context, l.thread ");

		// get the totals
		getTotals();

		// get the raw data
		getMatrixData();
	}

/**
 * This method gets the total timer/counter vector across all dimensions.
 * This method should be implemented for all class extensions that wish to
 * build a distance matrix.  In the case of an EventDistance object, this
 * method gets the total amount that each event spent on all threads.  In the
 * case of a ThreadDistance object, this method gets the total amount that
 * each thread spent in all events.
 *
 */
	abstract protected void getTotals();

/**
 * This method gets the relative timer/counter value matrix for all threads,
 * all events.
 * This method should be implemented for all class extensions that wish to
 * build a distance matrix.  In the case of an EventDistance object, this
 * method gets the amount that each event spent on each thread.  In the
 * case of a ThreadDistance object, this method gets the amount that
 * each thread spent in each event.
 *
 */
	abstract protected void getMatrixData();

/**
 * This method creates the appropriate DistanceMatrix for the analysis.
 * This method should be implemented by any classes that wish to build a
 * distance matrix.  In the case of a ThreadDistance object, this method
 * creates a threadCount X eventCount matrix.  In the case of an EventDistance
 * object, this method creates an eventCount X threadCount matrix.
 *
 * @param	threadCount the number of threads in the parallel execution.
 * @param	eventCount the number of events in the parallel execution.
 * @return	a reference to a DistanceMatrix object.
 */
	abstract protected DistanceMatrix createTheMatrix(int threadCount, int eventCount);
	
/**
 * This method calculates the Manhattan distance.
 * Each row in the relative data matrix is compared to every other row in
 * the relative data matrix, using a Manhattan distance calculation.
 * If there are NxE values in the relative data matrix, this results in 
 * an NxN distance matrix.  Values will be in the range 0.0 to 2.0.
 *
 * @return	a two-dimensional array which stores the relative distances 
 */
	public double[][] getManhattanDistance() {
		if (results == null) getRawData();
		results.getManhattanDistance();
		return results.distanceMatrix;
	}

/**
 * This method calculates the Euclidean distance.
 * Each row in the relative data matrix is compared to every other row in
 * the relative data matrix, using a Euclidian distance calculation. 
 * If there are NxE values in the relative data matrix, this results in 
 * an NxN distance matrix.  Values will be in the range 0.0 to 1.4-ish.
 *
 * @return	a two-dimensional array which stores the relative distances 
 */
	public double[][] getEuclideanDistance() {
		if (results == null) getRawData();
		results.getEuclideanDistance();
		return results.distanceMatrix;
	}

/**
 * This method dumps the data in the distance matrix to a string.
 * This is a debug method used to check the output from the distance calculation.
 *
 * @return	a String with the distances from the distance matrix.
 */
	public String toString() { 
		return results == null ? new String("") : results.toString(); 
	}

/**
 * This method dumps the data in the distance matrix to an image.
 * This is a method used to check the output from the distance calculation.
 *
 * @return	an array of image data.
 */
	public int[] toImage(boolean scaledRange, boolean triangle) { 
		if (results == null)
			return new int[0];
		else
			return results.toImage(scaledRange, triangle); 
	}
}

