package dms.dss;

/**
 * Holds all the data for a function data object in the database.
 * This object is returned by the DataSession class and all of its subtypes.
 * The FunctionData object contains all the information associated with
 * an function location instance from which the TAU performance data has been generated.
 * A function location is associated with one node, context, thread, function, trial, 
 * experiment and application, and has data for all selected metrics in it.
 * <p>
 * A FunctionData object has information
 * related to one particular function location in the trial, including the ID of the function,
 * the node, context and thread that identify the location, and the data collected for this
 * location, such as inclusive time, exclusive time, etc.  If there are multiple metrics recorded
 * in the trial, and no metric filter is applied when the FunctionData object is requested, then
 * all metric data for this location will be returned.  The index of the metric needs to be
 * passed in to get data for a particular metric.  If there is only one metric, then no metric
 * index need be passed in.
 *
 * <P>CVS $Id: FunctionDataObject.java,v 1.14 2003/08/27 17:07:38 khuck Exp $</P>
 * @author	Kevin Huck, Robert Bell
 * @version	0.1
 * @since	0.1
 * @see		DataSession#getFunctionData
 * @see		DataSession#setFunction
 * @see		DataSession#setNode
 * @see		DataSession#setContext
 * @see		DataSession#setThread
 * @see		DataSession#setMetric
 * @see		Function
 */
public class FunctionDataObject {
	private int node;
	private int context;
	private int thread;
	private int functionID;
	private double[] doubleList;
	private int numCalls;
	private int numSubroutines;
	private static int fieldCount = 5;

/**
 * Base constructor.
 *
 */
	public FunctionDataObject() {
		super();
		doubleList = new double[fieldCount];
	}

/**
 * Alternative constructor.
 *
 * @param metricCount specifies how many metrics are expected for this trial.
 */
	public FunctionDataObject(int metricCount) {
		super();
		int trueSize = metricCount * fieldCount;
		doubleList = new double[trueSize];
	}

/**
 * Returns the node for this data location.
 *
 * @return the node index.
 */
	public int getNode () {
		return this.node;
	}

/**
 * Returns the context for this data location.
 *
 * @return the context index.
 */
	public int getContext () {
		return this.context;
	}

/**
 * Returns the thread for this data location.
 *
 * @return the thread index.
 */
	public int getThread () {
		return this.thread;
	}

/**
 * Returns the unique ID for the function that owns this data
 *
 * @return the functionID.
 * @see		Function
 */
	public int getFunctionIndexID () {
		return this.functionID;
	}

/**
 * Returns the inclusive percentage value for the specified metric at this location.
 *
 * @param	metricIndex the index of the metric desired.
 * @return	the inclusive percentage.
 */
	public double getInclusivePercentage (int metricIndex) {
		return getDouble(metricIndex, 0);
	}

/**
 * Returns the inclusive percentage value for the first (or only) metric at this location.
 *
 * @return	the inclusive percentage.
 */
	public double getInclusivePercentage () {
		return getDouble(0, 0);
	}

/**
 * Returns the inclusive value for the specified metric at this location.
 *
 * @param	metricIndex the index of the metric desired.
 * @return	the inclusive percentage.
 */
	public double getInclusive (int metricIndex) {
		return getDouble(metricIndex, 1);
	}

/**
 * Returns the inclusive value for the first (or only) metric at this location.
 *
 * @return	the inclusive percentage.
 */
	public double getInclusive () {
		return getDouble(0, 1);
	}

/**
 * Returns the exclusive percentage value for the specified metric at this location.
 *
 * @param	metricIndex the index of the metric desired.
 * @return	the exclusive percentage.
 */
	public double getExclusivePercentage (int metricIndex) {
		return getDouble(metricIndex, 2);
	}

/**
 * Returns the exclusive percentage value for the first (or only) metric at this location.
 *
 * @return	the exclusive percentage.
 */
	public double getExclusivePercentage () {
		return getDouble(0, 2);
	}

/**
 * Returns the exclusive value for the specified metric at this location.
 *
 * @param	metricIndex the index of the metric desired.
 * @return	the exclusive percentage.
 */
	public double getExclusive (int metricIndex) {
		return getDouble(metricIndex, 3);
	}

/**
 * Returns the exclusive value for the first (or only) metric at this location.
 *
 * @return	the exclusive percentage.
 */
	public double getExclusive () {
		return getDouble(0, 3);
	}

/**
 * Returns the inclusive value per call for the specified metric at this location.
 *
 * @param	metricIndex the index of the metric desired.
 * @return	the inclusive percentage.
 */
	public double getInclusivePerCall (int metricIndex) {
		return getDouble(metricIndex, 4);
	}

/**
 * Returns the inclusive per call value for the first (or only) metric at this location.
 *
 * @return	the inclusive percentage.
 */
	public double getInclusivePerCall () {
		return getDouble(0, 4);
	}

/**
 * Returns the number of calls to this function at this location.
 *
 * @return	the number of calls.
 */
	public int getNumCalls () {
		return this.numCalls;
	}

/**
 * Returns the number of subroutines for this function at this location.
 *
 * @return	the number of subroutines.
 */
	public int getNumSubroutines () {
		return this.numSubroutines;
	}

	private void incrementStorage(){
		int currentLength = doubleList.length;
		double[] newArray = new double[currentLength+fieldCount];
		for(int i=0;i<currentLength;i++){
			newArray[i] = doubleList[i];
		}
		doubleList = newArray;
	}

    private void insertDouble(int dataValueLocation, int offset, double inDouble){
		int actualLocation = (dataValueLocation*fieldCount)+offset;
		if (actualLocation > doubleList.length)
			incrementStorage();
		try{
			doubleList[actualLocation] = inDouble;
		} catch(Exception e){
			// do something
		}
	}

    private double getDouble(int dataValueLocation, int offset){
		int actualLocation = (dataValueLocation*fieldCount)+offset;
		try{
			return doubleList[actualLocation];
		} catch(Exception e){
			// do something
		}
		return -1;
	}

/**
 * Sets the node of the current location that this data object represents.
 * <i> NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	node the node for this location.
 */
	public void setNode (int node) {
		this.node = node;
	}

/**
 * Sets the context of the current location that this data object represents.
 * <i> NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	context the context for this location.
 */
	public void setContext (int context) {
		this.context = context;
	}

/**
 * Sets the thread of the current location that this data object represents.
 * <i> NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	thread the thread for this location.
 */
	public void setThread (int thread) {
		this.thread = thread;
	}

/**
 * Sets the unique function ID for the function at this location.
 * <i> NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	functionID a unique function ID.
 */
	public void setFunctionIndexID (int functionID) {
		this.functionID = functionID;
	}

/**
 * Sets the inclusive percentage value for the specified metric at this location.
 * <i> NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	metricIndex the index of the metric
 * @param	inclusivePercentage the inclusive percentage value at this location
 */
	public void setInclusivePercentage (int metricIndex, double inclusivePercentage) {
		insertDouble(metricIndex, 0, inclusivePercentage);
	}

/**
 * Sets the inclusive value for the specified metric at this location.
 * <i> NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	metricIndex the index of the metric
 * @param	inclusive the inclusive value at this location
 */
	public void setInclusive (int metricIndex, double inclusive) {
		insertDouble(metricIndex, 1, inclusive);
	}

/**
 * Sets the exclusive percentage value for the specified metric at this location.
 * <i> NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	metricIndex the index of the metric
 * @param	exclusivePercentage the exclusive percentage value at this location
 */
	public void setExclusivePercentage (int metricIndex, double exclusivePercentage) {
		insertDouble(metricIndex, 2, exclusivePercentage);
	}

/**
 * Sets the exclusive value for the specified metric at this location.
 * <i> NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	metricIndex the index of the metric
 * @param	exclusive the exclusive value at this location
 */
	public void setExclusive (int metricIndex, double exclusive) {
		insertDouble(metricIndex, 3, exclusive);
	}

/**
 * Sets the inclusive per call value for the specified metric at this location.
 * <i> NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	metricIndex the index of the metric
 * @param	inclusivePerCall the inclusive per call value at this location
 */
	public void setInclusivePerCall (int metricIndex, double inclusivePerCall) {
		insertDouble(metricIndex, 4, inclusivePerCall);
	}

/**
 * Sets the number of times that the function was called at this location.
 * <i> NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	numCalls the number of times the function was called
 */
	public void setNumCalls (int numCalls) {
		this.numCalls = numCalls;
	}

/**
 * Sets the number of subroutines the function has at this location.
 * <i> NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	numSubroutines the number of subroutines the function has at this location.
 */
	public void setNumSubroutines (int numSubroutines) {
		this.numSubroutines = numSubroutines;
	}

}

