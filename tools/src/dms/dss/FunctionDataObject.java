package dms.dss;

/**
 * Holds all the data for a function data object in the database.
 *
 * <P>CVS $Id: FunctionDataObject.java,v 1.12 2003/08/12 00:08:29 khuck Exp $</P>
 * @author	Kevin Huck, Robert Bell
 * @version	%I%, %G%
 */
public class FunctionDataObject {
	private int node;
	private int context;
	private int thread;
	private int functionID;
	private double[] doubleList;
	private int numCalls;
	private int numSubroutines;

	public FunctionDataObject() {
		super();
		doubleList = new double[5];
	}

	public FunctionDataObject(int arraySize) {
		super();
		doubleList = new double[arraySize];
	}

	public void incrementStorage(){
		int currentLength = doubleList.length;
		double[] newArray = new double[currentLength+14];
		for(int i=0;i<currentLength;i++){
			newArray[i] = doubleList[i];
		}
		doubleList = newArray;
	}

    private void insertDouble(int dataValueLocation, int offset, double inDouble){
		int actualLocation = (dataValueLocation*5)+offset;
		try{
			doubleList[actualLocation] = inDouble;
		} catch(Exception e){
			// do something
		}
	}

    private double getDouble(int dataValueLocation, int offset){
		int actualLocation = (dataValueLocation*5)+offset;
		try{
			return doubleList[actualLocation];
		} catch(Exception e){
			// do something
		}
		return -1;
	}

	public void setNode (int node) {
		this.node = node;
	}

	public void setContext (int context) {
		this.context = context;
	}

	public void setThread (int thread) {
		this.thread = thread;
	}

	public void setFunctionIndexID (int functionID) {
		this.functionID = functionID;
	}

	public void setInclusivePercentage (int metricIndex, double inclusivePercentage) {
		insertDouble(metricIndex, 0, inclusivePercentage);
	}

	public void setInclusive (int metricIndex, double inclusive) {
		System.out.println ("setInclusive: " + metricIndex + ", " + inclusive);
		insertDouble(metricIndex, 1, inclusive);
	}

	public void setExclusivePercentage (int metricIndex, double exclusivePercentage) {
		insertDouble(metricIndex, 2, exclusivePercentage);
	}

	public void setExclusive (int metricIndex, double exclusive) {
		insertDouble(metricIndex, 3, exclusive);
	}

	public void setInclusivePerCall (int metricIndex, double inclusivePerCall) {
		insertDouble(metricIndex, 4, inclusivePerCall);
	}

	public void setNumCalls (int numCalls) {
		this.numCalls = numCalls;
	}

	public void setNumSubroutines (int numSubroutines) {
		this.numSubroutines = numSubroutines;
	}

	public int getNode () {
		return this.node;
	}

	public int getContext () {
		return this.context;
	}

	public int getThread () {
		return this.thread;
	}

	public int getFunctionIndexID () {
		return this.functionID;
	}

	public double getInclusivePercentage (int metricIndex) {
		return getDouble(metricIndex, 0);
	}

	public double getInclusivePercentage () {
		return getDouble(0, 0);
	}

	public double getInclusive (int metricIndex) {
		return getDouble(metricIndex, 1);
	}

	public double getInclusive () {
		return getDouble(0, 1);
	}

	public double getExclusivePercentage (int metricIndex) {
		return getDouble(metricIndex, 2);
	}

	public double getExclusivePercentage () {
		return getDouble(0, 2);
	}

	public double getExclusive (int metricIndex) {
		return getDouble(metricIndex, 3);
	}

	public double getExclusive () {
		return getDouble(0, 3);
	}

	public double getInclusivePerCall (int metricIndex) {
		return getDouble(metricIndex, 4);
	}

	public double getInclusivePerCall () {
		return getDouble(0, 4);
	}

	public int getNumCalls () {
		return this.numCalls;
	}

	public int getNumSubroutines () {
		return this.numSubroutines;
	}

}

