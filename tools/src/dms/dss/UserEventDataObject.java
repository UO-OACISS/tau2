package dms.dss;

public class UserEventDataObject extends DataObject{
    private int userEventID;
    private double numOfSamples;
    private double max;
    private double min;
    private double mean;
    private double stdev;

    public void setUserEventIndexID (int userEventID) {
	this.userEventID = userEventID;
    }

    public void setNumOfSamples (double numOfSamples) {
	this.numOfSamples = numOfSamples;
    }

    public void setMax (double max) {
	this.max = max;
    }

    public void setMin (double min) {
	this.min = min;
    }

    public void setMin (double min) {
	this.min = min;
    }

    public void setMean (double mean) {
	this.mean = mean;
    }

    public void setStdev (double stdev) {
	this.stdev = stdev;
    }

    public int userEventIndexID () {
	return this.userEventID;
    }

    public double getNumOfSamples () {
	return this.NumOfSamples;
    }

    public double getMax () {
	return this.max;
    }

    public double getMin () {
	return this.min;
    }
    
    public double getMean () {
	return this.mean;
    }

    public double getStdev () {
	return this.stdev;
    }
}


