package edu.uoregon.tau.perfdmf;

public class UserEventProfile {

    private UserEvent userEvent;

    private double data[];
    int numSnapshots;

    //    private double numEvents;
    //    private double minValue;
    //    private double maxValue;
    //    private double meanValue;
    //    private double sumSqr;
    //    private double stdDev;

    private static final int NUM_SAMPLES = 0;
    private static final int MAX = 1;
    private static final int MIN = 2;
    private static final int MEAN = 3;
    private static final int SUM_SQR = 4;
    private static final int STDDEV = 5;

    private static final int SNAPSHOT_SIZE = 6;

    public UserEventProfile(UserEvent userEvent) {
        this(userEvent, 1);
    }

    public UserEventProfile(UserEvent userEvent, int numSnapshots) {
        this.userEvent = userEvent;
        data = new double[SNAPSHOT_SIZE * numSnapshots];
        this.numSnapshots = numSnapshots;
    }

    public UserEvent getUserEvent() {
        return userEvent;
    }

    public void addSnapshot() {
        int newLength = SNAPSHOT_SIZE * (numSnapshots + 1);
        if (newLength > data.length) {
            double[] newArray = new double[(int) (newLength * 1.5)];
            System.arraycopy(data, 0, newArray, 0, data.length);
            data = newArray;
        }
        numSnapshots++;
    }
    
    public String getName() {
        return userEvent.getName();
    }

    public void updateMax() {
        return;
        //
        //        if (numEvents > userEvent.getMaxUserEventNumberValue())
        //            userEvent.setMaxUserEventNumberValue(numEvents);
        //
        //        if (minValue > userEvent.getMaxUserEventMinValue())
        //            userEvent.setMaxUserEventMinValue(minValue);
        //
        //        if (maxValue > userEvent.getMaxUserEventMaxValue())
        //            userEvent.setMaxUserEventMaxValue(maxValue);
        //
        //        if (meanValue > userEvent.getMaxUserEventMeanValue())
        //            userEvent.setMaxUserEventMeanValue(meanValue);
        //
        //        if (sumSqr > userEvent.getMaxUserEventSumSquared())
        //            userEvent.setMaxUserEventSumSquared(sumSqr);
        //
        //        if (stdDev > userEvent.getMaxUserEventStdDev())
        //            userEvent.setMaxUserEventStdDev(stdDev);
    }

    // number of samples
    public void setNumSamples(double numSamples, int snapshot) {
        if (snapshot == -1) {
            snapshot = numSnapshots - 1;
        }
        data[NUM_SAMPLES + (snapshot * SNAPSHOT_SIZE)] = numSamples;
    }

    public void setNumSamples(double numSamples) {
        setNumSamples(numSamples, -1);
    }

    public double getNumSamples() {
        return getNumSamples(-1);
    }

    public double getNumSamples(int snapshot) {
        if (snapshot == -1) {
            snapshot = numSnapshots - 1;
        }
        return data[NUM_SAMPLES + (snapshot * SNAPSHOT_SIZE)];
    }

    // minimum value
    public void setMinValue(double min, int snapshot) {
        if (snapshot == -1) {
            snapshot = numSnapshots - 1;
        }
        data[MIN + (snapshot * SNAPSHOT_SIZE)] = min;
    }

    public void setMinValue(double min) {
        setMinValue(min, -1);
    }

    public double getMinValue() {
        return getMinValue(-1);
    }

    public double getMinValue(int snapshot) {
        if (snapshot == -1) {
            snapshot = numSnapshots - 1;
        }
        return data[MIN + (snapshot * SNAPSHOT_SIZE)];
    }

    // maximum value
    public void setMaxValue(double max, int snapshot) {
        if (snapshot == -1) {
            snapshot = numSnapshots - 1;
        }
        data[MAX + (snapshot * SNAPSHOT_SIZE)] = max;
    }
    public void setMaxValue(double max) {
        setMaxValue(max, -1);
    }

    public double getMaxValue() {
        return getMaxValue(-1);
    }

    public double getMaxValue(int snapshot) {
        if (snapshot == -1) {
            snapshot = numSnapshots - 1;
        }
        return data[MAX + (snapshot * SNAPSHOT_SIZE)];
    }

    // mean value
    public void setMeanValue(double mean, int snapshot) {
        if (snapshot == -1) {
            snapshot = numSnapshots - 1;
        }
        data[MEAN + (snapshot * SNAPSHOT_SIZE)] = mean;
    }
    public void setMeanValue(double mean) {
        setMeanValue(mean, -1);
    }
    

    public double getMeanValue() {
        return getMeanValue(-1);
    }

    public double getMeanValue(int snapshot) {
        if (snapshot == -1) {
            snapshot = numSnapshots - 1;
        }
        return data[MEAN + (snapshot * SNAPSHOT_SIZE)];
    }

    // sum squared
    public void setSumSquared(double inDouble) {
        setSumSquared(inDouble, -1);
    }

    public void setSumSquared(double inDouble, int snapshot) {
        if (snapshot == -1) {
            snapshot = numSnapshots - 1;
        }

        data[SUM_SQR + (snapshot * SNAPSHOT_SIZE)] = inDouble;
        data[STDDEV + (snapshot * SNAPSHOT_SIZE)] = java.lang.Math.sqrt(java.lang.Math.abs((getSumSquared(snapshot) / getNumSamples(snapshot))
                - (getMeanValue(snapshot) * getMeanValue(snapshot))));
    }

    public double getSumSquared() {
        return getSumSquared(-1);
    }

    public double getSumSquared(int snapshot) {
        if (snapshot == -1) {
            snapshot = numSnapshots - 1;
        }
        return data[SUM_SQR + (snapshot * SNAPSHOT_SIZE)];
    }

    // standard deviation

    public void setStdDev(double stddev, int snapshot) {
        if (snapshot == -1) {
            snapshot = numSnapshots - 1;
        }

        data[STDDEV + (snapshot * SNAPSHOT_SIZE)] = stddev;
        data[SUM_SQR + (snapshot * SNAPSHOT_SIZE)] = (stddev * stddev) + (getMeanValue(snapshot) * getMeanValue(snapshot))
                + getNumSamples(snapshot);
    }

    public void setStdDev(double stddev) {
        setStdDev(stddev, -1);
    }

    public double getStdDev() {
        return getStdDev(-1);
    }

    public double getStdDev(int snapshot) {
        if (snapshot == -1) {
            snapshot = numSnapshots - 1;
        }
        return data[STDDEV + (snapshot * SNAPSHOT_SIZE)];
    }

}