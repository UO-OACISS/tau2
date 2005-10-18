package edu.uoregon.tau.perfdmf;

import java.util.*;

public class UserEventProfile {

    private UserEvent userEvent = null;
    private double numEvents;
    private double minValue;
    private double maxValue;
    private double meanValue;
    private double sumSqr;
    private double stdDev;

    
    public UserEventProfile(UserEvent userEvent) {
       this.userEvent = userEvent;
    }

    public UserEvent getUserEvent() {
        return userEvent;
    }

    
    public void updateMax() {
        
        if (numEvents > userEvent.getMaxUserEventNumberValue())
            userEvent.setMaxUserEventNumberValue(numEvents);

        if (minValue > userEvent.getMaxUserEventMinValue())
            userEvent.setMaxUserEventMinValue(minValue);
        
        if (maxValue > userEvent.getMaxUserEventMaxValue())
            userEvent.setMaxUserEventMaxValue(maxValue);
        
        if (meanValue > userEvent.getMaxUserEventMeanValue())
            userEvent.setMaxUserEventMeanValue(meanValue);
        
        if (sumSqr > userEvent.getMaxUserEventSumSquared())
            userEvent.setMaxUserEventSumSquared(sumSqr);
        
        if (stdDev > userEvent.getMaxUserEventStdDev())
            userEvent.setMaxUserEventStdDev(stdDev);
    }
    
    public void setNumSamples(double inInt) {
        numEvents = inInt;
    }

    public double getNumSamples() {
        return numEvents;
    }

    public void setMinValue(double inDouble) {
        minValue = inDouble;
    }

    public double getMinValue() {
        return minValue;
    }

    public void setMaxValue(double inDouble) {
        maxValue = inDouble;
    }

    public double getMaxValue() {
        return maxValue;
    }

    public void setMeanValue(double inDouble) {
        meanValue = inDouble;
    }

    public double getMeanValue() {
        return meanValue;
    }

    public void setSumSquared(double inDouble) {
        sumSqr = inDouble;

        stdDev = java.lang.Math.sqrt(java.lang.Math.abs((sumSqr / numEvents)
                - (meanValue * meanValue)));
    
    }

    public double getSumSquared() {
        return sumSqr;
    }

    public double getStdDev() {
        return stdDev;
    }
    
    public void setStdDev(double stdDev) {
        this.stdDev = stdDev;
        
        this.sumSqr = (stdDev * stdDev) + (meanValue * meanValue) + numEvents;
        
    }
    
    

  
}