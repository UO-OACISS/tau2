package edu.uoregon.tau.perfdmf;

import java.util.*;

public class UserEventProfile {

    private UserEvent userEvent = null;
    private int numEvents;
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
    
    public void setUserEventNumberValue(int inInt) {
        numEvents = inInt;
    }

    public int getUserEventNumberValue() {
        return numEvents;
    }

    public void setUserEventMinValue(double inDouble) {
        minValue = inDouble;
    }

    public double getUserEventMinValue() {
        return minValue;
    }

    public void setUserEventMaxValue(double inDouble) {
        maxValue = inDouble;
    }

    public double getUserEventMaxValue() {
        return maxValue;
    }

    public void setUserEventMeanValue(double inDouble) {
        meanValue = inDouble;
    }

    public double getUserEventMeanValue() {
        return meanValue;
    }

    public void setUserEventSumSquared(double inDouble) {
        sumSqr = inDouble;

        stdDev = java.lang.Math.sqrt(java.lang.Math.abs((sumSqr / numEvents)
                - (meanValue * meanValue)));
    
    }

    public double getUserEventSumSquared() {
        return sumSqr;
    }

    public double getUserEventStdDev() {
        return stdDev;
    }
    
    

  
}