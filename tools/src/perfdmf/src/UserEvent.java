
package edu.uoregon.tau.perfdmf;

import java.awt.*;
import java.io.*;

public class UserEvent implements Serializable, Comparable {
    public UserEvent(String name, int id) {
        this.name = name;
        this.id = id;
    }
    
    public String getName() {
        return name;
    }

    public int getID() {
        return id;
    }

    public void setMaxUserEventNumberValue(double maxUserEventNumberValue) {
        this.maxUserEventNumberValue = maxUserEventNumberValue;
    }

    public double getMaxUserEventNumberValue() {
        return maxUserEventNumberValue;
    }

    public void setMaxUserEventMinValue(double maxUserEventMinValue) {
        this.maxUserEventMinValue = maxUserEventMinValue;
    }

    public double getMaxUserEventMinValue() {
        return maxUserEventMinValue;
    }

    public void setMaxUserEventMaxValue(double maxUserEventMaxValue) {
        this.maxUserEventMaxValue = maxUserEventMaxValue;
    }

    public double getMaxUserEventMaxValue() {
        return maxUserEventMaxValue;
    }

    public void setMaxUserEventMeanValue(double maxUserEventMeanValue) {
        this.maxUserEventMeanValue = maxUserEventMeanValue;
    }

    public double getMaxUserEventMeanValue() {
        return maxUserEventMeanValue;
    }

    public void setMaxUserEventSumSquared(double maxUserEventSumSquared) {
        this.maxUserEventSumSquared = maxUserEventSumSquared;
    }

    public double getMaxUserEventSumSquared() {
        return maxUserEventSumSquared;
    }

    public void setMaxUserEventStdDev(double maxUserEventStdDev) {
        this.maxUserEventStdDev = maxUserEventStdDev;
    }

    public double getMaxUserEventStdDev() {
        return maxUserEventStdDev;
    }


    public int compareTo(Object inObject) {
        return name.compareTo(((UserEvent) inObject).getName());
    }

    
    // Color Stuff
    public Color getColor() {
        if (colorFlag)
            return specificColor;
        else
            return color;
    }
    
    public void setColorFlag(boolean colorFlag) {
        this.colorFlag = colorFlag;
    }

    public boolean isColorFlagSet() {
        return colorFlag;
    }

    public void setSpecificColor(Color specificColor) {
        this.specificColor = specificColor;
    }
    
    public void setColor(Color color) {
        this.color = color;
    }

    private String name;
    private int id;
    private Color color;
    private Color specificColor;
    private boolean colorFlag = false;
    
    private double maxUserEventNumberValue = 0;
    private double maxUserEventMinValue = 0;
    private double maxUserEventMaxValue = 0;
    private double maxUserEventMeanValue = 0;
    private double maxUserEventSumSquared = 0;
    private double maxUserEventStdDev = 0;

}