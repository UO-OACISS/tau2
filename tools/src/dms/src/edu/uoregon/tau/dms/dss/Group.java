
package edu.uoregon.tau.dms.dss;

import java.awt.*;
import java.io.*;

public class Group implements Serializable, Comparable {

    public Group(String name, int id) {
        this.name = name;
        this.id = id;
    }

    public String getName() {
        return name;
    }
    
    public int getID() {
        return id;
    }

    public int compareTo(Object inObject) {
        return name.compareTo((String) inObject);
    }


    
    //######
    //Begin - Color section.
    //######
    public void setColor(Color color) {
        this.color = color;
    }

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

    //######
    //End - Color section.
    //######

    //Color Settings.
    private boolean colorFlag = false;
    private Color color = null;
    private Color specificColor = null;
    
    String name;
    int id;
}