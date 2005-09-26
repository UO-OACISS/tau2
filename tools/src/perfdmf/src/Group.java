
package edu.uoregon.tau.perfdmf;

import java.awt.*;
import java.io.*;

public class Group implements Serializable, Comparable {

    private String name;
    private int id;
    
    private boolean colorFlag = false;
    private Color color = null;
    private Color specificColor = null;
    

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
        return name.compareTo(((Group) inObject).getName());
    }

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
}