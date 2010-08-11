
package edu.uoregon.tau.perfdmf;

import java.awt.Color;
import java.io.Serializable;

public class Group implements Serializable, Comparable<Group> {

    /**
	 * 
	 */
	private static final long serialVersionUID = 1073298395018154627L;
	private String name;
    private int id;
    
    private boolean colorFlag = false;
    private Color color = null;
    private Color specificColor = null;
    
    private double time;

    public Group(String name, int id) {
        this.name = name;
        this.id = id;
    }

    public String getName() {
        return name;
    }
    
    public void setName(String name) {
        this.name = name;
    }
    
    public int getID() {
        return id;
    }

    public double getTime()
    {
    	return time;
    }
    public int compareTo(Group inObject) {
        return name.compareTo(inObject.getName());
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
    public void setTime(double t)
    {
    	time = t;
    }
    public boolean isColorFlagSet() {
        return colorFlag;
    }

    public void setSpecificColor(Color specificColor) {
        this.specificColor = specificColor;
    }
}