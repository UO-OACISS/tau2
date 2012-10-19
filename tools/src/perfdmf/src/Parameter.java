
package edu.uoregon.tau.perfdmf;

import java.awt.Color;
import java.io.Serializable;

public class Parameter implements Serializable, Comparable<Parameter> {

    /**
	 * This class was created to store parameter name, value pairs
	 * for parameter based profiles.
	 */
	private String name;
    private String value;
    private int id;
    
    public Parameter(String name, String value, int id) {
        this.name = name;
        this.value = value;
        this.id = id;
    }

    public String getName() {
        return name;
    }
    
    public void setName(String name) {
        this.name = name;
    }

     public String getValue() {
        return value;
    }
    
    public void setValue(String value) {
        this.value = value;
    }
    
    public int getID() {
        return id;
    }

    public void setID(int id) {
        this.id = id;
    }

    public int compareTo(Parameter inObject) {
        return name.compareTo(inObject.getName());
    }

}