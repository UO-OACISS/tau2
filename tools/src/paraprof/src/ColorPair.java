/* 
 ColorPair.java

 Title:      ParaProf
 Author:     Robert Bell
 Description:  
 */

package edu.uoregon.tau.paraprof;

import java.awt.Color;

public class ColorPair implements Comparable<ColorPair> {

    private String name = null;
    private Color color = null;

    public ColorPair(String name, Color color) {
        this.name = name;
        this.color = color;
    }

    public String getName() {
        return name;
    }

    public Color getColor() {
        return color;
    }

    public int compareTo(ColorPair inObject) {
        String tmpString = inObject.getName();
        return name.compareTo(tmpString);
    }
}
