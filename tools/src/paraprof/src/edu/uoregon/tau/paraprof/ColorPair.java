/* 
 ColorPair.java

 Title:      ParaProf
 Author:     Robert Bell
 Description:  
 */

package edu.uoregon.tau.paraprof;

import java.awt.*;
import edu.uoregon.tau.dms.dss.*;

public class ColorPair implements Comparable {

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

    public int compareTo(Object inObject) {
        String tmpString = ((ColorPair) inObject).getName();
        return name.compareTo(tmpString);
    }
}
