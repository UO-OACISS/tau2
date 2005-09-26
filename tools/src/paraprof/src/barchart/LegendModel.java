package edu.uoregon.tau.paraprof.barchart;

import java.awt.Color;

public interface LegendModel {

    
    int getNumElements();
    String getLabel(int index);
    Color getColor(int index);
    
}
