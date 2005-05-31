
package edu.uoregon.tau.paraprof.vis;

import java.util.ArrayList;
import java.util.List;

/**
 * Factory for creating simple plots
 *    
 * TODO: Implement other factory methods
 *
 * <P>CVS $Id: PlotFactory.java,v 1.4 2005/05/31 23:21:53 amorris Exp $</P>
 * @author	Alan Morris
 * @version	$Revision: 1.4 $
 */
public class PlotFactory {

    public static ScatterPlot createScatterPlot(String xLabel, String yLabel, String zLabel, String colorLabel, float values[][],
            boolean normalized, ColorScale colorScale) {

        ScatterPlot scatterPlot = new ScatterPlot();
        scatterPlot.setNormalized(normalized);
        Axes axes = new Axes();

        // find the minimum and maximum values for for each axis 
        float[] minScatterValues = new float[4];
        float[] maxScatterValues = new float[4];
        for (int i = 0; i < 4; i++) {
            maxScatterValues[i] = Float.MIN_VALUE;
            minScatterValues[i] = Float.MAX_VALUE;
        }
        for (int i = 0; i < values.length; i++) {
            for (int j = 0; j < 4; j++) {
                maxScatterValues[j] = Math.max(maxScatterValues[j], values[i][j]);
                minScatterValues[j] = Math.min(minScatterValues[j], values[i][j]);
            }
        }

        List[] axisStrings = new ArrayList[4];
        
        for (int i = 0; i < 4; i++) {
            if (minScatterValues[i] == Float.MAX_VALUE) {
                minScatterValues[i] = 0;
            }
            if (maxScatterValues[i] == Float.MIN_VALUE) {
                maxScatterValues[i] = 0;
            }


            axisStrings[i] = new ArrayList();
            
            if (scatterPlot.getNormalized()) {
                axisStrings[i].add(Float.toString(minScatterValues[i]));
                axisStrings[i].add(Double.toString(minScatterValues[i] + (maxScatterValues[i] - minScatterValues[i]) * .25));
                axisStrings[i].add(Double.toString(minScatterValues[i] + (maxScatterValues[i] - minScatterValues[i]) * .50));
                axisStrings[i].add(Double.toString(minScatterValues[i] + (maxScatterValues[i] - minScatterValues[i]) * .75));
                axisStrings[i].add(Double.toString(maxScatterValues[i]));
            } else {
                axisStrings[i].add("0");
                axisStrings[i].add(Double.toString(maxScatterValues[i] * .25));
                axisStrings[i].add(Double.toString(maxScatterValues[i] * .50));
                axisStrings[i].add(Double.toString(maxScatterValues[i] * .75));
                axisStrings[i].add(Float.toString(maxScatterValues[i]));
            }
        }
        
        axes.setStrings(xLabel, yLabel, zLabel, axisStrings[0], axisStrings[1], axisStrings[2]);
        colorScale.setStrings((String)axisStrings[3].get(0), (String)axisStrings[3].get(4), colorLabel);
        
        // Initialize the scatterPlot
        scatterPlot.setSize(15, 15, 15);
        scatterPlot.setAxes(axes);
        scatterPlot.setColorScale(colorScale);
        scatterPlot.setValues(values);
        
        return scatterPlot;
    }

}
