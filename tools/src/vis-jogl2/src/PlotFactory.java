/*
 * PlotFactory.java
 *
 * Copyright 2005-2006                                
 * Performance Research Laboratory, University of Oregon
 */

package edu.uoregon.tau.vis;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

/**
 * Factory for creating simple plots.<p>
 *    
 * TODO: Implement other factory methods.
 *
 * <P>CVS $Id: PlotFactory.java,v 1.7 2009/08/21 19:00:18 amorris Exp $</P>
 * @author	Alan Morris
 * @version	$Revision: 1.7 $
 */
public class PlotFactory {

    /**
     * Helper to create a ScatterPlot.
     * @param xLabel x axis label.
     * @param yLabel y axis label
     * @param zLabel z axis label
     * @param colorLabel colors axis label.
     * @param values values to use.
     * @param normalized whether or not to normalized the data along each axis.
     * @param colorScale the ColorScale to use.
     * @return the ScatterPlot created.
     */
    @SuppressWarnings("unchecked")
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

        List<String>[] axisStrings = new ArrayList[4];

        for (int i = 0; i < 4; i++) {
            if (minScatterValues[i] == Float.MAX_VALUE) {
                minScatterValues[i] = 0;
            }
            if (maxScatterValues[i] == Float.MIN_VALUE) {
                maxScatterValues[i] = 0;
            }

            axisStrings[i] = new ArrayList<String>();

            if (scatterPlot.getNormalized()) {
                axisStrings[i].add(getSaneDoubleString(minScatterValues[i]));
                axisStrings[i].add(getSaneDoubleString(minScatterValues[i] + (maxScatterValues[i] - minScatterValues[i]) * .25));
                axisStrings[i].add(getSaneDoubleString(minScatterValues[i] + (maxScatterValues[i] - minScatterValues[i]) * .50));
                axisStrings[i].add(getSaneDoubleString(minScatterValues[i] + (maxScatterValues[i] - minScatterValues[i]) * .75));
                axisStrings[i].add(getSaneDoubleString(maxScatterValues[i]));
            } else {
                axisStrings[i].add("0");
                axisStrings[i].add(getSaneDoubleString(maxScatterValues[i] * .25));
                axisStrings[i].add(getSaneDoubleString(maxScatterValues[i] * .50));
                axisStrings[i].add(getSaneDoubleString(maxScatterValues[i] * .75));
                axisStrings[i].add(Float.toString(maxScatterValues[i]));
            }
        }

        axes.setStrings(xLabel, yLabel, zLabel, axisStrings[0], axisStrings[1], axisStrings[2]);
        colorScale.setStrings((String) axisStrings[3].get(0), (String) axisStrings[3].get(4), colorLabel);
        //colorScale.setStrings("Ggs\ngGo\nBob","asasdfasdfasdfasdfasdf\nbgr","ABCTGasdf\nted");

        // Initialize the scatterPlot
        scatterPlot.setSize(15, 15, 15);
        scatterPlot.setAxes(axes);
        scatterPlot.setColorScale(colorScale);
        scatterPlot.setValues(values);

        return scatterPlot;
    }

    public static String getSaneDoubleString(double d) {
        return formatDouble(d, 4, false);
    }

    // this has been temporarily transplanted here
    // left pad : pad string 's' up to length plen, but put the whitespace on
    // the left
    public static String lpad(String s, int plen) {
        int len = plen - s.length();
        if (len <= 0)
            return s;
        char padchars[] = new char[len];
        for (int i = 0; i < len; i++)
            padchars[i] = ' ';
        String str = new String(padchars, 0, len);
        return str.concat(s);
    }

    public static String formatDouble(double d, int width, boolean pad) {

        // first check if the regular toString is in exponential form
        boolean exp = false;
        String str = Double.toString(d);
        for (int i = 0; i < str.length(); i++) {
            if (str.charAt(i) == 'E') {
                exp = true;
                break;
            }
        }

        if (!exp) {
            // not exponential form
            String formatString = "";

            // create a format string of the same length, (e.g. ###.### for 123.456)
            for (int i = 0; i < str.length(); i++) {
                if (str.charAt(i) != '.') {
                    formatString = formatString + "#";
                } else {
                    formatString = formatString + ".";
                }
            }

            //            DecimalFormat bil = new DecimalFormat("#,###,###,##0");
            //            DecimalFormat mil = new DecimalFormat("#,###,##0");
            //            DecimalFormat thou = new DecimalFormat("#,##0");

            // now we reduce that format string as follows

            // first, do the minimum of 'width' or the length of the regular toString

            int min = width;
            if (formatString.length() < min) {
                min = formatString.length();
            }

            // we don't want more than 4 digits past the decimal point
            // this 4 would be the old ParaProf.defaultNumberPrecision
            if (formatString.indexOf('.') + 4 < min) {
                min = formatString.indexOf('.') + 4;
            }

            formatString = formatString.substring(0, min);

            // remove trailing dot
            if (formatString.charAt(formatString.length() - 1) == '.')
                formatString = formatString.substring(0, formatString.length() - 2);

            DecimalFormat dF = new DecimalFormat(formatString);

            str = dF.format(d);
            //System.out.println("value: " + d + ", width: " + width + ", returning: '" + lpad(str, width) + "'");
            if (pad) {
                return lpad(str, width);
            } else {
                return str;
            }

        }

        // toString used exponential form, so we ought to also

        String formatString;
        if (d < 0.1) {
            formatString = "0.0";
        } else {
            // we want up to four significant digits
            formatString = "0.0###";
        }

        formatString = formatString + "E0";
        DecimalFormat dF = new DecimalFormat(formatString);

        str = dF.format(d);
        if (pad) {
            return lpad(str, width);
        } else {
            return str;
        }
    }
}
