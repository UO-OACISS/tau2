package edu.uoregon.tau.common;

import java.awt.Color;
import java.net.URL;

import javax.swing.ImageIcon;

import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.Plot;
import org.jfree.chart.plot.XYPlot;

public class Utility {

    public static ImageIcon getImageIconResource(String name) {
        java.net.URL imgURL = getResource(name);
        if (imgURL != null) {
            return new ImageIcon(imgURL);
        } else {
            return null;
        }
    }

    public static URL getResource(String name) {
        URL url = null;
        url = Utility.class.getResource(name);
        if (url == null) {
            url = Utility.class.getResource("/" + name);
        }
        if (url == null) {
            url = Utility.class.getResource("resources/" + name);
        }
        return url;
    }

    public static void applyDefaultChartTheme(JFreeChart chart) {
        chart.setBackgroundPaint(new Color(238, 238, 238));
        Plot plot = chart.getPlot();
        
        plot.setBackgroundPaint(Color.white);
        if (plot instanceof XYPlot) {
            XYPlot xyplot = (XYPlot) plot;
            xyplot.setDomainGridlinePaint(Color.gray);
            xyplot.setDomainMinorGridlinePaint(Color.gray);
            xyplot.setRangeGridlinePaint(Color.gray);
            xyplot.setRangeMinorGridlinePaint(Color.gray);
        } else if (plot instanceof CategoryPlot) {
            CategoryPlot cplot = (CategoryPlot) plot; 
            cplot.setBackgroundPaint(Color.white);
            cplot.setDomainGridlinePaint(Color.gray);
            cplot.setRangeGridlinePaint(Color.gray);
        }
    }
}
