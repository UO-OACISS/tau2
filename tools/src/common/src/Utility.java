package edu.uoregon.tau.common;

import java.awt.Color;
import java.awt.Font;
import java.net.URL;

import javax.swing.ImageIcon;

import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.Axis;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.Plot;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.title.LegendTitle;

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

    /**
     * Applies default chart fonts and chart color settings
     * @param chart
     */
    public static void applyDefaultChartTheme(JFreeChart chart) {
        chart.setBackgroundPaint(new Color(238, 238, 238));
        
        //Set legend label font size to 16
        LegendTitle l = chart.getLegend();
        l.setItemFont(new Font(null, Font.PLAIN,16));
        
        Plot plot = chart.getPlot();
        
        plot.setBackgroundPaint(Color.white);
        if (plot instanceof XYPlot) {
            XYPlot xyplot = (XYPlot) plot;
            
            setAxisFont(xyplot.getRangeAxis());
            setAxisFont(xyplot.getDomainAxis());
            
            xyplot.setDomainGridlinePaint(Color.gray);
            xyplot.setDomainMinorGridlinePaint(Color.gray);
            xyplot.setRangeGridlinePaint(Color.gray);
            xyplot.setRangeMinorGridlinePaint(Color.gray);
        } else if (plot instanceof CategoryPlot) {
            CategoryPlot cplot = (CategoryPlot) plot; 
            
            setAxisFont(cplot.getRangeAxis());
            setAxisFont(cplot.getDomainAxis());
            
            cplot.setBackgroundPaint(Color.white);
            cplot.setDomainGridlinePaint(Color.gray);
            cplot.setRangeGridlinePaint(Color.gray);
        }
    }
    
    /**
     * Sets the font size for the given axis to 16
     * @param axis
     */
    private static void setAxisFont(Axis axis){
    	Font f = axis.getLabelFont();
    	f=f.deriveFont((float) 16.0);
    	axis.setLabelFont(f);
    }
}
