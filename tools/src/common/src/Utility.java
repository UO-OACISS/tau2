package edu.uoregon.tau.common;

import java.awt.Color;
import java.net.URL;

import javax.swing.ImageIcon;

import org.jfree.chart.JFreeChart;
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

        chart.getPlot().setBackgroundPaint(Color.white);
        if (chart.getXYPlot() != null) {
            XYPlot plot = chart.getXYPlot();
            plot.setDomainGridlinePaint(Color.gray);
            plot.setDomainMinorGridlinePaint(Color.gray);
            plot.setRangeGridlinePaint(Color.gray);
            plot.setRangeMinorGridlinePaint(Color.gray);

        } else if (chart.getCategoryPlot() != null) {
            chart.getCategoryPlot().setBackgroundPaint(Color.white);
            chart.getCategoryPlot().setDomainGridlinePaint(Color.gray);
            chart.getCategoryPlot().setRangeGridlinePaint(Color.gray);
        }
    }
}
