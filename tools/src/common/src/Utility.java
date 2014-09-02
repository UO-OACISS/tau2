package edu.uoregon.tau.common;

import java.awt.Color;
import java.awt.Component;
import java.awt.Container;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.GridBagConstraints;
import java.net.URL;
import java.util.NoSuchElementException;
import java.util.StringTokenizer;

import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JTable;
import javax.swing.table.DefaultTableCellRenderer;

import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.Axis;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.Plot;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.title.LegendTitle;

public class Utility {

	private static final String SPACE=" ";
	private static final String MSPACE_REGX=" {2,}";
    public static String removeRuns(String str) {
    	
    	String[] cut = str.split(MSPACE_REGX);
    	StringBuilder strb = new StringBuilder(str.length());
    	strb.append(cut[0]);
    	for(int i =1;i<cut.length;i++){
    		strb.append(SPACE);
    		strb.append(cut[i]);
    	}

        return strb.toString();
    }

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
     * Join together strings, the opposite of String.split
     * 
     * @return      The joined string
     */
    public static String join(CharSequence delimiter, CharSequence... elements) {
        StringBuilder sb = new StringBuilder(256);
        boolean first = true;
        for (CharSequence s : elements) {
            if (first) {
                first = false;
            } else {
                sb.append(delimiter);
            }
            sb.append(s);
        }
        return sb.toString();
    }

    /**
     * Applies default chart fonts and chart color settings
     * @param chart
     */
    public static void applyDefaultChartTheme(JFreeChart chart) {
        chart.setBackgroundPaint(new Color(238, 238, 238));

        //Set legend label font size to 16
        LegendTitle l = chart.getLegend();
        if (l != null) {
            l.setItemFont(new Font(null, Font.PLAIN, 16));
        }

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
    private static void setAxisFont(Axis axis) {
        if (axis == null) {
            return;
        }
        Font f = axis.getLabelFont();
        if (f == null) {
            f = new Font(null, Font.PLAIN, 16);
        } else {
            f = f.deriveFont((float) 16.0);
        }
        axis.setLabelFont(f);
    }

    /**
     * Shorten a function name
     * @param longName the long function name
     * @return the shortened function name
     */
    public static String shortenFunctionName(String longName) {
		if (longName == null) {
			return "";
		}
        StringTokenizer st = new StringTokenizer(longName, "(");
        String shorter = null;
        try {
            shorter = st.nextToken();
            if (shorter.length() < longName.length()) {
                shorter = shorter + "()";
            }
        } catch (NoSuchElementException e) {
            shorter = longName;
        }
        longName = shorter;
        st = new StringTokenizer(longName, "[{");
        shorter = null;
        try {
            shorter = st.nextToken();
        } catch (NoSuchElementException e) {
            shorter = longName;
        }
        return shorter.trim();
    }

    public static void addCompItem(Container container, Component c, GridBagConstraints gbc, int x, int y, int w, int h) {
        gbc.gridx = x;
        gbc.gridy = y;
        gbc.gridwidth = w;
        gbc.gridheight = h;
        container.add(c, gbc);
    }

    public static void addCompItem(JFrame frame, Component c, GridBagConstraints gbc, int x, int y, int w, int h) {
        gbc.gridx = x;
        gbc.gridy = y;
        gbc.gridwidth = w;
        gbc.gridheight = h;
        frame.getContentPane().add(c, gbc);
    }
    
    
    /**
     *  Sets the given table's row height based on the size of the selected font.
     */
    public static void setTableFontHeight(JTable table, Font f){
    	DefaultTableCellRenderer tcr = new DefaultTableCellRenderer();
    	Component c = tcr.getTableCellRendererComponent(table, "null", true, true, 0, 0);
    	FontMetrics fm = c.getFontMetrics(f);
    	table.setRowHeight(fm.getHeight());
    }

}
