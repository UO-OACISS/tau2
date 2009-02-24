/*
 * Created on Jun 30, 2005
 *
 * TODO To change the template for this generated file go to
 * Window - Preferences - Java - Code Style - Code Templates
 */
package edu.uoregon.tau.perfexplorer.server;

import org.jfree.chart.ChartColor;
import java.awt.Color;

/**
 * Description
 *
 * <P>CVS $Id: PEChartColor.java,v 1.2 2009/02/24 00:53:45 khuck Exp $</P>
 * @author khuck
 * @version 0.1
 * @since   0.1
 *
 */
public class PEChartColor extends ChartColor {
    /**
     * Creates a Color with an opaque sRGB with red, green and blue values in range 0-255.
     *
     * @param r  the red component in range 0x00-0xFF.
     * @param g  the green component in range 0x00-0xFF.
     * @param b  the blue component in range 0x00-0xFF.
     */
    public PEChartColor(int r, int g, int b) {
        super(r, g, b); 
    }

    public static Color[] createDefaultColorArray() {
	    	return new Color[] {
			Color.red,
			Color.blue,
			Color.green,
			Color.yellow,
			Color.orange,
			Color.magenta,
			Color.cyan,
			Color.pink,
			Color.gray,
			ChartColor.DARK_RED,
			ChartColor.DARK_BLUE,
			ChartColor.DARK_GREEN,
			ChartColor.DARK_YELLOW,
			ChartColor.DARK_MAGENTA,
			ChartColor.DARK_CYAN,
			Color.darkGray,
			ChartColor.LIGHT_RED,
			ChartColor.LIGHT_BLUE,
			ChartColor.LIGHT_GREEN,
			ChartColor.LIGHT_YELLOW,
			ChartColor.LIGHT_MAGENTA,
			ChartColor.LIGHT_CYAN,
			Color.lightGray,
			ChartColor.VERY_DARK_RED,
			ChartColor.VERY_DARK_BLUE,
			ChartColor.VERY_DARK_GREEN,
			ChartColor.VERY_DARK_YELLOW,
			ChartColor.VERY_DARK_MAGENTA,
			ChartColor.VERY_DARK_CYAN,
			ChartColor.VERY_LIGHT_RED,
			ChartColor.VERY_LIGHT_BLUE,
			ChartColor.VERY_LIGHT_GREEN,
			ChartColor.VERY_LIGHT_YELLOW,
			ChartColor.VERY_LIGHT_MAGENTA,
			ChartColor.VERY_LIGHT_CYAN
	    	};
    }
}
