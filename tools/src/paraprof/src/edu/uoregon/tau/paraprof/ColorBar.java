package edu.uoregon.tau.paraprof;

import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.*;

import javax.swing.JComponent;
import java.awt.Color;

/**
 * ColorBar.java
 * This component sits atop the call graph window displaying a bar of colors (the scale).
 * It also contains the static functions for getting a color given a value 0..1
 *  
 * 
 * <P>CVS $Id: ColorBar.java,v 1.2 2004/12/29 00:09:48 amorris Exp $</P>
 * @author	Alan Morris
 * @version	$Revision: 1.2 $
 * @see CallGraphWindow
 */
public class ColorBar extends JComponent {

    private static final double colorsR[] = {
        0, 0, 0, 1, 1,   1 };
    private static final double colorsG[] = {
        0, 1, 1, 1, 0.5, 0 };
    private static final double colorsB[] = {
        1, 1, 0, 0, 0, 0 };
    
        
    public ColorBar() {
        setMinimumSize(new Dimension(10, 20));
    }

    public Dimension getPreferredSize() {
        return new Dimension(10, 20);
    }

    public void paintComponent(Graphics g) {
        try {
            super.paintComponent(g);

            Graphics2D g2D = (Graphics2D) g;

            int width = this.getWidth();

            int numBoxes = ((width - 1) / 11);

            int lastx = 0;
            for (int i = 0; i < numBoxes; i++) {
                g2D.setColor(getColor(i / (float) numBoxes));
                g2D.fillRect((i * 11) + 1, 1, 10, 18);
            }
      } catch (Exception e) {
            // what can I possibly do?
       }

    }

    
    /**
     * Retrieves a color in the colorscale based on a ratio (0..1) (blue..red)
     * 
     * @param	ratio: a value between 0 and 1
     * @return	a color from the scale blue..red (with some other colors inbetween)
     */
    public static Color getColor(float ratio) {
        double r, g, b;
    
        int section;

        if (ratio < 0.125) {
            ratio *= 8;
            section = 0;
        } else if (ratio < 0.25) {
            ratio = (float) (ratio - 0.125) * 8;
            section = 1;
        } else if (ratio < 0.5) {
            ratio = (float) (ratio - 0.25) * 4;
            section = 2;
        } else if (ratio < 0.75) {
            ratio = (float) (ratio - 0.50) * 4;
            section = 3;
        } else {
            ratio = (float) (ratio - 0.75) * 4;
            section = 4;
        }

        r = colorsR[section] + ratio * (colorsR[section + 1] - colorsR[section]);
        g = colorsG[section] + ratio * (colorsG[section + 1] - colorsG[section]);
        b = colorsB[section] + ratio * (colorsB[section + 1] - colorsB[section]);

        if (r > 1)
            r = 1;
        if (g > 1)
            g = 1;
        if (b > 1)
            b = 1;

        return new Color((float) r, (float) g, (float) b);
    }

}
