/*
 * Created on Dec 20, 2004
 *
 * TODO To change the template for this generated file go to
 * Window - Preferences - Java - Code Style - Code Templates
 */
package edu.uoregon.tau.paraprof;

import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.*;

import javax.swing.JComponent;
import java.awt.Color;


/**
 * @author amorris
 *
 * TODO ...
 */
public class ColorBar extends JComponent {


    public ColorBar() {
        setMinimumSize(new Dimension(10,20));
    }
    
    /* (non-Javadoc)
     * @see javax.swing.JComponent#getPreferredSize()
     */
    public Dimension getPreferredSize() {
        return new Dimension(10,20);
        // TODO Auto-generated method stub
        //return super.getPreferredSize();
    }


    
///* (non-Javadoc)
// * @see java.awt.Component#getSize()
// */
//public Dimension getSize() {
//    // TODO Auto-generated method stub
//    Dimension dm = super.getSize();
//    
//    return new Dimension(dm.width, 20);
//    
//}
//    
///* (non-Javadoc)
// * @see javax.swing.JComponent#getHeight()
// */
//public int getHeight() {
//    // TODO Auto-generated method stub
//    return 20;
//}

    
     /* (non-Javadoc)
     * @see javax.swing.JComponent#paint(java.awt.Graphics)
     */
    public void paint(Graphics g) {
        // TODO Auto-generated method stub
        //super.paint(g);
        
        Graphics2D g2D = (Graphics2D) g;
        
        int width = this.getWidth();
        
        int numBoxes = ((width-1) / 11);
        
        int lastx = 0;
        for (int i=0; i < numBoxes; i++) {
            
            g2D.setColor(getColor(i/(float)numBoxes));
            g2D.fillRect((i*11)+1,1,10,18);
        }
    }
    
    
    static Color getColor(float ratio) {
        double r,g,b;
        
        double colorsR[] = new double[6];
        double colorsG[] = new double[6];
        double colorsB[] = new double[6];
        
        colorsR[0] = 0;
        colorsG[0] = 0;
        colorsB[0] = 1;

        colorsR[1] = 0;
        colorsG[1] = 1;
        colorsB[1] = 1;

        colorsR[2] = 0;
        colorsG[2] = 1;
        colorsB[2] = 0;

        colorsR[3] = 1;
        colorsG[3] = 1;
        colorsB[3] = 0;

        colorsR[4] = 1;
        colorsG[4] = 0.5;
        colorsB[4] = 0;

        colorsR[5] = 1;
        colorsG[5] = 0;
        colorsB[5] = 0;

        
        
        int section;
        
        if (ratio < 0.125) {
            ratio *= 8;
            section = 0;
        } else if (ratio < 0.25) {
            ratio = (float)(ratio - 0.125) * 8;
            section = 1;
        } else if (ratio < 0.5) {
            ratio = (float)(ratio - 0.25) * 4;
            section = 2;
        } else if (ratio < 0.75) {
            ratio = (float)(ratio - 0.50) * 4;
            section = 3;
        } else {
            ratio = (float)(ratio - 0.75) * 4;
            section = 4;
        }
        
        r = colorsR[section] + ratio * (colorsR[section+1] - colorsR[section]);
        g = colorsG[section] + ratio * (colorsG[section+1] - colorsG[section]);
        b = colorsB[section] + ratio * (colorsB[section+1] - colorsB[section]);
        
        if (r > 1) r = 1;
        if (g > 1) g = 1;
        if (b > 1) b = 1;
        
        return new Color((float)r,(float)g,(float)b);
    }
    
}
