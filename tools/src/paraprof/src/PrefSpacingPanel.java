/* 
 
 PrefSpacingPanel.java
 
 Title:      ParaProf
 Author:     Robert Bell
 Description:  
 */

package edu.uoregon.tau.paraprof;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.Graphics2D;

import javax.swing.JPanel;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

public class PrefSpacingPanel extends JPanel implements ChangeListener {

    /**
	 * 
	 */
	private static final long serialVersionUID = -6551921675313830683L;

	public PrefSpacingPanel() {
        setSize(new java.awt.Dimension(xPanelSize, yPanelSize));
        setPreferredSize(new java.awt.Dimension(xPanelSize, yPanelSize));
        //Set the default tool tip for this panel.
        setBackground(Color.white);
        this.repaint();
    }

    public void paintComponent(Graphics g) {
        super.paintComponent(g);

        int tmpXPanelSize = 0;
        int tmpYPanelSize = 0;
        Graphics2D g2D = (Graphics2D) g;

        //Set local spacing and bar heights.
        
        barHeight = ParaProf.preferencesWindow.getFontSize();

        //Set up the yCoord.
        yCoord = 25 + barHeight;

        //Create font.

        
        Font font = new Font(ParaProf.preferencesWindow.getFontName(), ParaProf.preferencesWindow.getFontStyle(),
                ParaProf.preferencesWindow.getFontSize());
        //Font font = ParaProf.preferencesWindow.getFont();
        g2D.setFont(font);
        FontMetrics fmFont = g2D.getFontMetrics(font);
        barSpacing = fmFont.getHeight();

        //calculate the maximum string width.
        int maxStringWidth = 0;
        for (int k = 0; k < 3; k++) {
            String s1 = "n,c,t 0,0," + k;
            int tmpInt = fmFont.stringWidth(s1);

            if (maxStringWidth < tmpInt)
                maxStringWidth = tmpInt;
        }

        //Now set the start location of the bars.
        barXStart = maxStringWidth + 25;
        barXCoord = barXStart;

        for (int i = 0; i < 3; i++) {
            String s1 = "n,c,t 0,0," + i;
            int tmpStringWidth = fmFont.stringWidth(s1);
            g2D.drawString(s1, (barXStart - tmpStringWidth - 5), yCoord);

            //After the above check, do the usual drawing stuff.
            for (int j = 0; j < 3; j++) {
                tmpColor = ParaProf.colorChooser.getColor(j);
                g2D.setColor(tmpColor);
                g2D.fillRect(barXCoord, (yCoord - barHeight), 40, barHeight);
                tmpXPanelSize = Math.max(tmpXPanelSize, barXCoord + 40);
                barXCoord = barXCoord + 30;
            }

            barXCoord = barXStart;
            g2D.setColor(Color.black);
            yCoord = yCoord + (barSpacing);
        }

        tmpYPanelSize = yCoord - barSpacing;

        if (xPanelSize != tmpXPanelSize || yPanelSize != tmpYPanelSize) {
            xPanelSize = tmpXPanelSize;
            yPanelSize = tmpYPanelSize;
            this.setSize(new java.awt.Dimension(xPanelSize, yPanelSize));
        }

    }

    public Dimension getPreferredSize() {
        return new Dimension((xPanelSize + 10), (yPanelSize + 10));
    }

    public void stateChanged(ChangeEvent event) {
        ParaProf.preferencesWindow.updateFontSize();
        this.repaint();
    }

    //Instance data.

    int xPanelSize = 200;
    int yPanelSize = 200;

    int barXStart = -1;
    int barXCoord = -1;
    int yCoord = -1;

    int barSpacing = -1;
    int barHeight = -1;

    Color tmpColor;
}
