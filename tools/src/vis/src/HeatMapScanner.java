// http://forums.sun.com/thread.jspa?forumID=20&threadID=773555

package edu.uoregon.tau.vis;

import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.event.MouseInputAdapter;
 
class HeatMapScanner extends MouseInputAdapter {
    HeatMap heatmap;
    JWindow toolTip;
    JLabel label;
 
    public HeatMapScanner(HeatMap heatmap) {
        this.heatmap = heatmap;
        initToolTip();
    }
 
    private void initToolTip() {
        label = new JLabel(" ");
        label.setOpaque(true);
        label.setBackground(UIManager.getColor("ToolTip.background"));
        toolTip = new JWindow(new Frame());
        toolTip.getContentPane().add(label);
    }
 
    public void mousePressed(MouseEvent e) {
    }
 
    public void mouseMoved(MouseEvent e) {
        Point p = e.getPoint();
        boolean hovering = false;

        if(!toolTip.isVisible()) {
            String s = heatmap.getToolTip(p);
			label.setText(s);
			toolTip.pack();
			toolTip.setVisible(true);
            hovering = true;
            SwingUtilities.convertPointToScreen(p, heatmap);
            toolTip.setLocation(p.x+5, p.y-toolTip.getHeight()-5);
    	}
        if(!hovering && toolTip.isVisible())
            toolTip.setVisible(false);
    }
}