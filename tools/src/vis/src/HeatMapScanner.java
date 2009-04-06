// http://forums.sun.com/thread.jspa?forumID=20&threadID=773555

package edu.uoregon.tau.vis;

import java.awt.*;
import java.awt.event.*;

import javax.swing.*;
import javax.swing.event.MouseInputAdapter;
 
class HeatMapScanner extends MouseInputAdapter implements KeyListener  {
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

        String s = heatmap.getToolTip(p);
		label.setText(s);
		toolTip.pack();
		toolTip.setVisible(true);
        SwingUtilities.convertPointToScreen(p, heatmap);
        toolTip.setLocation(p.x+5, p.y-toolTip.getHeight()-5);
    }
    
    public void mouseExited(MouseEvent e) {
        toolTip.setVisible(false);
    }
    
	public void keyPressed(KeyEvent evt) {
		// TODO Auto-generated method stub
	}

	public void keyReleased(KeyEvent evt) {
		// TODO Auto-generated method stub
	}

	public void keyTyped(KeyEvent evt) {
        try {
            char key = evt.getKeyChar();
        	int currentSize = this.heatmap.getPreferredSize().height;
        	int newSize = currentSize;
            // zoom in and out on +/-
            if (key == '+' || key == '=') {
            	if (currentSize <= 512) {
            		newSize = Math.max(currentSize * 2, 512);
            	} else if (currentSize >= 576) {
            		newSize += 64;
            	} else { // size between 512 and 576
            		newSize = 512;
            	}
            	if (newSize != currentSize) {
	    			heatmap.setPreferredSize(new Dimension(newSize,newSize));
	    			heatmap.setSize(newSize,newSize);
            	}
            } else if (key == '-' || key == '_') {
            	if (currentSize <= 512) {
            		newSize = Math.min(currentSize / 2, heatmap.getMapSize());
            	} else if (currentSize > 576) {
            		newSize = Math.max(currentSize - 64, 512);
            	} else { // size between 512 and 576
            		newSize = 512;
            	}
            	if (newSize != currentSize && newSize >= heatmap.getMapSize()) {
	    			heatmap.setPreferredSize(new Dimension(newSize,newSize));
	    			heatmap.setSize(newSize,newSize);
            	}
            }
        } catch (Exception e) {
        }
	}
	

}