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
        char key = evt.getKeyChar();
    	System.out.print("Key Pressed: " + key);		
	}

	public void keyReleased(KeyEvent evt) {
		// TODO Auto-generated method stub
        char key = evt.getKeyChar();
    	System.out.print("Key Pressed: " + key);		
	}

	public void keyTyped(KeyEvent evt) {
        try {
            char key = evt.getKeyChar();
        	System.out.print("Key Typed: " + key);
        	int currentSize = this.heatmap.getPreferredSize().height;
            // zoom in and out on +/-
            if (key == '+' || key == '=') {
            	if (currentSize <= 64) {
            		currentSize = Math.max(currentSize * 2, 64);
            	} else if (currentSize >= 128) {
            		currentSize += 64;
            	} else { // size between 64 and 128
            		currentSize = 128;
            	}
    			heatmap.setPreferredSize(new Dimension(currentSize,currentSize));
    			heatmap.setSize(currentSize,currentSize);
            } else if (key == '-' || key == '_') {
            	if (currentSize <= 64) {
            		currentSize = currentSize / 2;
            	} else if (currentSize > 128) {
            		currentSize = Math.max(currentSize - 64, 128);
            	} else { // size between 64 and 128
            		currentSize = 64;
            	}            	
    			heatmap.setPreferredSize(new Dimension(currentSize,currentSize));
    			heatmap.setSize(currentSize,currentSize);
            }
        } catch (Exception e) {
        }
	}
	

}