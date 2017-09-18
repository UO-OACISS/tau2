// http://forums.sun.com/thread.jspa?forumID=20&threadID=773555

package edu.uoregon.tau.vis;

import java.awt.*;
import java.awt.event.*;

import javax.swing.*;
import javax.swing.event.MouseInputAdapter;
 
class HeatMapScanner extends MouseInputAdapter implements KeyListener, MouseWheelListener  {
    private HeatMap heatmap;
    private JWindow toolTip;
    private JLabel label;
    private int maxSize = 0;
    private int viewportSize = 512;
    private int zoomMax = 0;
 
    public HeatMapScanner(HeatMap heatmap, int maxSize) {
        this.heatmap = heatmap;
        initToolTip();
        this.maxSize = maxSize;
        this.zoomMax = this.viewportSize * this.maxSize;
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
    
    public void mouseWheelMoved(MouseWheelEvent e) {
    	int notches = e.getWheelRotation();
    	if (notches < 0) {
        	zoomIn();
    	} else {
    		zoomOut();
    	}
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
            // zoom in and out on +/-
            if (key == '+' || key == '=') {
            	zoomIn();
            } else if (key == '-' || key == '_') {
            	zoomOut();
            }
        } catch (Exception e) {
        }
	}

	private int zoomOut() {
    	int currentSize = this.heatmap.getPreferredSize().height;
    	int newSize = currentSize;
		if (currentSize <= this.viewportSize) {
			newSize = Math.min(currentSize / 2, heatmap.getMapSize());
		} else if (currentSize > 576) {
			newSize = Math.max(currentSize / 2, this.viewportSize);
		} else { // size between this.viewportSize and 576
			newSize = this.viewportSize;
		}
		if (newSize != currentSize && newSize >= this.viewportSize) {
//                	if (newSize != currentSize && newSize >= heatmap.getMapSize()) {
			heatmap.setPreferredSize(new Dimension(newSize,newSize));
			heatmap.setSize(newSize,newSize);
		}
		return newSize;
	}

	private int zoomIn() {
    	int currentSize = this.heatmap.getPreferredSize().height;
    	int newSize = currentSize;
		if (currentSize <= this.viewportSize) {
			newSize = Math.max(currentSize * 2, this.viewportSize);
		} else if (currentSize >= 576) {
			newSize *= 2;
		} else { // size between this.viewportSize and 576
			newSize = this.viewportSize;
		}
		if (newSize != currentSize && newSize <= this.zoomMax && newSize > 0) {
			heatmap.setPreferredSize(new Dimension(newSize,newSize));
			heatmap.setSize(newSize,newSize);
		}
		return newSize;
	}
	

}