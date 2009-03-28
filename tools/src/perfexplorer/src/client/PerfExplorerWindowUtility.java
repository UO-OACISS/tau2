package edu.uoregon.tau.perfexplorer.client;

import java.awt.Dimension;
import java.awt.Point;
import java.awt.Toolkit;
import javax.swing.JFrame;

public class PerfExplorerWindowUtility {

	public static final void centerWindow(JFrame frame, int windowWidth, int windowHeight,
			int offsetX, int offsetY, boolean setSize) {
		if (setSize) {
			frame.setPreferredSize(new Dimension(windowWidth, windowHeight));
		}		
		//Grab the screen size.
        Toolkit tk = Toolkit.getDefaultToolkit();
        Dimension screenDimension = tk.getScreenSize();
        int screenHeight = screenDimension.height;
        int screenWidth = screenDimension.width;

        Point savedPosition = null;

        if (savedPosition == null || (savedPosition.x + windowWidth) > screenWidth
                || (savedPosition.y + windowHeight > screenHeight)) {

            //Find the center position with respect to this window.
            int xPosition = (screenWidth - windowWidth) / 2;
            int yPosition = (screenHeight - windowHeight) / 2;
            xPosition += offsetX;
            yPosition += offsetY;
            frame.setLocation(xPosition, yPosition);
        } else {
            frame.setLocation(savedPosition);
        }
	}
}
