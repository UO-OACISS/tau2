/* 
  
  ParaProfImageInterface.java
  
  Title:       ParaProfImageInterface.java
  Author:      Robert Bell
  Description: Handles the output of the various panels to image files.
*/

package edu.uoregon.tau.paraprof;

import java.awt.*;

interface ParaProfImageInterface{

    //instruction: 1) 0:screen  2) 1:image (visible) 3) 2:image (full) 4) 3:print 
    
    // fullWindow 0 or 1
    // toStreen 0
    
//    public void renderIt(Graphics2D g, int instruction, boolean header);
    
    public void renderIt(Graphics2D g2D, boolean toScreen, boolean fullWindow, boolean drawHeader);

    public Dimension getImageSize(boolean fullScreen, boolean header);
}
