/* 
  
  ParaProfImageInterface.java
  
  Title:       ParaProfImageOutput.java
  Author:      Robert Bell
  Description: Handles the output of the various panels to image files.
*/

package paraprof;

import java.awt.*;

interface ParaProfImageInterface{

    //instruction: 1) 0:screen  2) 1:image 3) 2:print 
    public void renderIt(Graphics2D g, int instruction);
    public Dimension getImageSize();
}
