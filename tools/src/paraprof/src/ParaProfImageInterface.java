/* 
  
  ParaProfImageInterface.java
  
  Title:       ParaProfImageOutput.java
  Author:      Robert Bell
  Description: Handles the output of the various panels to image files.
*/

package paraprof;

import java.awt.*;

interface ParaProfImageInterface{

    void renderIt(Graphics2D g, String instruction);
    Dimension getImageSize();
}
