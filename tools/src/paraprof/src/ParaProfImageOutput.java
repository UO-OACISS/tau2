/* 
  
  ParaProfImageOutput.java
  
  Title:       ParaProfImageOutput.java
  Author:      Robert Bell
  Description: Handles the output of the various panels to image files.
*/

package paraprof;

import java.util.*;
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import javax.swing.*;
import javax.imageio.*;
import javax.imageio.stream.*;
import javax.swing.border.*;
import javax.swing.event.*;
import java.awt.print.*;
import java.awt.geom.*;

public class ParaProfImageOutput{

    public ParaProfImageOutput(){
    }

    public void saveImage(ParaProfImageInterface ref){
	
	BufferedImage bi = new BufferedImage(500, 500, BufferedImage.TYPE_INT_ARGB);
	Graphics2D g2D = bi.createGraphics();
	ref.renderIt(g2D, "image");

	String format = "JPG";
	ImageWriter writer = null;
	Iterator iter = IOImage.getImageWritersByFormatName(format);
	if(iter.hasNext())
	    writer = (ImageWriter) iter.next();

	File f = new File("test.JPG");
	ImageOutputStream imageOut = IIO.createImageOutputStream(f);
	writer.setOutput(imageOut);

	IIOImage iioImage = new IIOImage(bi,null,null);
	
	writer.write(iioImage);
	

	System.out.println("Save image request received!");
    }
}
