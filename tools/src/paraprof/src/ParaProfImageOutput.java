/* 
  
  ParaProfImageOutput.java
  
  Title:       ParaProfImageOutput.java
  Author:      Robert Bell
  Description: Handles the output of the various panels to image files.
*/

package paraprof;

import java.util.*;
import java.awt.*;
import java.awt.image.*;
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
	System.out.println("We have turned this feature off as it has not been fully tested.");
	System.out.println("Saving to an image will be supported in the next release!");
	/*
	try{
	    
	    //Ask the user for a filename and location.
	    JFileChooser tmpFileChooser = new JFileChooser();
	    tmpFileChooser.setDialogTitle("Save Image File");
	    //Set the directory.
	    tmpFileChooser.setCurrentDirectory(new File(System.getProperty("user.dir")));

	    tmpFileChooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
	    int resultValue = tmpFileChooser.showSaveDialog((Component)ref);
	    System.out.println("Saving image ...");
	    if(resultValue == JFileChooser.APPROVE_OPTION){
		//Try and get the file.
		File f = tmpFileChooser.getSelectedFile();
		String tmpString = f.getCanonicalPath();    
		
		Dimension d = ref.getImageSize();
		BufferedImage bi = new BufferedImage((int)d.getWidth(), (int)d.getHeight(), BufferedImage.TYPE_INT_RGB);
		Graphics2D g2D = bi.createGraphics();
		
		//Paint the background white.
		g2D.setColor(Color.white);
		g2D.fillRect(0, 0, (int)d.getWidth(), (int)d.getHeight());
		
		//Reset the drawing color to black.  The renderIt methods expect it.
		g2D.setColor(Color.black);
		
		//Draw to this graphics object.
		ref.renderIt(g2D, "image");

		//Now write the image to file.
		String format = "JPG";
		ImageWriter writer = null;
		Iterator iter = ImageIO.getImageWritersByFormatName(format);
		if(iter.hasNext()){
		    writer = (ImageWriter) iter.next();
		}
		ImageOutputStream imageOut = ImageIO.createImageOutputStream(f);
		writer.setOutput(imageOut);
		IIOImage iioImage = new IIOImage(bi,null,null);
		writer.write(bi);

		System.out.println("Done saving image.");
	    }
	    else{
		if(ParaProf.debugIsOn)
		    System.out.println("Did not get a file name to save image to.");
		return;
	    }
	}
	catch(Exception e){
	    ParaProf.systemError(null, null, "PPII02");
	}*/
    }
}
