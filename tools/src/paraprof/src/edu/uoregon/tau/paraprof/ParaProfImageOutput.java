/* 
 
 ParaProfImageOutput.java
 
 Title:       ParaProfImageOutput.java
 Author:      Robert Bell
 Description: Handles the output of the various panels to image files.
 */

package edu.uoregon.tau.paraprof;

import java.util.*;
import java.awt.*;
import java.awt.image.*;
import java.io.*;
import javax.swing.*;
import javax.imageio.*;
import javax.imageio.stream.*;
import edu.uoregon.tau.dms.dss.*;

public class ParaProfImageOutput {

    public ParaProfImageOutput() {
    }

    public void saveImage(ParaProfImageInterface ref) throws ParaProfException, IOException {

        //Ask the user for a filename and location.
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setDialogTitle("Save Image File");
        //Set the directory.
        fileChooser.setCurrentDirectory(new File(System.getProperty("user.dir")));
        //Get the current file filters.
        javax.swing.filechooser.FileFilter fileFilters[] = fileChooser.getChoosableFileFilters();
        for (int i = 0; i < fileFilters.length; i++)
            fileChooser.removeChoosableFileFilter(fileFilters[i]);
        fileChooser.addChoosableFileFilter(new ParaProfImageFormatFileFilter(ParaProfImageFormatFileFilter.PNG));
        fileChooser.addChoosableFileFilter(new ParaProfImageFormatFileFilter(ParaProfImageFormatFileFilter.JPG));
        fileChooser.setFileSelectionMode(JFileChooser.FILES_ONLY);

        ParaProfImageOptionsPanel paraProfImageOptionsPanel = new ParaProfImageOptionsPanel((Component) ref);
        fileChooser.setAccessory(paraProfImageOptionsPanel);
        fileChooser.addPropertyChangeListener(paraProfImageOptionsPanel);
        int resultValue = fileChooser.showSaveDialog((Component) ref);
        if (resultValue == JFileChooser.APPROVE_OPTION) {
            System.out.println("Saving image ...");
            //Get both the file and FileFilter.
            File f = fileChooser.getSelectedFile();
            javax.swing.filechooser.FileFilter fileFilter = fileChooser.getFileFilter();
            String path = f.getCanonicalPath();
            //Append extension if required.

            //Only create if we recognize the format.
            ParaProfImageFormatFileFilter paraProfImageFormatFileFilter = null;
            if (fileFilter instanceof ParaProfImageFormatFileFilter) {
                paraProfImageFormatFileFilter = (ParaProfImageFormatFileFilter) fileFilter;
                String extension = ParaProfImageFormatFileFilter.getExtension(f);
                //Could probably collapse this if/else based on the order of evaluation of arguments (ie, to make sure
                //the extension is not null before trying to call equals on it).  However, it is easier to understand
                //what is going on this way.
                if (extension == null) {
                    path = path + "." + paraProfImageFormatFileFilter.getExtension();
                    f = new File(path);
                } else if (!(extension.equals("png") || extension.equals("jpg"))) {
                    path = path + "." + paraProfImageFormatFileFilter.getExtension();
                    f = new File(path);
                }

                
                // I'm doing this twice right now because the getImageSize won't be correct until 
                // renderIt has been called with the appropriate settings.  Stupid, I know.
                Dimension d = ref.getImageSize(paraProfImageOptionsPanel.isFullScreen(),
                        paraProfImageOptionsPanel.isPrependHeader());
                BufferedImage bi = new BufferedImage((int) d.getWidth(), (int) d.getHeight(),
                        BufferedImage.TYPE_INT_RGB);
                Graphics2D g2D = bi.createGraphics();

              
                //Draw to this graphics object.
                ref.renderIt(g2D, false, paraProfImageOptionsPanel.isFullScreen(),
                        paraProfImageOptionsPanel.isPrependHeader());

                
                d = ref.getImageSize(paraProfImageOptionsPanel.isFullScreen(),
                        paraProfImageOptionsPanel.isPrependHeader());
                bi = new BufferedImage((int) d.getWidth(), (int) d.getHeight(),
                        BufferedImage.TYPE_INT_RGB);
                g2D = bi.createGraphics();

                //Paint the background white.
                g2D.setColor(Color.white);
                g2D.fillRect(0, 0, (int) d.getWidth(), (int) d.getHeight());

                //Reset the drawing color to black.  The renderIt methods expect it.
                g2D.setColor(Color.black);

                //Draw to this graphics object.
                ref.renderIt(g2D, false, paraProfImageOptionsPanel.isFullScreen(),
                        paraProfImageOptionsPanel.isPrependHeader());

                
                //Now write the image to file.
                ImageWriter writer = null;
                Iterator iter = ImageIO.getImageWritersByFormatName(paraProfImageFormatFileFilter.getExtension().toUpperCase());
                if (iter.hasNext()) {
                    writer = (ImageWriter) iter.next();
                }
                ImageOutputStream imageOut = ImageIO.createImageOutputStream(f);
                writer.setOutput(imageOut);
                IIOImage iioImage = new IIOImage(bi, null, null);

                //Try setting quality.
                if (paraProfImageOptionsPanel.imageQualityEnabled()) {
                    ImageWriteParam iwp = writer.getDefaultWriteParam();
                    iwp.setCompressionMode(ImageWriteParam.MODE_EXPLICIT);
                    iwp.setCompressionQuality(paraProfImageOptionsPanel.getImageQuality());
                    System.out.println("Quality is: " + iwp.getCompressionQuality());
                    writer.write(null, iioImage, iwp);
                } else
                    writer.write(iioImage);

                System.out.println("Done saving image.");
            } else {
                if (UtilFncs.debug)
                    System.out.println("Aborted saving image ... not a recognized type!");
            }
        } else {
            if (UtilFncs.debug)
                System.out.println("Did not get a file name to save image to.");
            return;
        }
    }
}
