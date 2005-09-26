/* 
 
 ParaProfImageOutput.java
 
 Title:       ParaProfImageOutput.java
 Author:      Robert Bell
 Description: Handles the output of the various panels to image files.
 */

package edu.uoregon.tau.paraprof;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Iterator;

import javax.imageio.*;
import javax.imageio.stream.ImageOutputStream;
import javax.swing.JFileChooser;
import javax.swing.JOptionPane;

import edu.uoregon.tau.paraprof.interfaces.ImageExport;

public class ParaProfImageOutput {

    // do not allow instantiation
    private ParaProfImageOutput() {
    }

    public static void saveImage(ImageExport ref) throws IOException {

        //Ask the user for a filename and location.
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setDialogTitle("Save Image File");
        //Set the directory.
        fileChooser.setCurrentDirectory(new File(System.getProperty("user.dir")));
        //Get the current file filters.
        javax.swing.filechooser.FileFilter fileFilters[] = fileChooser.getChoosableFileFilters();
        for (int i = 0; i < fileFilters.length; i++)
            fileChooser.removeChoosableFileFilter(fileFilters[i]);
        fileChooser.addChoosableFileFilter(new ParaProfImageFormatFileFilter(ParaProfImageFormatFileFilter.JPG));
        fileChooser.addChoosableFileFilter(new ParaProfImageFormatFileFilter(ParaProfImageFormatFileFilter.PNG));
        fileChooser.setFileSelectionMode(JFileChooser.FILES_ONLY);

        ParaProfImageOptionsPanel paraProfImageOptionsPanel = new ParaProfImageOptionsPanel((Component) ref, true, false);
        fileChooser.setAccessory(paraProfImageOptionsPanel);
        fileChooser.addPropertyChangeListener(paraProfImageOptionsPanel);
        int resultValue = fileChooser.showSaveDialog((Component) ref);
        if (resultValue != JFileChooser.APPROVE_OPTION) {
            return;
        }

        //Get both the file and FileFilter.
        File file = fileChooser.getSelectedFile();
        String path = file.getCanonicalPath();

        ParaProfImageFormatFileFilter paraProfImageFormatFileFilter = null;
        javax.swing.filechooser.FileFilter fileFilter = fileChooser.getFileFilter();
        if (fileFilter instanceof ParaProfImageFormatFileFilter) {
            paraProfImageFormatFileFilter = (ParaProfImageFormatFileFilter) fileFilter;
        } else {
            throw new ParaProfException("Unknown format : " + fileFilter);
            //???
        }
        String extension = ParaProfImageFormatFileFilter.getExtension(file);
        if (extension == null) {
            extension = paraProfImageFormatFileFilter.getExtension();
            path = path + "." + extension;
            file = new File(path);
        }

        if (file.exists()) {
            int response = JOptionPane.showConfirmDialog((Component) ref, file + " already exists\nOverwrite existing file?",
                    "Confirm Overwrite", JOptionPane.OK_CANCEL_OPTION, JOptionPane.QUESTION_MESSAGE);
            if (response == JOptionPane.CANCEL_OPTION)
                return;
        }

        // I'm doing this twice right now because the getImageSize won't be correct until 
        // renderIt has been called with the appropriate settings.  Stupid, I know.
        Dimension d = ref.getImageSize(paraProfImageOptionsPanel.isFullScreen(), paraProfImageOptionsPanel.isPrependHeader());
        BufferedImage bi = new BufferedImage((int) d.getWidth(), (int) d.getHeight(), BufferedImage.TYPE_INT_RGB);
        Graphics2D g2D = bi.createGraphics();

        //Draw to this graphics object.
        ref.export(g2D, false, paraProfImageOptionsPanel.isFullScreen(), paraProfImageOptionsPanel.isPrependHeader());

        d = ref.getImageSize(paraProfImageOptionsPanel.isFullScreen(), paraProfImageOptionsPanel.isPrependHeader());
        bi = new BufferedImage((int) d.getWidth(), (int) d.getHeight(), BufferedImage.TYPE_INT_RGB);
        g2D = bi.createGraphics();

        //Paint the background white.
        g2D.setColor(Color.white);
        g2D.fillRect(0, 0, (int) d.getWidth(), (int) d.getHeight());

        //Reset the drawing color to black.  The renderIt methods expect it.
        g2D.setColor(Color.black);

        //Draw to this graphics object.
        ref.export(g2D, false, paraProfImageOptionsPanel.isFullScreen(), paraProfImageOptionsPanel.isPrependHeader());

        //Now write the image to file.
        ImageWriter writer = null;
        Iterator iter = ImageIO.getImageWritersByFormatName(paraProfImageFormatFileFilter.getExtension().toUpperCase());
        if (iter.hasNext()) {
            writer = (ImageWriter) iter.next();
        }
        ImageOutputStream imageOut = ImageIO.createImageOutputStream(file);
        writer.setOutput(imageOut);
        IIOImage iioImage = new IIOImage(bi, null, null);

        //Try setting quality.
        if (extension.compareTo("jpg") == 0) {
            ImageWriteParam iwp = writer.getDefaultWriteParam();
            iwp.setCompressionMode(ImageWriteParam.MODE_EXPLICIT);
            iwp.setCompressionQuality(paraProfImageOptionsPanel.getImageQuality());
            writer.write(null, iioImage, iwp);
        } else {
            writer.write(iioImage);
        }
    }

    public static void save3dImage(final ThreeDeeWindow ref) throws IOException {

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

        final ParaProfImageOptionsPanel paraProfImageOptionsPanel = new ParaProfImageOptionsPanel((Component) ref, false, false);
        fileChooser.setAccessory(paraProfImageOptionsPanel);
        fileChooser.addPropertyChangeListener(paraProfImageOptionsPanel);
        int resultValue = fileChooser.showSaveDialog((Component) ref);
        if (resultValue != JFileChooser.APPROVE_OPTION) {
            return;
        }

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

            final String extensionString = paraProfImageFormatFileFilter.getExtension().toUpperCase();

            final File filename = f;

            fileChooser.setVisible(false);

            BufferedImage bi = ref.getImage();

            //Now write the image to file.
            ImageWriter writer = null;
            Iterator iter = ImageIO.getImageWritersByFormatName(extensionString);
            if (iter.hasNext()) {
                writer = (ImageWriter) iter.next();
            }
            ImageOutputStream imageOut = ImageIO.createImageOutputStream(filename);
            writer.setOutput(imageOut);
            IIOImage iioImage = new IIOImage(bi, null, null);

            //Try setting quality.
            if (paraProfImageOptionsPanel.imageQualityEnabled()) {
                ImageWriteParam iwp = writer.getDefaultWriteParam();
                iwp.setCompressionMode(ImageWriteParam.MODE_EXPLICIT);
                iwp.setCompressionQuality(paraProfImageOptionsPanel.getImageQuality());
                writer.write(null, iioImage, iwp);
            } else {
                writer.write(iioImage);
            }

        }
    }

}
