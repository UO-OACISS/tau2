/* 
 
 ParaProfImageOutput.java
 
 Title:       ParaProfImageOutput.java
 Author:      Robert Bell
 Description: Handles the output of the various panels to image files.
 */

package edu.uoregon.tau.paraprof;

import java.awt.Color;
import java.awt.Component;
import java.awt.Dimension;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Iterator;

import javax.imageio.IIOImage;
import javax.imageio.ImageIO;
import javax.imageio.ImageWriteParam;
import javax.imageio.ImageWriter;
import javax.imageio.stream.ImageOutputStream;
import javax.swing.JFileChooser;
import javax.swing.JOptionPane;

import edu.uoregon.tau.common.ImageExport;
import edu.uoregon.tau.common.ImageFormatFileFilter;
import edu.uoregon.tau.common.ImageOptionsPanel;

public class ParaProfImageOutput {

    // do not allow instantiation
    private ParaProfImageOutput() {}

    public static void saveImage(ImageExport ref) throws IOException {

        //Ask the user for a filename and location.
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setDialogTitle("Save Image File");
        //Set the directory.
        fileChooser.setCurrentDirectory(new File(System.getProperty("user.dir")));
        //Get the current file filters.
        javax.swing.filechooser.FileFilter fileFilters[] = fileChooser.getChoosableFileFilters();
        for (int i = 0; i < fileFilters.length; i++) {
            fileChooser.removeChoosableFileFilter(fileFilters[i]);
        }
        fileChooser.addChoosableFileFilter(new ImageFormatFileFilter(ImageFormatFileFilter.JPG));
        fileChooser.addChoosableFileFilter(new ImageFormatFileFilter(ImageFormatFileFilter.PNG));
        fileChooser.setFileSelectionMode(JFileChooser.FILES_ONLY);

        ImageOptionsPanel paraProfImageOptionsPanel = new ImageOptionsPanel((Component) ref, true, false);
        fileChooser.setAccessory(paraProfImageOptionsPanel);
        fileChooser.addPropertyChangeListener(paraProfImageOptionsPanel);
        int resultValue = fileChooser.showSaveDialog((Component) ref);
        if (resultValue != JFileChooser.APPROVE_OPTION) {
            return;
        }
        File file = fileChooser.getSelectedFile();

        String path = file.getCanonicalPath();

        ImageFormatFileFilter paraProfImageFormatFileFilter = null;
        javax.swing.filechooser.FileFilter fileFilter = fileChooser.getFileFilter();
        if (fileFilter instanceof ImageFormatFileFilter) {
            paraProfImageFormatFileFilter = (ImageFormatFileFilter) fileFilter;
        } else {
            throw new ParaProfException("Unknown format : " + fileFilter);
        }
        String extension = ImageFormatFileFilter.getExtension(file);
        if (extension == null
                || ((extension.toUpperCase().compareTo("JPG") != 0) && (extension.toUpperCase().compareTo("PNG") != 0))) {
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
        saveImage(ref, file, paraProfImageOptionsPanel.isFullScreen(), paraProfImageOptionsPanel.isPrependHeader(),
                paraProfImageOptionsPanel.getImageQuality());
    }

    public static void saveImage(ImageExport ref, File file) throws IOException {
        saveImage(ref, file, false, true, 100);
    }

    public static void saveImage(ImageExport ref, String file) throws IOException {
        saveImage(ref, new File(file), false, true, 100);
    }

    public static void saveImage(ImageExport ref, File file, boolean fullScreen, boolean prependHeader, float imageQuality)
            throws IOException {
        String extension = ImageFormatFileFilter.getExtension(file);

        // I'm doing this twice right now because the getImageSize won't be correct until 
        // renderIt has been called with the appropriate settings.  Stupid, I know.
        Dimension d = ref.getImageSize(fullScreen, prependHeader);
        d.height = Math.max(d.height, 1);
        d.width = Math.max(d.width, 1);
        BufferedImage bi = new BufferedImage((int) d.getWidth(), (int) d.getHeight(), BufferedImage.TYPE_INT_RGB);
        Graphics2D g2D = bi.createGraphics();

        //Draw to this graphics object.
        ref.export(g2D, false, fullScreen, prependHeader);

        d = ref.getImageSize(fullScreen, prependHeader);
        d.height = Math.max(d.height, 1);
        d.width = Math.max(d.width, 1);
        bi = new BufferedImage((int) d.getWidth(), (int) d.getHeight(), BufferedImage.TYPE_INT_RGB);
        g2D = bi.createGraphics();

        //Paint the background white.
        g2D.setColor(Color.white);
        g2D.fillRect(0, 0, (int) d.getWidth(), (int) d.getHeight());

        //Reset the drawing color to black.  The renderIt methods expect it.
        g2D.setColor(Color.black);

        //Draw to this graphics object.
        ref.export(g2D, false, fullScreen, prependHeader);

        //Now write the image to file.
        ImageWriter writer = null;
        Iterator<ImageWriter> iter = ImageIO.getImageWritersByFormatName(extension.toUpperCase());
        if (iter.hasNext()) {
            writer = (ImageWriter) iter.next();
        }
        if (writer == null) {
            JOptionPane.showMessageDialog((Component) ref, "Couldn't find Image Writer for extension '."
                    + extension.toUpperCase() + "'");
            return;
        }
        ImageOutputStream imageOut = ImageIO.createImageOutputStream(file);
        writer.setOutput(imageOut);
        IIOImage iioImage = new IIOImage(bi, null, null);

        //Try setting quality.
        if (extension.toUpperCase().compareTo("JPG") == 0) {
            ImageWriteParam iwp = writer.getDefaultWriteParam();
            iwp.setCompressionMode(ImageWriteParam.MODE_EXPLICIT);
            iwp.setCompressionQuality(imageQuality);
            writer.write(null, iioImage, iwp);
        } else {
            writer.write(iioImage);
        }
    }

    public static void save3dImage(final ThreeDeeImageProvider ref) throws IOException {

        //Ask the user for a filename and location.
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setDialogTitle("Save Image File");
        //Set the directory.
        fileChooser.setCurrentDirectory(new File(System.getProperty("user.dir")));
        //Get the current file filters.
        javax.swing.filechooser.FileFilter fileFilters[] = fileChooser.getChoosableFileFilters();
        for (int i = 0; i < fileFilters.length; i++) {
            fileChooser.removeChoosableFileFilter(fileFilters[i]);
        }
        fileChooser.addChoosableFileFilter(new ImageFormatFileFilter(ImageFormatFileFilter.JPG));
        fileChooser.addChoosableFileFilter(new ImageFormatFileFilter(ImageFormatFileFilter.PNG));
        fileChooser.setFileSelectionMode(JFileChooser.FILES_ONLY);

        final ImageOptionsPanel paraProfImageOptionsPanel = new ImageOptionsPanel(ref.getComponent(), false, false);
        fileChooser.setAccessory(paraProfImageOptionsPanel);
        fileChooser.addPropertyChangeListener(paraProfImageOptionsPanel);
        int resultValue = fileChooser.showSaveDialog(ref.getComponent());
        if (resultValue != JFileChooser.APPROVE_OPTION) {
            return;
        }

        File file = fileChooser.getSelectedFile();
        String path = file.getCanonicalPath();

        ImageFormatFileFilter paraProfImageFormatFileFilter = null;
        javax.swing.filechooser.FileFilter fileFilter = fileChooser.getFileFilter();
        if (fileFilter instanceof ImageFormatFileFilter) {
            paraProfImageFormatFileFilter = (ImageFormatFileFilter) fileFilter;
        } else {
            throw new ParaProfException("Unknown format : " + fileFilter);
        }
        
        String extension = ImageFormatFileFilter.getExtension(file);
        if (extension == null
                || ((extension.toUpperCase().compareTo("JPG") != 0) && (extension.toUpperCase().compareTo("PNG") != 0))) {
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
        
        //Only create if we recognize the format.
        ImageFormatFileFilter imageFormatFileFilter = null;
        if (fileFilter instanceof ImageFormatFileFilter) {
            imageFormatFileFilter = (ImageFormatFileFilter) fileFilter;

            //Could probably collapse this if/else based on the order of evaluation of arguments (ie, to make sure
            //the extension is not null before trying to call equals on it).  However, it is easier to understand
            //what is going on this way.
            if (extension == null) {
                path = path + "." + imageFormatFileFilter.getExtension();
                file = new File(path);
            } else if (!(extension.equals("png") || extension.equals("jpg"))) {
                path = path + "." + imageFormatFileFilter.getExtension();
                file = new File(path);
            }

            final String extensionString = imageFormatFileFilter.getExtension().toUpperCase();

            final File filename = file;

            fileChooser.setVisible(false);

            BufferedImage bi = ref.getImage();

            //Now write the image to file.
            ImageWriter writer = null;
            Iterator<ImageWriter> iter = ImageIO.getImageWritersByFormatName(extensionString);
            if (iter.hasNext()) {
                writer = (ImageWriter) iter.next();
            }
            ImageOutputStream imageOut = ImageIO.createImageOutputStream(filename);
            writer.setOutput(imageOut);
            IIOImage iioImage = new IIOImage(bi, null, null);

            //Try setting quality.
            if (extension.toUpperCase().compareTo("JPG") == 0) {
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
