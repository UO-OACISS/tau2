package edu.uoregon.tau.common;

import java.io.*;
/**
 * A custom FileFilter for graphics formats
 * 
 * <P>CVS $Id: ImageFormatFileFilter.java,v 1.2 2006/03/29 18:42:37 amorris Exp $</P>
 * @author  Robert Bell
 * @version $Revision: 1.2 $
 */
public class ImageFormatFileFilter extends javax.swing.filechooser.FileFilter {

    public static String JPG = "jpg";
    public static String PNG = "png";
    public static String SVG = "svg";
    public static String EPS = "eps";

    private String extension = null;

    public ImageFormatFileFilter(String extension) {
        super();
        this.extension = extension;
    }

    public boolean accept(File f) {
        boolean accept = f.isDirectory();
        if (!accept) {
            String extension = ImageFormatFileFilter.getExtension(f);
            if (extension != null)
                accept = this.extension.equals(extension);
        }
        return accept;
    }

    public String getDescription() {
        if (extension.equals("jpg"))
            return "JPEG File (*.jpg)";
        else if (extension.equals("png"))
            return "PNG File (*.png)";
        else if (extension.equals("svg"))
            return "Scalable Vector Graphics (*.svg)";
        else if (extension.equals("eps"))
            return "Encapsulated PostScript (*.eps)";
        else
            return "Unknown Extension (*.*)";
    }

    public String toString() {
        return this.getDescription();
    }

    public String getExtension() {
        return extension;
    }

    public static String getExtension(File f) {
        String s = f.getPath();
        String extension = null;

        int i = s.lastIndexOf('.');
        if (i > 0 && i < s.length() - 1)
            extension = s.substring(i + 1).toLowerCase();

        return extension;
    }
}
