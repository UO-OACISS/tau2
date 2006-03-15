package edu.uoregon.tau.paraprof;

import java.io.*;

/**
 * A custom FileFilter
 * 
 * <P>CVS $Id: ParaProfFileFilter.java,v 1.1 2006/03/15 22:32:27 amorris Exp $</P>
 * @author  Robert Bell, Alan Morris
 * @version $Revision: 1.1 $
 */

public class ParaProfFileFilter extends javax.swing.filechooser.FileFilter {

    private String extension = null;

    public static String JPG = "jpg";
    public static String PNG = "png";
    public static String PPK = "ppk";
    public static String SVG = "svg";
    public static String EPS = "eps";
    public static String TXT = "txt";

    public ParaProfFileFilter(String extension) {
        super();
        this.extension = extension;
    }

    public boolean accept(File f) {
        boolean accept = f.isDirectory(); // must accept directories for JFileChooser to work properly
        if (!accept) {
            String extension = ParaProfFileFilter.getExtension(f);
            if (extension != null) {
                accept = this.extension.equals(extension);
            }
        }
        return accept;
    }

    public String getDescription() {
        if (extension.equals("jpg"))
            return "JPEG File (*.jpg)";
        else if (extension.equals("png"))
            return "PNG File (*.png)";
        else if (extension.equals("txt"))
            return "Tab Delimited (*.txt)";
        else if (extension.equals("ppk"))
            return "ParaProf Packed Profile (*.ppk)";
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
        if (i > 0 && i < s.length() - 1) {
            extension = s.substring(i + 1).toLowerCase();
        }

        return extension;
    }
}
