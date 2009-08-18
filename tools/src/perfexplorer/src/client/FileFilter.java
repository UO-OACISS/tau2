package edu.uoregon.tau.perfexplorer.client;

import java.io.File;

/**
 * A custom FileFilter, copied from ParaProfFileFilter
 * 
 * <P>CVS $Id: FileFilter.java,v 1.1 2009/08/18 21:05:46 khuck Exp $</P>
 * @author  Kevin Huck
 * @version $Revision: 1.1 $
 */

public class FileFilter extends javax.swing.filechooser.FileFilter {

    public static String PPK = "ppk";
    public static String TXT = "txt";

    private String extension = null;

    public FileFilter(String extension) {
        super();
        this.extension = extension;
    }

    public boolean accept(File f) {
        boolean accept = f.isDirectory(); // must accept directories for JFileChooser to work properly
        if (!accept) {
            String extension = FileFilter.getExtension(f);
            if (extension != null) {
                accept = this.extension.equals(extension);
            }
        }
        return accept;
    }

    public String getDescription() {
        if (extension.equals(TXT))
            return "Tab Delimited (*.txt)";
        else if (extension.equals(PPK))
            return "TAU Packed Profile (*.ppk)";
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
