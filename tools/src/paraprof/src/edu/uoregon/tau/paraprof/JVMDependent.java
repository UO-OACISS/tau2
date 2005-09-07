package edu.uoregon.tau.paraprof;

import java.awt.Component;
import java.awt.Toolkit;
import java.awt.datatransfer.Clipboard;
import java.awt.datatransfer.ClipboardOwner;
import java.awt.datatransfer.StringSelection;
import java.io.*;

import javax.swing.JFileChooser;
import javax.swing.JOptionPane;

import org.apache.batik.dom.GenericDOMImplementation;
import org.apache.batik.svggen.SVGGraphics2D;
import org.w3c.dom.DOMImplementation;
import org.w3c.dom.Document;

import edu.uoregon.tau.paraprof.interfaces.ImageExport;

public class JVMDependent {

    public static final String version = "1.4";
    
    public static void main(String[] args) {
        System.out.println("I was compiled with Java 1.4");
    }

    public static void setClipboardContents(String contents, ClipboardOwner owner) {
        if (contents == null || contents == "")
            return;
        Toolkit tk = Toolkit.getDefaultToolkit();
        StringSelection st = new StringSelection(contents);
        Clipboard cp = tk.getSystemSelection();
        if (cp != null) { // some systems (e.g. windows) don't have a system selection clipboard
            cp.setContents(st, owner);
        }
        cp = tk.getSystemClipboard();
        if (cp != null) {
            cp.setContents(st, owner);
        }
    }
    
    
    public static void exportSVG(ImageExport ie) throws Exception {
        //Ask the user for a filename and location.
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setDialogTitle("Save Image File");
        //Set the directory.
        fileChooser.setCurrentDirectory(new File(System.getProperty("user.dir")));
        //Get the current file filters.
        javax.swing.filechooser.FileFilter fileFilters[] = fileChooser.getChoosableFileFilters();
        for (int i = 0; i < fileFilters.length; i++)
            fileChooser.removeChoosableFileFilter(fileFilters[i]);
        fileChooser.addChoosableFileFilter(new ParaProfImageFormatFileFilter(ParaProfImageFormatFileFilter.SVG));
        fileChooser.setFileSelectionMode(JFileChooser.FILES_ONLY);

        ParaProfImageOptionsPanel paraProfImageOptionsPanel = new ParaProfImageOptionsPanel((Component) ie, true);
        fileChooser.setAccessory(paraProfImageOptionsPanel);
        fileChooser.addPropertyChangeListener(paraProfImageOptionsPanel);
        int resultValue = fileChooser.showSaveDialog((Component) ie);
        if (resultValue != JFileChooser.APPROVE_OPTION) {
            return;
        }

        
        File file = fileChooser.getSelectedFile();
        String path = file.getCanonicalPath();

        String extension = ParaProfImageFormatFileFilter.getExtension(file);
        if (extension == null) {
            path = path + ".svg";
            file = new File(path);
        }

        if (file.exists()) {
            int response = JOptionPane.showConfirmDialog((Component)ie, file + " already exists\nOverwrite existing file?",
                    "Confirm Overwrite", JOptionPane.OK_CANCEL_OPTION, JOptionPane.QUESTION_MESSAGE);
            if (response == JOptionPane.CANCEL_OPTION)
                return;
        }

        
        // Get a DOMImplementation
        DOMImplementation domImpl = GenericDOMImplementation.getDOMImplementation();

        // Create an instance of org.w3c.dom.Document
        Document document = domImpl.createDocument(null, "svg", null);

        // Create an instance of the SVG Generator
        SVGGraphics2D svgGenerator = new SVGGraphics2D(document);

        ie.export(svgGenerator, false, paraProfImageOptionsPanel.isFullScreen(),
                paraProfImageOptionsPanel.isPrependHeader());
        
        // Finally, stream out SVG to the standard output using UTF-8
        // character to byte encoding
        boolean useCSS = true; // we want to use CSS style attribute
        FileOutputStream fos = new FileOutputStream(file);
        Writer out = new OutputStreamWriter(fos, "UTF-8");
        svgGenerator.stream(out, useCSS);
    }

    
    
}
