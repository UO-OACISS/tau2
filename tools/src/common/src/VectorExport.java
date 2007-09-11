package edu.uoregon.tau.common;

import java.awt.Component;
import java.awt.Dimension;
import java.awt.Toolkit;
import java.awt.datatransfer.Clipboard;
import java.awt.datatransfer.ClipboardOwner;
import java.awt.datatransfer.StringSelection;
import java.io.*;

import javax.swing.JComponent;
import javax.swing.JFileChooser;
import javax.swing.JOptionPane;

import org.apache.batik.dom.GenericDOMImplementation;
import org.apache.batik.svggen.SVGGeneratorContext;
import org.apache.batik.svggen.SVGGraphics2D;
import org.w3c.dom.DOMImplementation;
import org.w3c.dom.Document;

public class VectorExport {
	private static String workingDirectory = null;

    public static void promptForVectorExport(ImageExport ie, String applicationName) throws Exception {
        //Ask the user for a filename and location.
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setDialogTitle("Save Vector Graphics File");
        //Set the directory.
		if (workingDirectory == null) {
        	fileChooser.setCurrentDirectory(new File(System.getProperty("user.dir")));
		} else {
        	fileChooser.setCurrentDirectory(new File(workingDirectory));
		}
        //Get the current file filters.
        javax.swing.filechooser.FileFilter fileFilters[] = fileChooser.getChoosableFileFilters();
        for (int i = 0; i < fileFilters.length; i++)
            fileChooser.removeChoosableFileFilter(fileFilters[i]);
        fileChooser.addChoosableFileFilter(new ImageFormatFileFilter(ImageFormatFileFilter.EPS));
        fileChooser.addChoosableFileFilter(new ImageFormatFileFilter(ImageFormatFileFilter.SVG));
        fileChooser.setFileSelectionMode(JFileChooser.FILES_ONLY);

        ImageOptionsPanel optionsPanel = new ImageOptionsPanel((Component) ie, true, true);
        fileChooser.setAccessory(optionsPanel);
        fileChooser.addPropertyChangeListener(optionsPanel);
        int resultValue = fileChooser.showSaveDialog((Component) ie);
        if (resultValue != JFileChooser.APPROVE_OPTION) {
            return;
        }

        File file = fileChooser.getSelectedFile();
        String path = file.getCanonicalPath();
		// save where we were
		VectorExport.workingDirectory = file.getParent();

        String extension = ImageFormatFileFilter.getExtension(file);
        if (extension == null || ((extension.toUpperCase().compareTo("SVG")!=0) && (extension.toUpperCase().compareTo("EPS")!=0)) ) {
            javax.swing.filechooser.FileFilter fileFilter = fileChooser.getFileFilter();
            if (fileFilter instanceof ImageFormatFileFilter) {
                ImageFormatFileFilter paraProfImageFormatFileFilter = (ImageFormatFileFilter) fileFilter;
                path = path + "." + paraProfImageFormatFileFilter.getExtension();
            }
            file = new File(path);
        }

        if (file.exists()) {
            int response = JOptionPane.showConfirmDialog((Component) ie, file + " already exists\nOverwrite existing file?",
                    "Confirm Overwrite", JOptionPane.OK_CANCEL_OPTION, JOptionPane.QUESTION_MESSAGE);
            if (response == JOptionPane.CANCEL_OPTION)
                return;
        }

        boolean textAsShapes = optionsPanel.getTextAsShapes();
		export(ie, file, textAsShapes, applicationName, optionsPanel.isFullScreen(), optionsPanel.isPrependHeader());
	}
        
    public static void export(ImageExport ie, File file, boolean textAsShapes, String applicationName, boolean fullScreen, boolean prependHeader) throws Exception {
        String extension = ImageFormatFileFilter.getExtension(file).toLowerCase();
        if (extension.compareTo("svg") == 0) {
            // Get a DOMImplementation
            DOMImplementation domImpl = GenericDOMImplementation.getDOMImplementation();
            // Create an instance of org.w3c.dom.Document
            Document document = domImpl.createDocument(null, "svg", null);
            // Create an instance of the SVG Generator
            SVGGraphics2D svgGenerator = new SVGGraphics2D(SVGGeneratorContext.createDefault(document), textAsShapes);
            ie.export(svgGenerator, false, fullScreen, prependHeader);
            // Finally, stream out SVG to the standard output using UTF-8
            // character to byte encoding
            boolean useCSS = true; // we want to use CSS style attribute
            FileOutputStream fos = new FileOutputStream(file);
            Writer out = new OutputStreamWriter(fos, "UTF-8");
            svgGenerator.stream(out, useCSS);
        } else if (extension.compareTo("eps") == 0) {

            Dimension d = ie.getImageSize(fullScreen, prependHeader);
            EPSOutput g = new EPSOutput(applicationName, file, d.width, d.height);
            ie.export(g, false, fullScreen, prependHeader);
            g.finish();
            g = new EPSOutput(applicationName, file, d.width, d.height);
            g.setDrawTextAsShapes(textAsShapes);
            ie.export(g, false, fullScreen, prependHeader);
            g.finish();
        } else {
            JOptionPane.showMessageDialog((JComponent)ie, "Unknown format: '" + extension + "'");
        }
    }
}
