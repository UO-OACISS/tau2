package edu.uoregon.tau.paraprof;

import java.awt.print.*;
import java.awt.*;
import edu.uoregon.tau.dms.dss.*;
import javax.swing.*;
import javax.swing.event.*;
import java.awt.event.*;

public class ParaProfUtils {

    static boolean verbose;
    static boolean verboseSet;

    // Suppress default constructor for noninstantiability
    private ParaProfUtils() {
        // This constructor will never be invoked
    }

    private static void checkVerbose() {
        if (!verboseSet) {
            if (System.getProperty("paraprof.verbose") != null) {
                verbose = true;
            }
            verboseSet = true;
        }
    }
    public static void verr(String string) {
        checkVerbose();
        
        if (verbose) {
            System.err.println(string);
        }
    }

    public static void vout(String string) {
        checkVerbose();

        if (verbose) {
            System.out.println(string);
        }
    }

    public static void vout(Object obj, String string) {
        checkVerbose();

        if (verbose) {

            String className = obj.getClass().getName();
            int lastDot = className.lastIndexOf('.');
            if (lastDot != -1) {
                className = className.substring(lastDot + 1);
            }

            System.out.println(className + ": " + string);
        }
    }
    
    public static void verr(Object obj, String string) {
        checkVerbose();

        if (verbose) {

            String className = obj.getClass().getName();
            int lastDot = className.lastIndexOf('.');
            if (lastDot != -1) {
                className = className.substring(lastDot + 1);
            }

            System.err.println(className + ": " + string);
        }
    }

    public static void helperAddRadioMenuItem(String name, String command, boolean on, ButtonGroup group,
            JMenu menu, ActionListener act) {
        JRadioButtonMenuItem item = new JRadioButtonMenuItem(name, on);
        item.addActionListener(act);
        item.setActionCommand(command);
        group.add(item);
        menu.add(item);
    }

    public static void addCompItem(Container jPanel, Component c, GridBagConstraints gbc, int x, int y, int w,
            int h) {
        gbc.gridx = x;
        gbc.gridy = y;
        gbc.gridwidth = w;
        gbc.gridheight = h;
        jPanel.add(c, gbc);
    }

    public static void print(Printable printable) {
        PrinterJob job = PrinterJob.getPrinterJob();
        PageFormat defaultFormat = job.defaultPage();
        PageFormat selectedFormat = job.pageDialog(defaultFormat);
        if (defaultFormat != selectedFormat) { // only proceed if the user did not select cancel
            job.setPrintable(printable, selectedFormat);
            //if (job.getPrintService() != null) {
            if (job.printDialog()) { // only proceed if the user did not select cancel
                try {
                    job.print();
                } catch (PrinterException e) {
                    ParaProfUtils.handleException(e);
                }
            }
            //}
        }

    }

    public static void scaleForPrint(Graphics g, PageFormat pageFormat, int width, int height) {
        double pageWidth = pageFormat.getImageableWidth();
        double pageHeight = pageFormat.getImageableHeight();
        int cols = (int) (width / pageWidth) + 1;
        int rows = (int) (height / pageHeight) + 1;
        double xScale = pageWidth / width;
        double yScale = pageHeight / height;
        double scale = Math.min(xScale, yScale);

        double tx = 0.0;
        double ty = 0.0;
        if (xScale > scale) {
            tx = 0.5 * (xScale - scale) * width;
        } else {
            ty = 0.5 * (yScale - scale) * height;
        }

        Graphics2D g2 = (Graphics2D) g;

        g2.translate((int) pageFormat.getImageableX(), (int) pageFormat.getImageableY());
        g2.translate(tx, ty);
        g2.scale(scale, scale);
    }

    public static void handleException(Exception e) {
        new ParaProfErrorDialog(e);
    }

}
