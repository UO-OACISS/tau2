package edu.uoregon.tau.paraprof;

import java.awt.print.*;
import java.awt.*;
import edu.uoregon.tau.dms.dss.*;

public class ParaProfUtils {

    // Suppress default constructor for noninstantiability
    private ParaProfUtils() {
        // This constructor will never be invoked
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

    public static double getValue(PPFunctionProfile ppFunctionProfile, int valueType, boolean percent) {
        double value = 0;
        switch (valueType) {
        case 2:
            if (percent)
                value = ppFunctionProfile.getExclusivePercentValue();
            else
                value = ppFunctionProfile.getExclusiveValue();
            break;
        case 4:
            if (percent)
                value = ppFunctionProfile.getInclusivePercentValue();
            else
                value = ppFunctionProfile.getInclusiveValue();
            break;
        case 6:
            value = ppFunctionProfile.getNumberOfCalls();
            break;
        case 8:
            value = ppFunctionProfile.getNumberOfSubRoutines();
            break;
        case 10:
            value = ppFunctionProfile.getInclusivePerCall();
            break;
        default:
            throw new ParaProfException("Invalid Value Type: " + valueType);
        }
        return value;
    }

//    public static double getMaxValue(Function function, int valueType, boolean percent, ParaProfTrial ppTrial) {
//        double maxValue = 0;
//        switch (valueType) {
//        case 2:
//            if (percent) {
//                maxValue = function.getMaxExclusivePercent(ppTrial.getSelectedMetricID());
//            } else {
//                maxValue = function.getMaxExclusive(ppTrial.getSelectedMetricID());
//            }
//            break;
//        case 4:
//            if (percent) {
//                maxValue = function.getMaxInclusivePercent(ppTrial.getSelectedMetricID());
//            } else {
//                maxValue = function.getMaxInclusive(ppTrial.getSelectedMetricID());
//            }
//            break;
//        case 6:
//            maxValue = function.getMaxNumCalls();
//            break;
//        case 8:
//            maxValue = function.getMaxNumSubr();
//            break;
//        case 10:
//            maxValue = function.getMaxInclusivePerCall(ppTrial.getSelectedMetricID());
//            break;
//        default:
//            throw new ParaProfException("Invalid Value Type: " + valueType);
//        }
//        return maxValue;
//    }

//    public static double getMaxThreadValue(edu.uoregon.tau.dms.dss.Thread thread, int valueType,
//            boolean percent, ParaProfTrial ppTrial) {
//        double maxValue = 0;
//        switch (valueType) {
//        case 2:
//            if (percent)
//                maxValue = thread.getMaxExclusivePercent(ppTrial.getSelectedMetricID());
//            else
//                maxValue = thread.getMaxExclusive(ppTrial.getSelectedMetricID());
//            break;
//        case 4:
//            if (percent)
//                maxValue = thread.getMaxInclusivePercent(ppTrial.getSelectedMetricID());
//            else
//                maxValue = thread.getMaxInclusive(ppTrial.getSelectedMetricID());
//            break;
//        case 6:
//            maxValue = thread.getMaxNumCalls();
//            break;
//        case 8:
//            maxValue = thread.getMaxNumSubr();
//            break;
//        case 10:
//            maxValue = thread.getMaxInclusivePerCall(ppTrial.getSelectedMetricID());
//            break;
//        default:
//            throw new ParaProfException("Invalid Value Type: " + valueType);
//        }
//        return maxValue;
//    }

}
