/*
 * Name: UtilFncs.java 
 * Author: Robert Bell 
 * Description: Some useful functions for the system.
 */

package edu.uoregon.tau.dms.dss;

import java.util.*;
import java.awt.*;
import javax.swing.*;
import java.io.*;
import java.text.*;

public class UtilFncs {

    // left pad : pad string 's' up to length plen, but put the whitespace on
    // the left
    public static String lpad(String s, int plen) {
        int len = plen - s.length();
        if (len <= 0)
            return s;
        char padchars[] = new char[len];
        for (int i = 0; i < len; i++)
            padchars[i] = ' ';
        String str = new String(padchars, 0, len);
        return str.concat(s);
    }

    // pad : pad string 's' up to length plen
    public static String pad(String s, int plen) {
        int len = plen - s.length();
        if (len <= 0)
            return s;
        char padchars[] = new char[len];
        for (int i = 0; i < len; i++)
            padchars[i] = ' ';
        return s.concat(new String(padchars, 0, len));
    }

    //    public static double adjustDoublePresision(double d, int precision) {
    //        String result = null;
    //        try {
    //            String formatString = "#.#";
    //            for (int i = 0; i < (precision - 1); i++) {
    //                formatString = formatString + "#";
    //            }
    //            // 	    if(d < 0.001){
    //            // 		for(int i=0;i<4;i++){
    //            // 		    formatString = formatString+"0";
    //            // 		}
    //            // 	    }
    //
    //            formatString = formatString + "E0";
    //
    //            DecimalFormat dF = new DecimalFormat(formatString);
    //            result = dF.format(d);
    //        } catch (Exception e) {
    //            UtilFncs.systemError(e, null, "UF01");
    //        }
    //
    //        try {
    //            return Double.parseDouble(result);
    //
    //        } catch (java.lang.NumberFormatException e) {
    //            //	    System.out.println ("Uh oh! " + d);
    //            return d;
    //        }
    //    }

    // format a double for display within 'width' chars, kind of like C-printf's %G
    public static String formatDouble(double d, int width) {

        // first check if the regular toString is in exponential form
        boolean exp = false;
        String str = Double.toString(d);
        for (int i = 0; i < str.length(); i++) {
            if (str.charAt(i) == 'E') {
                exp = true;
                break;
            }
        }

        if (!exp) {
            // not exponential form
            String formatString = "";

            // create a format string of the same length, (e.g. ###.### for 123.456)
            for (int i = 0; i < str.length(); i++) {
                if (str.charAt(i) != '.')
                    formatString = formatString + "#";
                else
                    formatString = formatString + ".";
            }

            // now we reduce that format string as follows

            // first, do the minimum of 'width' or the length of the regular toString

            int min = width;
            if (formatString.length() < min)
                min = formatString.length();

            // we don't want more than 4 digits past the decimal point
            // this 4 would be the old ParaProf.defaultNumberPrecision
            if (formatString.indexOf('.') + 4 < min)
                min = formatString.indexOf('.') + 4;

            formatString = formatString.substring(0, min);

            // remove trailing dot
            if (formatString.charAt(formatString.length() - 1) == '.')
                formatString = formatString.substring(0, formatString.length() - 2);

            DecimalFormat dF = new DecimalFormat(formatString);

            str = dF.format(d);
            //System.out.println ("value: " + d + ", width: " + width + ", returning: '" + lpad(str,width) + "'");
            return lpad(str, width);

        }

        // toString used exponential form, so we ought to also

        // we want up to four significant digits
        String formatString = "0.0###";

        formatString = formatString + "E0";
        DecimalFormat dF = new DecimalFormat(formatString);

        str = dF.format(d);
        return lpad(str, width);
    }

    //This method is used in a number of windows to determine the actual output
    // string
    //displayed. Current types are:
    //0 - microseconds
    //1 - milliseconds
    //2 - seconds
    //3 - hr:min:sec
    //At present, the passed in double value is assumed to be in microseconds.
    public static String getOutputString(int type, double d, int width) {
        switch (type) {
        case 0:
            return (UtilFncs.formatDouble(d, width));
        case 1:
            return (UtilFncs.formatDouble((d / 1000), width));
        case 2:
            return (UtilFncs.formatDouble((d / 1000000), width));
        case 3:
            int hr = 0;
            int min = 0;
            hr = (int) (d / 3600000000.00);
            //Calculate the number of microseconds left after hours are
            // subtracted.
            d = d - hr * 3600000000.00;
            min = (int) (d / 60000000.00);
            //Calculate the number of microseconds left after minutes are
            // subtracted.
            d = d - min * 60000000.00;

            String hours = Integer.toString(hr);
            String mins = Integer.toString(min);

            String secs = formatDouble(d / 1000000, 7);

            // remove the whitespace
            int idx = 0;
            for (int i = 0; i < secs.length(); i++) {
                if (secs.charAt(i) != ' ') {
                    idx = i;
                    break;
                }
            }
            secs = secs.substring(idx);

            return lpad(hours + ":" + mins + ":" + secs, width);

        default:
            throw new RuntimeException("Unexpected string type: " + type);
        }
    }

    public static String getUnitsString(int type, boolean time, boolean derived) {
        if (derived) {
            if (!time)
                return "counts";
            switch (type) {
            case 0:
                return "Derived metric shown in microseconds format";
            case 1:
                return "Derived metric shown in milliseconds format";
            case 2:
                return "Derived metric shown in seconds format";
            case 3:
                return "Derived metric shown in hour:minute:seconds format";
            default:
                throw new RuntimeException("Unexpected string type: " + type);

            }
        } else {
            if (!time)
                return "counts";
            switch (type) {
            case 0:
                return "microseconds";
            case 1:
                return "milliseconds";
            case 2:
                return "seconds";
            case 3:
                return "hour:minute:seconds";
            default:
                throw new RuntimeException("Unexpected string type: " + type);
            }
        }
    }

    public static String getValueTypeString(int type) {
        switch (type) {
        case 2:
            return "exclusive";
        case 4:
            return "inclusive";
        case 6:
            return "number of calls";
        case 8:
            return "number of subroutines";
        case 10:
            return "per call value";
        case 12:
            return "number of userEvents";
        case 14:
            return "minimum number of userEvents";
        case 16:
            return "maximum number of userEvents";
        case 18:
            return "mean number of userEvents";
        case 20:
            return "Standard Deviation of User Event Value";
        default:
            throw new RuntimeException("Unexpected string type: " + type);
        }
    }

    //    public static int exists(int[] ref, int i) {
    //        if (ref == null)
    //            return -1;
    //        int test = ref.length;
    //        for (int j = 0; j < test; j++) {
    //            if (ref[j] == i)
    //                return j;
    //        }
    //        return -1;
    //    }

    public static int exists(Vector ref, int i) {
        //Assuming a vector of Integers.
        if (ref == null)
            return -1;
        Integer current = null;
        int test = ref.size();
        for (int j = 0; j < test; j++) {
            current = (Integer) ref.elementAt(j);
            if ((current.intValue()) == i)
                return j;
        }
        return -1;
    }

    //####################################
    //Error handling.
    //####################################
    //public static boolean debug = false;

//    public static void systemError(Object obj, Component component, String string) {
//
//        System.out.println("####################################");
//        boolean quit = true; //Quit by default.
//        if (obj != null) {
//            if (obj instanceof Exception) {
//                Exception exception = (Exception) obj;
//
//                if (UtilFncs.debug) {
//                    System.out.println(exception.toString());
//                    exception.printStackTrace();
//                    System.out.println("\n");
//                }
//                exception.printStackTrace();
//                System.out.println("An error was detected: " + string);
//                System.out.println(ParaProfError.contactString);
//            }
//            if (obj instanceof ParaProfError) {
//                ParaProfError paraProfError = (ParaProfError) obj;
//
//                if (paraProfError.exp != null) {
//                    System.out.println(paraProfError.exp.toString());
//                    paraProfError.exp.printStackTrace();
//                    System.out.println("\n");
//                }
//
//                if (UtilFncs.debug) {
//                    if ((paraProfError.showPopup) && (paraProfError.popupString != null))
//                        JOptionPane.showMessageDialog(paraProfError.component, "ParaProf Error",
//                                paraProfError.popupString, JOptionPane.ERROR_MESSAGE);
//                    if (paraProfError.exp != null) {
//                        System.out.println(paraProfError.exp.toString());
//                        paraProfError.exp.printStackTrace();
//                        System.out.println("\n");
//                    }
//                    if (paraProfError.location != null)
//                        System.out.println("Location: " + paraProfError.location);
//                    if (paraProfError.s0 != null)
//                        System.out.println(paraProfError.s0);
//                    if (paraProfError.s1 != null)
//                        System.out.println(paraProfError.s1);
//                    if (paraProfError.showContactString)
//                        System.out.println(ParaProfError.contactString);
//                } else {
//                    if ((paraProfError.showPopup) && (paraProfError.popupString != null))
//                        JOptionPane.showMessageDialog(paraProfError.component, paraProfError.popupString,
//                                "ParaProf Error", JOptionPane.ERROR_MESSAGE);
//                    if (paraProfError.location != null)
//                        System.out.println("Location: " + paraProfError.location);
//                    if (paraProfError.s0 != null)
//                        System.out.println(paraProfError.s0);
//                    if (paraProfError.s1 != null)
//                        System.out.println(paraProfError.s1);
//                    if (paraProfError.showContactString)
//                        System.out.println(ParaProfError.contactString);
//                }
//                quit = paraProfError.quit;
//            } else {
//                System.out.println("An error has been detected: " + string);
//            }
//        } else {
//            System.out.println("An error was detected at " + string);
//        }
//        System.out.println("####################################");
//        if (quit)
//            System.exit(0);
//    }

    //####################################
    //End - Error handling.
    //####################################

    //####################################
    //Test system state functionProfiles.
    //These functionProfiles are used to test
    //the current state of the system.
    //####################################

    //Print the passed in data session data out to a file or to the console.
    //If the passed in File object is null, data is printed to the console.
    // Component can be null.

    //    public static void outputData(DataSource dataSource, File file, Component component) {
    //        try {
    //            boolean toFile = false;
    //            PrintWriter out = null;
    //            TrialData trialData = dataSource.getTrialData();
    //            int numberOfMetrics = dataSource.getNumberOfMetrics();
    //            StringBuffer output = new StringBuffer(1000);
    //
    //            if (file != null) {
    //                out = new PrintWriter(new FileWriter(file));
    //                toFile = true;
    //            }
    //
    //            //######
    //            //Metric data.
    //            //######
    //            if (toFile) {
    //                out.println("<metrics>");
    //                out.println("<numofmetrics>" + numberOfMetrics + "</numofmetrics>");
    //            } else {
    //                System.out.println("<metrics>");
    //                System.out.println("<numofmetrics>" + numberOfMetrics + "</numofmetrics>");
    //            }
    //            for (int metric = 0; metric < numberOfMetrics; metric++) {
    //                if (toFile)
    //                    out.println(dataSource.getMetricName(metric));
    //                else
    //                    System.out.println(dataSource.getMetricName(metric));
    //            }
    //            if (toFile)
    //                out.println("</metrics>");
    //            else
    //                System.out.println("</metrics>");
    //            //######
    //            //End - Metric data.
    //            //######
    //
    //            //####################################
    //            //Global Data.
    //            //####################################
    //
    //            //######
    //            //Function data.
    //            //######
    //            
    //            Vector list = new Vector();
    //            for (Iterator functionIterator = trialData.getFunctions(); functionIterator.hasNext();) {
    //                list.add(functionIterator.next());
    //            }
    //
    //            
    //             //Name to ID map.
    //            if (toFile) {
    //                out.println("<funnameidmap>");
    //                out.println("<numoffunctions>" + list.size() + "</numoffunctions>");
    //            } else {
    //                System.out.println("<funnameidmap>");
    //                System.out.println("<numoffunctions>" + list.size() + "</numoffunctions>");
    //            }
    //            
    //            for (Enumeration e = list.elements(); e.hasMoreElements();) {
    //                Function f = (Function) e.nextElement();
    //                if (toFile)
    //                    out.println("\"" + f.getName() + "\""
    //                            + f.getID());
    //                else
    //                    System.out.println("\"" + f.getName() + "\""
    //                            + f.getID());
    //            }
    //            if (toFile)
    //                out.println("</funnameidmap>");
    //            else
    //                System.out.println("</funnameidmap>");
    //
    //            if (toFile)
    //                out.println("id mincl(..) mexcl(..) minclp(..) mexclp(..) museccall(..) mnoc mnos");
    //            else
    //                System.out.println("id mincl(..) mexcl(..) minclp(..) mexclp(..) museccall(..) mnoc mnos");
    //            for (Enumeration e = list.elements(); e.hasMoreElements();) {
    //                output.delete(0, output.length());
    //                Function function = (Function) e.nextElement();
    //                output.append(function.getID() + " ");
    //                for (int metric = 0; metric < numberOfMetrics; metric++) {
    //                    output.append(function.getMeanInclusiveValue(metric) + " ");
    //                    output.append(function.getMeanExclusiveValue(metric) + " ");
    //                    output.append(function.getMeanInclusivePercentValue(metric) + " ");
    //                    output.append(function.getMeanExclusivePercentValue(metric) + " ");
    //                    output.append(function.getMeanUserSecPerCall(metric) + " ");
    //                }
    //                output.append(function.getMeanNumberOfCalls() + " ");
    //                output.append(function.getMeanNumberOfSubRoutines() + "");
    //                if (toFile)
    //                    out.println(output);
    //                else
    //                    System.out.println(output);
    //            }
    //            //######
    //            //End - Function data.
    //            //######
    //
    //            //######
    //            //User event data.
    //            //######
    //            
    //            
    //            
    //            
    //
    //            //list = trialData.getMapping(2);
    //
    //            list = null;
    //            
    //            
    //            //Name to ID map.
    //            if (toFile) {
    //                out.println("<usereventnameidmap>");
    //                out.println("<numofuserevents>" + list.size() + "</numofuserevents>");
    //            } else {
    //                System.out.println("<usereventnameidmap>");
    //                System.out.println("<numofuserevents>" + list.size() + "</numofuserevents>");
    //            }
    //            for (Enumeration e = list.elements(); e.hasMoreElements();) {
    //                Function function = (Function) e.nextElement();
    //                if (toFile)
    //                    out.println("\"" + function.getName() + "\""
    //                            + function.getID());
    //                else
    //                    System.out.println("\"" + function.getName() + "\""
    //                            + function.getID());
    //            }
    //            if (toFile)
    //                out.println("</usereventnameidmap>");
    //            else
    //                System.out.println("</usereventnameidmap>");
    //            //######
    //            //End - User event data.
    //            //######
    //
    //            //####################################
    //            //End - Global Data.
    //            //####################################
    //
    //            //######
    //            //Thread data.
    //            //######
    //            if (toFile) {
    //                out.println("<threaddata>");
    //                out.println("funid incl(..) excl(..) inclp(..) exclp(..) useccall(..) mnoc mnos");
    //                out.println("usereventid num min max mean");
    //                out.println("<numofthreads>" + dataSource.getNCT().getTotalNumberOfThreads()
    //                        + "</numofthreads>");
    //            } else {
    //                System.out.println("<threaddata>");
    //                System.out.println("id incl(..) excl(..) inclp(..) exclp(..) useccall(..) noc nos");
    //                System.out.println("usereventid num min max mean");
    //                System.out.println("<numofthreads>" + dataSource.getNCT().getTotalNumberOfThreads()
    //                        + "</numofthreads>");
    //            }
    //
    //            String test = null;
    //
    //            for (Enumeration e1 = dataSource.getNCT().getNodes().elements(); e1.hasMoreElements();) {
    //                Node node = (Node) e1.nextElement();
    //                for (Enumeration e2 = node.getContexts().elements(); e2.hasMoreElements();) {
    //                    Context context = (Context) e2.nextElement();
    //                    for (Enumeration e3 = context.getThreads().elements(); e3.hasMoreElements();) {
    //                        Thread thread = (Thread) e3.nextElement();
    //                        ListIterator l = null;
    //                        if (toFile)
    //                            out.println("<thread>" + thread.getNodeID() + ","
    //                                    + thread.getContextID() + "," + thread.getThreadID()
    //                                    + "</thread");
    //                        else
    //                            System.out.println("<thread>" + thread.getNodeID() + ","
    //                                    + thread.getContextID() + "," + thread.getThreadID()
    //                                    + "</thread");
    //                        if (toFile)
    //                            out.println("<functiondata>");
    //                        else
    //                            System.out.println("<functiondata>");
    //                        l = thread.getFunctionListIterator();
    //                        while (l.hasNext()) {
    //                            output.delete(0, output.length());
    //                            FunctionProfile functionProfile = (FunctionProfile) l.next();
    //                            if (functionProfile != null) {
    //                                output.append(functionProfile.getFunction().getID() + " ");
    //                                for (int metric = 0; metric < numberOfMetrics; metric++) {
    //                                    output.append(functionProfile.getInclusiveValue(metric) + " ");
    //                                    output.append(functionProfile.getExclusiveValue(metric) + " ");
    //                                    output.append(functionProfile.getInclusivePercentValue(metric)
    //                                            + " ");
    //                                    output.append(functionProfile.getExclusivePercentValue(metric)
    //                                            + " ");
    //                                    output.append(functionProfile.getInclusivePerCall(metric) + " ");
    //                                }
    //                                output.append(functionProfile.getNumberOfCalls() + " ");
    //                                output.append(functionProfile.getNumberOfSubRoutines() + "");
    //                                if (toFile)
    //                                    out.println(output);
    //                                else
    //                                    System.out.println(output);
    //                            }
    //                        }
    //                        if (toFile)
    //                            out.println("</functiondata>");
    //                        else
    //                            System.out.println("</functiondata>");
    //                        if (toFile)
    //                            out.println("<usereventdata>");
    //                        else
    //                            System.out.println("<usereventdata>");
    //                        l = thread.getUsereventListIterator();
    //                        while (l.hasNext()) {
    //                            output.delete(0, output.length());
    //                            UserEventProfile uep = (UserEventProfile) l.next();
    //                            if (uep != null) {
    //                                output.append(uep.getUserEvent().getID() + " ");
    //                                for (int metric = 0; metric < numberOfMetrics; metric++) {
    //                                    output.append(uep.getUserEventNumberValue() + " ");
    //                                    output.append(uep.getUserEventMinValue() + " ");
    //                                    output.append(uep.getUserEventMaxValue() + " ");
    //                                    output.append(uep.getUserEventMeanValue() + "");
    //                                }
    //                                if (toFile)
    //                                    out.println(output);
    //                                else
    //                                    System.out.println(output);
    //                            }
    //                        }
    //                        if (toFile)
    //                            out.println("</usereventdata>");
    //                        else
    //                            System.out.println("</usereventdata>");
    //                    }
    //                }
    //            }
    //
    //            if (toFile)
    //                out.println("</threaddata>");
    //            else
    //                System.out.println("</threaddata>");
    //            //######
    //            //End - Thread data.
    //            //######
    //
    //            //Flush output buffer and close file if required.
    //            if (out != null) {
    //                out.flush();
    //                out.close();
    //            }
    //
    //        } catch (Exception exception) {
    //            UtilFncs.systemError(new ParaProfError("UF05",
    //                    "File write error! Check console for details.",
    //                    "An error occurred while trying to save txt file.", null, exception,
    //                    component, null, null, true, false, false), null, null);
    //        }
    //    }
    //    //####################################
    //    //End - Test system state functionProfiles.
    //    //####################################
    //    

    public static DataSource initializeDataSource(File[] sourceFiles, int fileType, boolean fixGprofNames)
            throws DataSourceException {
        DataSource dataSource = null;

        Vector v = new Vector();
        File filelist[];
        switch (fileType) {
        case 0: // TAU Profiles
            FileList fl = new FileList();

            if (sourceFiles.length < 1) {
                v = fl.helperFindProfiles(System.getProperty("user.dir"));
                if (v.size() == 0)
                    throw new DataSourceException("profiles type: no profiles specified");
            } else {
                if (sourceFiles[0].isDirectory()) {
                    if (sourceFiles.length > 1) {
                        throw new DataSourceException("profiles type: you can only specify one directory");
                    }

                    v = fl.helperFindProfiles(sourceFiles[0].toString());
                    if (v.size() == 0) {
                        throw new DataSourceException("No profiles found in directory: " + sourceFiles[0]);
                    }
                } else {
                    v.add(sourceFiles);
                }

            }

            dataSource = new TauDataSource(v);
            break;
        case 1:
            if (sourceFiles.length != 1) {
                throw new DataSourceException("pprof type: you must specify exactly one file");
            }
            if (sourceFiles[0].isDirectory()) {
                throw new DataSourceException("pprof type: you must specify a file, not a directory");
            }
            v.add(sourceFiles);
            dataSource = new TauPprofDataSource(v);
            break;
        case 2:
            dataSource = new DynaprofDataSource(sourceFiles);
            break;
        case 3:

            if (sourceFiles.length != 1) {
                throw new DataSourceException("MpiP type: you must specify exactly one file");
            }
            if (sourceFiles[0].isDirectory()) {
                throw new DataSourceException("MpiP type: you must specify a file, not a directory");
            }
            v.add(sourceFiles);
            dataSource = new MpiPDataSource(v);
            break;
        case 4:
            v.add(sourceFiles);
            dataSource = new HPMToolkitDataSource(v);
            break;
        case 5:
            dataSource = new GprofDataSource(sourceFiles, fixGprofNames);
            break;
        case 6:
            v.add(sourceFiles);
            dataSource = new PSRunDataSource(v);
            break;
        case 7:
            if (sourceFiles.length != 1) {
                throw new DataSourceException("Packed Profile type: you must specify exactly one file");
            }
            if (sourceFiles[0].isDirectory()) {
                throw new DataSourceException("Packed Profile type: you must specify a file, not a directory");
            }
            dataSource = new PackedProfileDataSource(sourceFiles[0]);
            break;
        default:
            break;
        }

        return dataSource;
    }

}