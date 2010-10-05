/*
 * Name: UtilFncs.java 
 * Author: Robert Bell 
 * Description: Some useful functions for the system.
 */

package edu.uoregon.tau.perfdmf;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;
import java.util.Vector;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

public class UtilFncs {

    public static class EmptyIterator implements ListIterator<FunctionProfile>, Iterator<FunctionProfile> {

        public int nextIndex() {
            return 0;
        }

        public int previousIndex() {
            return 0;
        }

        public void remove() {}

        public boolean hasNext() {
            return false;
        }

        public boolean hasPrevious() {
            return false;
        }

        public FunctionProfile next() {
            return null;
        }

        public FunctionProfile previous() {
            return null;
        }

        public void add(FunctionProfile o) {}

        public void set(FunctionProfile o) {}
    }

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
    public static String formatDouble(double d, int width, boolean pad) {

        // first check if the regular toString is in exponential form
        boolean exp = false;
        String str;
        
        if(d<0){
        	str="-";
        	 if (pad) {
                 return lpad(str, width);
             } else {
                 return str;
             }
        }
        	str= Double.toString(d);
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
                if (str.charAt(i) != '.') {
                    formatString = formatString + "#";
                } else {
                    formatString = formatString + ".";
                }
            }

            //            DecimalFormat bil = new DecimalFormat("#,###,###,##0");
            //            DecimalFormat mil = new DecimalFormat("#,###,##0");
            //            DecimalFormat thou = new DecimalFormat("#,##0");

            // now we reduce that format string as follows

            // first, do the minimum of 'width' or the length of the regular toString

            int min = width;
            if (formatString.length() < min) {
                min = formatString.length();
            }

            // we don't want more than 4 digits past the decimal point
            // this 4 would be the old ParaProf.defaultNumberPrecision
            if (formatString.indexOf('.') + 4 < formatString.length()) {
                min = formatString.indexOf('.') + 4;
            }

            formatString = formatString.substring(0, min);

            // remove trailing dot
            if (formatString.charAt(formatString.length() - 1) == '.')
                formatString = formatString.substring(0, formatString.length() - 2);

            DecimalFormat dF = new DecimalFormat(formatString);

            str = dF.format(d);
            //System.out.println("value: " + d + ", width: " + width + ", returning: '" + lpad(str, width) + "'");
            if (pad) {
                return lpad(str, width);
            } else {
                return str;
            }

        }

        // toString used exponential form, so we ought to also

        String formatString;
        if (d < 0.1) {
            formatString = "0.0";
        } else {
            // we want up to four significant digits
            formatString = "0.0###";
        }

        formatString = formatString + "E0";
        DecimalFormat dF = new DecimalFormat(formatString);

        str = dF.format(d);
        if (pad) {
            return lpad(str, width);
        } else {
            return str;
        }
    }

    //This method is used in a number of windows to determine the actual output
    // string
    //displayed. Current types are:
    //0 - microseconds
    //1 - milliseconds
    //2 - seconds
    //3 - hr:min:sec
    //At present, the passed in double value is assumed to be in microseconds.
    public static String getOutputString(int type, double d, int width, boolean timeDenominator) {
        switch (type) {
        case 0:
            return (UtilFncs.formatDouble(d, width, true));
        case 1:
            if (timeDenominator) {
                return (UtilFncs.formatDouble((d * 1000), width, true));
            } else {
                return (UtilFncs.formatDouble((d / 1000), width, true));
            }
        case 2:
            if (timeDenominator) {
                return (UtilFncs.formatDouble((d * 1000000), width, true));
            } else {
                return (UtilFncs.formatDouble((d / 1000000), width, true));
            }
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

            String mins;
            if (min < 10) {
                mins = "0" + Integer.toString(min);
            } else {
                mins = Integer.toString(min);
            }

            String secs;

            if (hr >= 1 || min >= 1) {
                // don't show fractional seconds if there is at least a minute
                secs = formatDouble(d / 1000000, 1, false);
            } else {
                secs = formatDouble(d / 1000000, 3, false);
            }

            if (secs.indexOf('E') != -1) { // never show exponential notation for hh:mm:ss
                secs = "00";
            }
            if (secs.length() == 1) {
                secs = "0" + secs;
            }
            //System.out.println("secs = " + (d / 1000000) + ", out = " + secs);
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

    //    public static String getValueTypeString(int type) {
    //        switch (type) {
    //        case 2:
    //            return "exclusive";
    //        case 4:
    //            return "inclusive";
    //        case 6:
    //            return "number of calls";
    //        case 8:
    //            return "number of subroutines";
    //        case 10:
    //            return "per call value";
    //        case 12:
    //            return "number of userEvents";
    //        case 14:
    //            return "minimum number of userEvents";
    //        case 16:
    //            return "maximum number of userEvents";
    //        case 18:
    //            return "mean number of userEvents";
    //        case 20:
    //            return "Standard Deviation of User Event Value";
    //        default:
    //            throw new RuntimeException("Unexpected string type: " + type);
    //        }
    //    }

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

    public static int exists(Vector<Integer> ref, int i) {
        //Assuming a vector of Integers.
        if (ref == null)
            return -1;
        Integer current = null;
        int test = ref.size();
        for (int j = 0; j < test; j++) {
            current = ref.elementAt(j);
            if ((current.intValue()) == i)
                return j;
        }
        return -1;
    }

    public static String getRightSide(String str) {
        int location = str.indexOf("=>");
        if (location >= 0) {
            return str.substring(location + 2).trim();
        } else {
            return str;
        }
    }

    public static String getRightMost(String str) {
        int location = str.lastIndexOf("=>");
        if (location >= 0) {
            return str.substring(location + 2).trim();
        } else {
            return str;
        }
    }

    public static String getAllButRightMost(String str) {
        int location = str.lastIndexOf("=>");
        if (location >= 0) {
            return str.substring(0, location).trim();
        } else {
            return str;
        }
    }

    public static String getLeftSide(String str) {
        int location = str.indexOf("=>");
        if (location >= 0) {
            return str.substring(0, location).trim();
        } else {
            return str;
        }
    }

    public static String getRevLeftSide(String str) {
        int location = str.indexOf("<=");
        if (location >= 0) {
            return str.substring(0, location).trim();
        } else {
            return str;
        }
    }

    public static String getContextEventRoot(String str) {
        int colon = str.indexOf(":");
        int location = str.indexOf("=>");
        if (location < 0) {
            if (colon < 0) {
                return str;
            } else {
                return str.substring(colon + 1);
            }
        }
        return str.substring(colon + 1, location).trim();
    }

    /**
     * Identifies profile format based on name (extension) and contents (first few lines)
     * 
     * @param file input file to identify
     * @return the format, from DataSource public static finals (e.g. DataSource.PPK)
     */
    public static int identifyData(File file) {

        // try to identify by file extension first
        String filename = file.getName();
        if (filename.toLowerCase().endsWith(".ppk")) {
            return DataSource.PPK;
        }
        if (filename.toLowerCase().endsWith(".cube")) {
            return DataSource.CUBE;
        }
        if (filename.toLowerCase().endsWith(".cube.gz")) {
            return DataSource.CUBE;
        }
        if (filename.toLowerCase().endsWith(".mpip")) {
            return DataSource.MPIP;
        }

        // try reading a few lines and try to determine what the format is
        try {

            FileInputStream fis = new FileInputStream(file);
            InputStream input;

            // see if it is gzip'd, if not, read directly
            try {
                GZIPInputStream gzip = new GZIPInputStream(fis);
                input = gzip;
            } catch (IOException ioe) {
                fis.close();
                input = new FileInputStream(file);
            }
            InputStreamReader isr = new InputStreamReader(input);
            BufferedReader br = new BufferedReader(isr);

            for (int i = 0; i < 10; i++) {
                String line = br.readLine();
                //System.out.println("line = " + line);
                if (line.startsWith("<profile_xml>")) {
                    fis.close();
                    return DataSource.SNAP;
                } else if (line.startsWith("<!DOCTYPE hwpc")) {
                    fis.close();
                    return DataSource.PSRUN;
                }
            }

            fis.close();
        } catch (Exception e) {
            // forget it, just go with the default
        }

        return DataSource.TAUPROFILE; // default
    }

    public static DataSource initializeDataSource(File[] sourceFiles, int fileType, boolean fixGprofNames){
    	return initializeDataSource(sourceFiles,fileType,fixGprofNames,null);
    }
    
    public static DataSource initializeDataSource(File[] sourceFiles, int fileType, boolean fixGprofNames, String range)
            throws DataSourceException {
        DataSource dataSource = null;

        List<File[]> v = new ArrayList<File[]>();
        //File filelist[];
        switch (fileType) {
        case DataSource.TAUPROFILE: // TAU Profiles
            FileList fl;
            if(range!=null)
            	fl= new FileList(range);
            else
            	fl = new FileList();

            if (sourceFiles.length < 1) {
                v = fl.helperFindProfiles(System.getProperty("user.dir"));
                if (v.size() == 0) {
                    throw new DataSourceException("profiles type: no profiles specified");
                }
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
        case DataSource.PPROF:
            if (sourceFiles.length != 1) {
                throw new DataSourceException("pprof type: you must specify exactly one file");
            }
            if (sourceFiles[0].isDirectory()) {
                throw new DataSourceException("pprof type: you must specify a file, not a directory");
            }
            v.add(sourceFiles);
            dataSource = new TauPprofDataSource(v);
            break;
        case DataSource.DYNAPROF:
            dataSource = new DynaprofDataSource(sourceFiles);
            break;
        case DataSource.MPIP:

            if (sourceFiles.length != 1) {
                throw new DataSourceException("MpiP type: you must specify exactly one file");
            }
            if (sourceFiles[0].isDirectory()) {
                throw new DataSourceException("MpiP type: you must specify a file, not a directory");
            }
            dataSource = new MpiPDataSource(sourceFiles[0]);
            break;
        case DataSource.HPM:
            v.add(sourceFiles);
            dataSource = new HPMToolkitDataSource(v);
            break;
        case DataSource.GPROF:
            dataSource = new GprofDataSource(sourceFiles, fixGprofNames);
            break;
        case DataSource.GOOGLE:
            dataSource = new GoogleDataSource(sourceFiles);
            break;
        case DataSource.PSRUN:
            v.add(sourceFiles);
            dataSource = new PSRunDataSource(v);
            break;
        case DataSource.PPK:
            if (sourceFiles.length != 1) {
                throw new DataSourceException("Packed Profile type: you must specify exactly one file");
            }
            if (sourceFiles[0].isDirectory()) {
                throw new DataSourceException("Packed Profile type: you must specify a file, not a directory");
            }
            dataSource = new PackedProfileDataSource(sourceFiles[0]);
            break;
        case DataSource.CUBE:
            if (sourceFiles.length != 1) {
                throw new DataSourceException("Cube type: you must specify exactly one file");
            }
            if (sourceFiles[0].isDirectory()) {
                throw new DataSourceException("Cube type: you must specify a file, not a directory");
            }

            try {
                //Class c = 
                	Class.forName("org.xml.sax.SAXException");
            } catch (ClassNotFoundException cnfe) {
                throw new DataSourceException("Sorry, cube format requires Java 1.4");
            }

            dataSource = new CubeDataSource(sourceFiles[0]);
            break;
        case DataSource.HPCTOOLKIT:
            if (sourceFiles.length != 1) {
                throw new DataSourceException("HPCToolkit type: you must specify exactly one file");
            }
            if (sourceFiles[0].isDirectory()) {
                throw new DataSourceException("HPCToolkit type: you must specify a file, not a directory");
            }

            try {
                //Class c = 
                	Class.forName("org.xml.sax.SAXException");
            } catch (ClassNotFoundException cnfe) {
                throw new DataSourceException("Sorry, HPCToolkit format requires Java 1.4");
            }

            dataSource = new HPCToolkitDataSource(sourceFiles[0]);
            break;

        case DataSource.SNAP:
            if (sourceFiles.length >= 1) {
                dataSource = new SnapshotDataSource(sourceFiles);
            } else {
                fl = new FileList();
                File[] list = fl.helperFindSnapshots(System.getProperty("user.dir"));
                if (list.length == 0) {
                    throw new DataSourceException("snapshot type: no snapshots found");
                }
                dataSource = new SnapshotDataSource(list);
            }
            break;

        case DataSource.OMPP:
            dataSource = new OmppDataSource(sourceFiles[0]);
            break;

        case DataSource.PERIXML:
            dataSource = new PeriXMLDataSource(sourceFiles[0]);
            break;

        case DataSource.GYRO:
            v.add(sourceFiles);
            dataSource = new GyroDataSource(v);
            break;

        case DataSource.GPTL:
            dataSource = new GPTLDataSource(sourceFiles[0]);
            break;

        case DataSource.IPM:
            dataSource = new IPMDataSource(sourceFiles[0]);
            break;

        case DataSource.PARAVER:
            dataSource = new ParaverDataSource(sourceFiles);
            break;

        case DataSource.GAMESS:
            dataSource = new GAMESSDataSource(sourceFiles[0]);
            break;

        default:
            throw new RuntimeException("Programming error: unknown format id = " + fileType);
        }

        dataSource.setFileType(fileType);
        return dataSource;
    }

    public static void mergeSnapshots(File[] sourceFiles, String outfile) throws Exception {
        FileOutputStream fos = new FileOutputStream(outfile);
        GZIPOutputStream gzipout = new GZIPOutputStream(fos);

        for (int i = 0; i < sourceFiles.length; i++) {
            FileInputStream fis = new FileInputStream(sourceFiles[i]);

            InputStream input;

            // see if it is gzip'd, if not, read directly
            try {
                GZIPInputStream gzip = new GZIPInputStream(fis);
                input = gzip;
            } catch (IOException ioe) {
                fis.close();
                input = new FileInputStream(sourceFiles[i]);
            }

            int count = 0;
            byte[] buffer = new byte[4096];
            while ((count = input.read(buffer)) >= 0) {
                gzipout.write(buffer, 0, count);
            }
            fis.close();
        }
        gzipout.close();
        fos.close();
    }
}
