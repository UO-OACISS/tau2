/*
 * PhaseConverter.java
 *
 * Copyright 2005                                                 
 * Department of Computer and Information Science, University of Oregon
 */
package edu.uoregon.tau.perfdmf.loader;

import jargs.gnu.CmdLineParser;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

import edu.uoregon.tau.perfdmf.*;

/**
 * Creates a phase profile from a callpath profile.
 * 
 * This class should be used to create a phase profile from a full callpath profile.
 * The resulting profile is what would have resulted if phase profiling had been used
 * with the given list of functions as the phases.
 * 
 * In addition the top level functions (those on the leftmost side) are always phases.
 *
 * <P>CVS $Id: PhaseConverter.java,v 1.2 2006/03/15 19:30:05 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.2 $
 */
public class PhaseConverter {

    public static void usage() {
        System.err.println("Usage: phaseconvert [options] <phase list file> <files>\n\n"
                + "try `phaseconvert --help' for more information");
    }

    public static void outputHelp() {
        System.err.println("Usage: phaseconvert [-i] [-f <filetype>] <phase list file> [profile files]\n\n"
                + "Converts a callpath profile into a phase profile\n\n" 
                + "The resulting profiles are written to a directory called \"converted\"\n"
                + "Options:\n\n"
                + "  -f, --filetype <filetype>      Specify type of performance data, options are:\n"
                + "                                   profiles (default), pprof, dynaprof, mpip,\n"
                + "                                   gprof, psrun, hpm, packed, cube, hpc\n"
                + "  -i, --fixnames                 Use the fixnames option for gprof\n\n" + "Notes:\n"
                + "  For the TAU profiles type, you can specify either a specific set of profile\n"
                + "files on the commandline, or you can specify a directory (by default the current\n"
                + "directory).  The specified directory will be searched for profile.*.*.* files,\n"
                + "or, in the case of multiple counters, directories named MULTI_* containing\n" + "profile data.\n\n"
                + "Examples:\n\n" + "  phaseconvert phases.txt\n"
                + "    This will load profile.* (or multiple counters directories MULTI_*) and\n"
                + "    create a phase profile with the phases given in phases.txt\n" + "  phaseconvert phases.txt profile.ppk\n"
                + "    This will convert the profile in profile.ppk to a phase profile\n");

    }

    static public void main(String[] args) {
        CmdLineParser parser = new CmdLineParser();
        CmdLineParser.Option helpOpt = parser.addBooleanOption('h', "help");
        CmdLineParser.Option typeOpt = parser.addStringOption('f', "filetype");
        CmdLineParser.Option fixOpt = parser.addBooleanOption('i', "fixnames");

        try {
            parser.parse(args);
        } catch (CmdLineParser.OptionException e) {
            System.err.println(e.getMessage());
            usage();
            System.exit(-1);
        }

        Boolean help = (Boolean) parser.getOptionValue(helpOpt);
        String fileTypeString = (String) parser.getOptionValue(typeOpt);
        Boolean fixNames = (Boolean) parser.getOptionValue(fixOpt);

        if (help != null && help.booleanValue()) {
            outputHelp();
            System.exit(-1);
        }

        String sourceFiles[] = parser.getRemainingArgs();

        if (sourceFiles.length < 1) {
            usage();
            System.exit(-1);
        }

        // Read in the list of phases
        File phaseListFile = new File(sourceFiles[0]);

        FileInputStream fis;
        try {
            fis = new FileInputStream(phaseListFile);
        } catch (FileNotFoundException fnfe) {
            System.err.println("File not found: " + phaseListFile);
            System.exit(-1);
            return;
        }
        
        InputStreamReader inReader = new InputStreamReader(fis);
        BufferedReader br = new BufferedReader(inReader);
        List<String> phases = new ArrayList<String>();
        try {
            String line = br.readLine();
            while (line != null) {
                phases.add(line.trim());
                line = br.readLine();
            }
        } catch (IOException ioe) {
            ioe.printStackTrace();
            System.exit(-1);
            return;
        }

        // handle the input profile
        int fileType = 0;
        String filePrefix = null;
        if (fileTypeString != null) {
            if (fileTypeString.equals("profiles")) {
                fileType = 0;
            } else if (fileTypeString.equals("pprof")) {
                fileType = 1;
            } else if (fileTypeString.equals("dynaprof")) {
                fileType = 2;
            } else if (fileTypeString.equals("mpip")) {
                fileType = 3;
            } else if (fileTypeString.equals("hpm")) {
                fileType = 4;
            } else if (fileTypeString.equals("gprof")) {
                fileType = 5;
            } else if (fileTypeString.equals("psrun")) {
                fileType = 6;
            } else if (fileTypeString.equals("packed")) {
                fileType = 7;
            } else if (fileTypeString.equals("cube")) {
                fileType = 8;
            } else if (fileTypeString.equals("hpc")) {
                fileType = 9;
            } else if (fileTypeString.equals("gyro")) {
                fileType = 100;
            } else {
                System.err.println("Please enter a valid file type.");
                LoadTrial.usage();
                System.exit(-1);
            }
        } else {
            if (sourceFiles.length == 2) {
                String filename = sourceFiles[1];
                if (filename.endsWith(".ppk")) {
                    fileType = 7;
                }
                if (filename.endsWith(".cube")) {
                    fileType = 8;
                }
            }
        }

        File[] files = new File[sourceFiles.length - 1];
        for (int i = 1; i < sourceFiles.length; i++) { // start from 1 since the first file is the phase list
            files[i - 1] = new File(sourceFiles[i]);
        }

        DataSource dataSource;
        try {
            System.out.println("Loading Profile...");
            dataSource = UtilFncs.initializeDataSource(files, fileType, fixNames != null && fixNames.booleanValue());
            dataSource.load();
        } catch (Exception e) {
            if (files == null || files.length != 0) { // We don't output an error message if paraprof was just invoked with no parameters.
                e.printStackTrace();
            }
            System.exit(-1);
            return;
        }

        System.out.println("Converting...");
        DataSource phaseDataSource = new PhaseConvertedDataSource(dataSource, phases);

        String name = "converted";
        File directory = new File(name);

        try {

            if (!directory.exists()) {
                boolean success = (new File(name).mkdir());
                if (!success) {
                    System.err.println("Failed to create directory: " + name);
                    System.exit(-1);
                }
            }
            System.out.println("Writing profiles to 'converted'");
            DataSourceExport.writeProfiles(phaseDataSource, new File(name));
        } catch (IOException ioe) {
            ioe.printStackTrace();
            System.exit(-1);
        }

    }

}
