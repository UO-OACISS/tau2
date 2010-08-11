package edu.uoregon.tau.perfdmf.database;

import java.io.File;
import java.util.List;
import java.util.Vector;

/**
 * Simple Class to retrieve database configurations stored on the file system.
 * 
 * @author scottb
 *
 */

abstract public class ConfigureFiles {

    /* Returns a List of ParseConfigs. */
    public static List<ParseConfig> getConfigurations() {
        List<File> files = getConfigurationFiles();
        List<ParseConfig> configs = new Vector<ParseConfig>();
        for (int i = 0; i < files.size(); i++) {
            configs.add(new ParseConfig(files.get(i).getAbsolutePath()));
        }
        return configs;
    }

    /* Returns a List of Files. */
    public static List<File> getConfigurationFiles() {
        List<String> names = getConfigurationNames();
        List<File> files = new Vector<File>();
        for (int i = 0; i < names.size(); i++) {
            if (names.get(i).compareTo("Default") == 0)
                files.add(new File(System.getProperty("user.home") + File.separator + ".ParaProf" + File.separator
                        + "perfdmf.cfg"));
            else
                files.add(new File(System.getProperty("user.home") + File.separator + ".ParaProf" + File.separator
                        + "perfdmf.cfg." + names.get(i)));
        }
        return files;
    }

    /* Returns a List of Strings. */
    public static List<String> getConfigurationNames() {
        File paraprofDirectory = new File(System.getProperty("user.home") + File.separator + ".ParaProf");
        String[] fileNames = paraprofDirectory.list();
        List<String> perfdmfConfigs = new Vector<String>();
        if (fileNames == null) {
            return perfdmfConfigs;
        }
        for (int i = 0; i < fileNames.length; i++) {

            if (fileNames[i].compareTo("perfdmf.cfg") == 0) {
                perfdmfConfigs.add("Default");
            } else if (fileNames[i].startsWith("perfdmf.cfg") && !fileNames[i].endsWith("~")) {
                String name = fileNames[i].substring(12);
                perfdmfConfigs.add(name);
            }
        }
        return perfdmfConfigs;
    }

    /* testing only */
    public static void main(String[] args) {
        System.out.println("Names: ");
        System.out.println(getConfigurationFiles());
    }
}