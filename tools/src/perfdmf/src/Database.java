package edu.uoregon.tau.perfdmf;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import edu.uoregon.tau.perfdmf.database.ParseConfig;

public class Database {
    private String name;
    private int id;
    private ParseConfig config;

    private static int idCounter;

    public Database(String name, ParseConfig config) {
        this.name = name;
        this.id = idCounter;
        idCounter++;
        this.config = config;
    }

    public int getID() {
        return id;
    }

    public ParseConfig getConfig() {
        return config;
    }

    private static Database createDatabase(String name, String configFile) {
        ParseConfig config = new ParseConfig(configFile.toString());
        Database database = new Database(name, config);
        return database;
    }

    public static List getDatabases() {
        File paraprofDirectory = new File(System.getProperty("user.home") + "/.ParaProf");
        String[] fileNames = paraprofDirectory.list();
        List perfdmfConfigs = new ArrayList();
        for (int i = 0; i < fileNames.length; i++) {
            if (fileNames[i].compareTo("perfdmf.cfg") == 0) {
                perfdmfConfigs.add(createDatabase("Default", paraprofDirectory + "/" + fileNames[i]));
            } else if (fileNames[i].startsWith("perfdmf.cfg") && !fileNames[i].endsWith("~")) {
                String name = fileNames[i].substring(12);
                perfdmfConfigs.add(createDatabase(name, paraprofDirectory + "/" + fileNames[i]));
            }
        }
        return perfdmfConfigs;
    }

    public String toString() {
        String dbDisplayName;
        dbDisplayName = config.getConnectionString();
        if (dbDisplayName.compareTo("") == 0) {
            dbDisplayName = "none";
        }
        //return "DB - " + name + " (" + dbDisplayName + ")"; 
        return name + " (" + dbDisplayName + ")"; 
    }

}
