package edu.uoregon.tau.paraprof;

import java.io.File;
import java.io.FilenameFilter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

import edu.uoregon.tau.common.AlphanumComparator;
import edu.uoregon.tau.perfdmf.Application;
import edu.uoregon.tau.perfdmf.DBDataSource;
import edu.uoregon.tau.perfdmf.DataSource;
import edu.uoregon.tau.perfdmf.DatabaseAPI;
import edu.uoregon.tau.perfdmf.Experiment;
import edu.uoregon.tau.perfdmf.PackedProfileDataSource;
import edu.uoregon.tau.perfdmf.Trial;

class PPKFileFilter implements FilenameFilter {
    public PPKFileFilter() {}

    public boolean accept(File okplace, String name) {
        if (name.endsWith(".ppk")) {
            return true;
        }
        return false;
    }
}

public class TrialGrabber {

    public static void getExperiments(String path, List<List<ParaProfTrial>> exps, List<String> expnames) {
        exps.clear();
        expnames.clear();

        File expdirs[] = (new File(path)).listFiles();

        List sorted = new ArrayList<File>();

        for (int i = 0; i < expdirs.length; i++) {
            if (expdirs[i].isDirectory()) {
                sorted.add(expdirs[i]);
            }
        }
        Collections.sort(sorted, new AlphanumComparator());

        for (int i = 0; i < sorted.size(); i++) {
            File exp = (File)sorted.get(i);
            String name = exp.toString();
            name = name.substring(name.lastIndexOf('/') + 1);
            expnames.add(name);
            File ppkfiles[] = exp.listFiles();

            List<ParaProfTrial> trials = new ArrayList<ParaProfTrial>();
            for (int j = 0; j < ppkfiles.length; j++) {
                trials.add(getTrial(ppkfiles[j]));
            }
            exps.add(trials);
        }
    }

    public static ParaProfTrial getTrial(String file) {
        return getTrial(new File(file));
    }

    public static ParaProfTrial getTrial(File file) {
        DataSource dataSource = new PackedProfileDataSource(file);
        try {
            dataSource.load();
        } catch (Exception e) {
            e.printStackTrace();
        }
        Trial trial = new Trial();
        trial.setDataSource(dataSource);
        String name = file.toString();
        name = name.substring(name.lastIndexOf('/') + 1);
        name = name.substring(0, name.length() - 4);
        trial.setName(name);
        ParaProfTrial ppTrial = new ParaProfTrial(trial);
        return ppTrial;
    }

    public static List<ParaProfTrial> getTrials(String path) {
        List<ParaProfTrial> trials = new ArrayList<ParaProfTrial>();
        File directory = new File(path);

        File files[] = directory.listFiles(new PPKFileFilter());
        List sorted = new ArrayList<File>();
        if (files == null) {
            System.out.println("Could not find any .ppk files in directory '"+directory+"'\n");
            return null;
        }
        for (int i = 0; i < files.length; i++) {
            sorted.add(files[i]);
        }
        Collections.sort(sorted, new AlphanumComparator());

        for (int i = 0; i < sorted.size(); i++) {
            trials.add(getTrial((String) sorted.get(i)));
        }

        return trials;
    }

    public static List<ParaProfTrial> getTrialsFromDatabase(String config, String appname, String expname) {
        List<ParaProfTrial> trials = new ArrayList<ParaProfTrial>();

        DatabaseAPI dbApi = new DatabaseAPI();
        try {
            dbApi.initialize(config, false);
        } catch (Exception e) {
            e.printStackTrace();
        }
        Application app = dbApi.getApplication(appname, false);
        if (app == null) {
            System.err.println("App '" + appname + "' not found");
            System.exit(-1);
        }

        Experiment exp = dbApi.getExperiment(appname, expname, false);
        if (exp == null) {
            System.err.println("Exp '" + expname + "' not found");
            System.exit(-1);
        }

        dbApi.setApplication(app);
        dbApi.setExperiment(exp);
        List<Trial> list = dbApi.getTrialList(true);
        for (Iterator<Trial> it = list.iterator(); it.hasNext();) {
            Trial trial = it.next();
            dbApi.setTrial(trial.getID(),true);//TODO: Do these really need xml metadata?
            DataSource dataSource = new DBDataSource(dbApi);
            try {
                dataSource.load();
            } catch (Exception e) {
                e.printStackTrace();
            }
            String name = trial.getName();
            trial.setDataSource(dataSource);
            name = name.substring(name.lastIndexOf('/') + 1);
            if (name.endsWith(".ppk")) {
                name = name.substring(0, name.length() - 4);
            }
            trial.setName(name);
            ParaProfTrial ppTrial = new ParaProfTrial(trial);
            trials.add(ppTrial);
        }

        return trials;
    }

}
