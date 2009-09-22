package edu.uoregon.tau.perfdmf;

import java.io.*;
import java.util.*;

public class EBSTraceReader {

    Map sampleMap = new HashMap();
    int node = -1;

    private void addLocation(String location, String callpath) {
        Object obj = sampleMap.get(callpath);
        if (obj != null) {
            List list = (List) obj;
            list.add(location);
        } else {
            List list = new ArrayList();
            list.add(location);
            sampleMap.put(callpath, list);
        }
    }

    private void processMap() {
        // Process the map we've generated
    }
    
    private void processEBSTrace(DataSource dataSource, File file) {
        try {
            // reset data
            sampleMap.clear();
            node = -1;

            FileInputStream fis = new FileInputStream(file);
            InputStreamReader inReader = new InputStreamReader(fis);
            BufferedReader br = new BufferedReader(inReader);

            String inputString = br.readLine();
            while (inputString != null) {

                if (inputString.startsWith("#")) {
                    if (inputString.startsWith("# node:")) {
                        String node_text = inputString.substring(8);
                        node = Integer.parseInt(node_text);
                    }
                } else {

                    String fields[] = inputString.split("\\|");
                    String location = fields[3].trim();
                    if (!location.startsWith("??:0")) {
                        long timestamp = Long.parseLong(fields[0].trim());
                        long deltaBegin = Long.parseLong(fields[1].trim());
                        long deltaEnd = Long.parseLong(fields[2].trim());
                        String metrics = fields[4].trim();
                        String callpath = fields[5].trim();

                        addLocation(location, callpath);
                    }
                }

                inputString = br.readLine();
            }
            
            processMap();

        } catch (Exception ex) {
            ex.printStackTrace();
            if (!(ex instanceof IOException || ex instanceof FileNotFoundException)) {
                throw new RuntimeException(ex == null ? null : ex.toString());
            }
        }
    }

    public static void processEBSTraces(DataSource dataSource, File path) {

        if (path.isDirectory() == false) {
            return;
        }

        FilenameFilter filter = new FilenameFilter() {
            public boolean accept(File dir, String name) {
                if (name.startsWith("ebstrace.processed.")) {
                    return true;
                }
                return false;
            }
        };

        File files[] = path.listFiles(filter);

        EBSTraceReader ebsTraceReader = new EBSTraceReader();
        for (int i = 0; i < files.length; i++) {
            ebsTraceReader.processEBSTrace(dataSource, files[i]);
        }
    }

}
