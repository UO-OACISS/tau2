package edu.uoregon.tau.perfdmf;

import java.io.*;
import java.util.*;

import edu.uoregon.tau.common.Utility;

public class EBSTraceReader {

    private static String reader_version = "0.2";

    private DataSource dataSource;

    // a map of TAU callpaths to sample callstacks
    private Map sampleMap = new HashMap();

    private int node = -1;
    private int tid = -1;

    private static boolean showCallSites = false;

    private Group callpathGroup;
    private Group sampleGroup;

    private Thread thread;

    public EBSTraceReader(DataSource dataSource) {
        this.dataSource = dataSource;
        callpathGroup = dataSource.getGroup("TAU_CALLPATH");
        sampleGroup = dataSource.addGroup("TAU_SAMPLE");
    }

    private void addSample(String callstack, String callpath) {
        Map map = (Map) sampleMap.get(callpath);
        if (map != null) {
            Integer count = (Integer) map.get(callstack);
            if (count != null) {
                map.put(callstack, new Integer(count.intValue() + 1));
            } else {
                map.put(callstack, new Integer(1));
            }
        } else {
            map = new HashMap();
            map.put(callstack, new Integer(1));
            sampleMap.put(callpath, map);
        }
    }

    private boolean stackMatch(String cpNode, String csNode) {
        //        System.out.println("Checking '" + cpNode + "' vs '" + csNode + "'");

        String fields[] = csNode.split(":");
        String csRoutine = fields[0];
        if (csRoutine.endsWith("()")) {
            csRoutine = csRoutine.substring(0, csRoutine.length() - 2);
        }

        //        System.out.println("csRoutine = '" + csRoutine + "'");
        if (cpNode.equals(csRoutine)) {
            //            System.out.println("TRUE");
            return true;
        }

        //        System.out.println("FALSE");
        return false;
    }

    private String resolveCallpath(String callpath, String callstack) {

        String cp[] = callpath.split("=>");
        for (int i = 0; i < cp.length; i++) {
            cp[i] = cp[i].trim();
        }

        String cs[] = callstack.split("=>");
        for (int i = 0; i < cs.length; i++) {
            cs[i] = cs[i].trim();
        }

        //int csIdx = cs.length-1;
        //int cpIdx = cp.length-1;;

        int matchCsIdx = 0, matchCpIdx = 0;

        done: for (int cpIdx = cp.length - 1; cpIdx >= 0; cpIdx--) {
            for (int csIdx = cs.length - 1; csIdx >= 0; csIdx--) {
                if (stackMatch(cp[cpIdx], cs[csIdx])) {
                    matchCsIdx = csIdx;
                    matchCpIdx = cpIdx;
                    break done;
                }
            }
        }

        //String result = callpath + " => " + callstack;

        String result = null;

        for (int i = 0; i < cp.length; i++) {
            if (result == null) {
                result = cp[i];
            } else {
                result = result + " => " + cp[i];
            }
        }

        //        for (int i = 0; i <= matchCpIdx; i++) {
        //            if (result == null) {
        //                result = cp[i];
        //            } else {
        //                result = result + " => " + cp[i];
        //            }
        //        }

        for (int i = matchCsIdx; i < cs.length; i++) {
            if (result == null) {
                result = cs[i];
            } else {
                result = result + " => " + cs[i];
            }
        }
        //
        //                System.out.println("callpath = " + callpath);
        //                System.out.println("callstack = " + callstack);
        //                System.out.println("result = " + result);
        //                System.out.println("--------------------");

        return result;
    }

    private void addIntermediateNodes(Thread thread, String callpath, int metric, double chunk, Group callpathGroup) {

        String cp[] = callpath.split("=>");
        for (int i = 0; i < cp.length; i++) {
            cp[i] = cp[i].trim();
        }

        String path = cp[0];
        for (int i = 1; i < cp.length; i++) {
            path = path + " => " + cp[i];

            Function function = dataSource.addFunction(path);
            function.addGroup(callpathGroup);

            FunctionProfile fp = thread.getOrCreateFunctionProfile(function, dataSource.getNumberOfMetrics());

            if (fp.getNumCalls() == 0) {
                fp.setInclusive(metric, fp.getInclusive(metric) + chunk);
            }
        }

    }

    // Process the map we've generated
    private void processMap() {

        Thread thread = dataSource.getThread(node, 0, tid);

        // we'll need to add these to the callpath group
        Group callpathGroup = dataSource.addGroup("TAU_CALLPATH");

        Group sampleGroup = dataSource.addGroup("TAU_SAMPLE");

        // for each TAU callpath
        for (Iterator it = sampleMap.keySet().iterator(); it.hasNext();) {
            String callpath = (String) it.next();

            // get the set of callstacks for this callpath
            Map callstacks = (Map) sampleMap.get(callpath);
            int numSamples = 0;
            for (Iterator it2 = callstacks.values().iterator(); it2.hasNext();) {
                Integer count = (Integer) it2.next();
                numSamples += count.intValue();
            }
            
            callpath = Utility.removeRuns(callpath);

            Function function = dataSource.getFunction(callpath);

            if (function == null) {
                System.err.println("Error: callpath not found in profile: \"" + callpath+"\"");
                continue;
            }

            FunctionProfile fp = thread.getFunctionProfile(function);

            FunctionProfile flatFP = null;
            if (callpath.lastIndexOf("=>") != -1) {
                String flatName = callpath.substring(callpath.lastIndexOf("=>") + 2).trim();
                Function flatFunction = dataSource.getFunction(flatName);
                if (flatFunction == null) {
                    System.err.println("Error: function not found in profile: " + flatName);
                    continue;
                }
                flatFP = thread.getFunctionProfile(flatFunction);
            }

            // not the best way to handle multiple metrics
            for (int m = 0; m < dataSource.getNumberOfMetrics(); m++) {
                double exclusive = fp.getExclusive(m);
                double chunk = exclusive / numSamples;
                fp.setExclusive(m, 0);
                if (flatFP != null) {
                    flatFP.setExclusive(m, 0);
                }

                for (Iterator it2 = callstacks.keySet().iterator(); it2.hasNext();) {
                    String callstack = (String) it2.next();
                    int count = ((Integer) callstacks.get(callstack)).intValue();
                    double value = chunk * count;

                    String resolvedCallpath = callpath + " => " + callstack;

                    Function newCallpathFunc = dataSource.addFunction(resolvedCallpath);
                    newCallpathFunc.addGroup(callpathGroup);
                    newCallpathFunc.addGroup(sampleGroup);

                    FunctionProfile callpathProfile = thread.getOrCreateFunctionProfile(newCallpathFunc,
                            dataSource.getNumberOfMetrics());

                    callpathProfile.setInclusive(m, callpathProfile.getInclusive(m) + value);
                    callpathProfile.setExclusive(m, callpathProfile.getExclusive(m) + value);
                    callpathProfile.setNumCalls(callpathProfile.getNumCalls() + count);

                    addIntermediateNodes(thread, resolvedCallpath, m, value, callpathGroup);

                    if (resolvedCallpath.lastIndexOf("=>") != -1) {
                        String flatName = UtilFncs.getRightMost(resolvedCallpath);

                        Function flatFunction = dataSource.addFunction(flatName);
                        newCallpathFunc.addGroup(sampleGroup);

                        FunctionProfile flatProfile = thread.getOrCreateFunctionProfile(flatFunction,
                                dataSource.getNumberOfMetrics());

                        flatProfile.setInclusive(m, flatProfile.getInclusive(m) + value);
                        flatProfile.setExclusive(m, flatProfile.getExclusive(m) + value);
                        flatProfile.setNumCalls(flatProfile.getNumCalls() + count);
                    }
                }
            }
        }
    }

    private static String stripFileLine(String location) {
        try {
            int lastColon = location.lastIndexOf(':');
            int secondToLastColon = location.lastIndexOf(':', lastColon - 1);
            return "[INTERMEDIATE] " + location.substring(0, secondToLastColon);

        } catch (Exception e) {
            System.out.println("Location = " + location);
            e.printStackTrace();
            return location;
        }
    }

    private static String stripPath(String location) {
        try {
            int lastColon = location.lastIndexOf(':');
            int secondToLastColon = location.lastIndexOf(':', lastColon - 1);
            String routine = location.substring(0, secondToLastColon);
            String path = location.substring(secondToLastColon + 1, lastColon);
            String lineno = location.substring(lastColon + 1);

            String filename = path.substring(path.lastIndexOf("/") + 1);

            return routine + ":" + filename + ":" + lineno;
        } catch (Exception e) {
            System.out.println("Location = " + location);
            e.printStackTrace();
            return location;
        }
    }

    static public void main(String[] args) {

        System.out.println(stripFileLine("foo:bar.cc:2060"));

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
                    if (inputString.startsWith("# Format version:")) {
                        String version_text = inputString.substring("# Format version:".length() + 1).trim();
                        if (!version_text.equals(reader_version)) {
                            String error = "Wrong EBS version, we are looking for " + reader_version + ", but found "
                                    + version_text;
                            System.err.println(error);
                            throw new RuntimeException(error);
                        }
                    }
                    if (inputString.startsWith("# node:")) {
                        String node_text = inputString.substring(8);
                        node = Integer.parseInt(node_text);
                        if (tid != -1) {
                            thread = dataSource.getThread(node, 0, tid);
                        }
                    }
                    if (inputString.startsWith("# thread:")) {
                        String tid_text = inputString.substring(10);
                        tid = Integer.parseInt(tid_text);
                        if (node != -1) {
                            thread = dataSource.getThread(node, 0, tid);
                        }
                    }
                } else {

                    String fields[] = inputString.split("\\|");
                    long timestamp = Long.parseLong(fields[0].trim());
                    long deltaBegin = Long.parseLong(fields[1].trim());
                    long deltaEnd = Long.parseLong(fields[2].trim());
                    String metrics = fields[3].trim();
                    String callpath = fields[4].trim();
                    String callstack = fields[5].trim();

                    if (!callstack.startsWith("??:")) {
                        try {

                            List csList = new ArrayList();

                            String callStackEntries[] = callstack.split("@");

                            // We hide the topmost node if there is more than one, otherwise
                            // there will be a duplicate intermediate node
                            int limit = callStackEntries.length;
                            if (limit > 1) {
                                limit--;
                            }

                            for (int i = 0; i < limit; i++) {
                                if (showCallSites) {
                                    csList.add(callStackEntries[i]);
                                } else {
                                    if (i == 0) {
                                        csList.add(stripPath(callStackEntries[i]));
                                    } else {
                                        csList.add(stripFileLine(callStackEntries[i]));
                                    }
                                }
                            }

                            Collections.reverse(csList);

                            String location = null;
                            for (Iterator it3 = csList.iterator(); it3.hasNext();) {
                                if (location == null) {
                                    location = (String) it3.next();
                                } else {
                                    location = location + " => " + it3.next();
                                }
                            }

                            addSample(location, callpath);
                        } catch (Exception e) {
                            e.printStackTrace();
                            System.out.println(inputString);
                        }
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
        long time = System.currentTimeMillis();
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

        EBSTraceReader ebsTraceReader = new EBSTraceReader(dataSource);
        for (int i = 0; i < files.length; i++) {
            ebsTraceReader.processEBSTrace(dataSource, files[i]);
        }

        time = (System.currentTimeMillis()) - time;
        //System.out.println("Time to process (in milliseconds): " + time);
    }

}
