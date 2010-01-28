package edu.uoregon.tau.perfdmf;

import java.io.*;
import java.util.*;

public class EBSTraceReader {

    private DataSource dataSource;

    // a map of TAU callpaths to sample callstacks
    private Map sampleMap = new HashMap();

    private int node = -1;
    private int tid = -1;

    private static boolean showCallSites = false;

    public EBSTraceReader(DataSource dataSource) {
        this.dataSource = dataSource;
    }

    private void addSample(List callstack, String callpath) {
        Object obj = sampleMap.get(callpath);
        if (obj != null) {
            List list = (List) obj;
            list.add(callstack);
        } else {
            List list = new ArrayList();
            list.add(callstack);
            sampleMap.put(callpath, list);
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
        //        System.out.println("callpath = " + callpath);
        //        System.out.println("callstack = " + callstack);
        //        System.out.println("result = " + result);
        //        System.out.println("--------------------");

        return result;
    }

    private void addIntermediateNodes(Thread thread, String callpath) {

        String cp[] = callpath.split("=>");
        for (int i = 0; i < cp.length; i++) {
            cp[i] = cp[i].trim();
        }

        String path = cp[0];
        for (int i = 1; i < cp.length; i++) {
            path = path + " => " + cp[i];

            Function function = dataSource.addFunction(path);

            FunctionProfile fp = thread.getFunctionProfile(function);
            if (fp == null) {
                fp = new FunctionProfile(function, dataSource.getNumberOfMetrics());
                thread.addFunctionProfile(fp);
            }
        }

    }

    // Process the map we've generated
    private void processMap() {

        Thread thread = dataSource.getThread(node, 0, tid);

        // we'll need to add these to the callpath group
        Group callpathGroup = dataSource.getGroup("TAU_CALLPATH");

        // for each TAU callpath
        for (Iterator it = sampleMap.keySet().iterator(); it.hasNext();) {
            String callpath = (String) it.next();

            // get the set of callstacks for this callpath
            List callstacks = (List) sampleMap.get(callpath);
            int numSamples = callstacks.size();

            Function function = dataSource.getFunction(callpath);

            if (function == null) {
                System.err.println("Error: callpath not found in profile: " + callpath);
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

                for (Iterator it2 = callstacks.iterator(); it2.hasNext();) {
                    List callstack = (List) it2.next();
                    //Collections.reverse(callstack);
                    String location = null;
                    for (Iterator it3 = callstack.iterator(); it3.hasNext();) {
                        if (location == null) {
                            location = (String) it3.next();
                        } else {
                            location = location + " => " + it3.next();
                        }
                    }
                    location = location.trim();

                    //                    System.out.println("need to resolve:");
                    //                    System.out.println("callpath: " + callpath);
                    //                    System.out.println("location: " + location);
                    
                    String resolvedCallpath = resolveCallpath(callpath, location);
                    System.out.println("resolvedCallpath = " + resolvedCallpath);
                    Function newCallpathFunc = dataSource.addFunction(resolvedCallpath);
                    newCallpathFunc.addGroup(callpathGroup);

                    addIntermediateNodes(thread, resolvedCallpath);

                    FunctionProfile callpathProfile = thread.getFunctionProfile(newCallpathFunc);
                    if (callpathProfile == null) {
                        callpathProfile = new FunctionProfile(newCallpathFunc, dataSource.getNumberOfMetrics());
                        thread.addFunctionProfile(callpathProfile);
                    }

                    callpathProfile.setInclusive(m, callpathProfile.getInclusive(m) + chunk);
                    callpathProfile.setExclusive(m, callpathProfile.getExclusive(m) + chunk);
                    callpathProfile.setNumCalls(callpathProfile.getNumCalls() + 1);

                    //                    Function newFlatFunc = dataSource.addFunction(location);
                    //                    FunctionProfile flatProfile = thread.getFunctionProfile(newFlatFunc);
                    //                    if (flatProfile == null) {
                    //                        flatProfile = new FunctionProfile(newFlatFunc, dataSource.getNumberOfMetrics());
                    //                        thread.addFunctionProfile(flatProfile);
                    //                    }

                    //                  
                    //                    flatProfile.setInclusive(m, flatProfile.getInclusive(m) + chunk);
                    //                    flatProfile.setExclusive(m, flatProfile.getExclusive(m) + chunk);
                    //                    flatProfile.setNumCalls(flatProfile.getNumCalls() + 1);
                }
            }
        }
    }

    private String stripFileLine(String location) {
        return location.substring(0, location.indexOf(':'));
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
                    if (inputString.startsWith("# thread:")) {
                        String tid_text = inputString.substring(10);
                        tid = Integer.parseInt(tid_text);
                    }
                } else {

                    String fields[] = inputString.split("\\|");
                    String location = fields[3].trim();
                    if (!location.startsWith("??:")) {
                        try {
                            long timestamp = Long.parseLong(fields[0].trim());
                            long deltaBegin = Long.parseLong(fields[1].trim());
                            long deltaEnd = Long.parseLong(fields[2].trim());
                            String metrics = fields[4].trim();
                            String callpath = fields[5].trim();
                            String callstack = fields[6].trim();

                            String callStackEntries[] = callstack.split(" ");
                            List csList = new ArrayList();
                            csList.add(location);
                            for (int i = 0; i < callStackEntries.length; i++) {
                                if (showCallSites) {
                                    csList.add(callStackEntries[i]);
                                } else {
                                    csList.add(stripFileLine(callStackEntries[i]));
                                }
                            }
                            Collections.reverse(csList);
                            addSample(csList, callpath);
                        } catch (Exception e) {
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
    }

}
