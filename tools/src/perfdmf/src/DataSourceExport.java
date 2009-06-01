package edu.uoregon.tau.perfdmf;

import java.io.*;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.zip.GZIPOutputStream;

public class DataSourceExport {

    private static int findGroupID(Group groups[], Group group) {
        for (int i = 0; i < groups.length; i++) {
            if (groups[i] == group) {
                return i;
            }
        }
        throw new RuntimeException("Couldn't find group: " + group.getName());
    }

    public static void writeDelimited(DataSource dataSource, File file) throws FileNotFoundException, IOException {
        FileOutputStream out = new FileOutputStream(file);
        writeDelimited(dataSource, out);
    }

    public static void writeDelimited(DataSource dataSource, OutputStream out) throws IOException {
        OutputStreamWriter outWriter = new OutputStreamWriter(out);
        BufferedWriter bw = new BufferedWriter(outWriter);

        int numMetrics = dataSource.getNumberOfMetrics();

        bw.write("Node\tContext\tThread\tFunction\tNumCalls\tNumSubr");

        for (int i = 0; i < numMetrics; i++) {
            String metricName = dataSource.getMetricName(i);
            bw.write("\tInclusive " + metricName);
            bw.write("\tExclusive " + metricName);
        }

        bw.write("\tGroup");
        bw.write("\n");

        for (Iterator it = dataSource.getAllThreads().iterator(); it.hasNext();) {
            Thread thread = (edu.uoregon.tau.perfdmf.Thread) it.next();

            for (Iterator it2 = thread.getFunctionProfileIterator(); it2.hasNext();) {
                FunctionProfile fp = (FunctionProfile) it2.next();
                if (fp != null) {
                    bw.write(thread.getNodeID() + "\t" + thread.getContextID() + "\t" + thread.getThreadID() + "\t");
                    bw.write(fp.getName() + "\t");
                    bw.write(fp.getNumCalls() + "\t");
                    bw.write(fp.getNumSubr() + "");
                    for (int i = 0; i < numMetrics; i++) {
                        bw.write("\t" + fp.getInclusive(i));
                        bw.write("\t" + fp.getExclusive(i));
                    }
                    bw.write("\t" + fp.getFunction().getGroupString());
                    bw.write("\n");
                }
            }
            bw.write("\n");
        }

        bw.write("Node\tContext\tThread\tUser Event\tNumSamples\tMin\tMax\tMean\tStdDev\n");
        for (Iterator it = dataSource.getAllThreads().iterator(); it.hasNext();) {
            Thread thread = (edu.uoregon.tau.perfdmf.Thread) it.next();

            for (Iterator it2 = thread.getUserEventProfiles(); it2.hasNext();) {
                UserEventProfile uep = (UserEventProfile) it2.next();
                if (uep != null) {

                    bw.write(thread.getNodeID() + "\t" + thread.getContextID() + "\t" + thread.getThreadID());
                    bw.write("\t" + uep.getUserEvent().getName());

                    bw.write("\t" + uep.getNumSamples());
                    bw.write("\t" + uep.getMinValue());
                    bw.write("\t" + uep.getMaxValue());
                    bw.write("\t" + uep.getMeanValue());
                    bw.write("\t" + uep.getStdDev());
                    bw.write("\n");

                }
            }
            bw.write("\n");
        }
        bw.close();
        outWriter.close();
        out.close();

    }

    public static void writePacked(DataSource dataSource, File file) throws FileNotFoundException, IOException {
        //File file = new File("/home/amorris/test.ppk");
        FileOutputStream ostream = new FileOutputStream(file);
        writePacked(dataSource, ostream);
    }

    public static void writePacked(DataSource dataSource, OutputStream ostream) throws IOException {
        GZIPOutputStream gzip = new GZIPOutputStream(ostream);
        BufferedOutputStream bw = new BufferedOutputStream(gzip);
        DataOutputStream p = new DataOutputStream(bw);

        Group derived = dataSource.getGroup("TAU_CALLPATH_DERIVED");
        int numFunctions = 0;

        for (Iterator it = dataSource.getFunctions(); it.hasNext();) {
            Function function = (Function) it.next();
            if (!function.isGroupMember(derived)) {
                numFunctions++;
            }
        }

        int numMetrics = dataSource.getNumberOfMetrics();
        int numUserEvents = dataSource.getNumUserEvents();
        int numGroups = dataSource.getNumGroups();

        // write out magic cookie
        p.writeChar('P'); // two bytes
        p.writeChar('P'); // two bytes
        p.writeChar('K'); // two bytes

        // write out version
        p.writeInt(2); // four bytes

        // write out lowest compatibility version
        p.writeInt(1); // four bytes

        // Write meta-data
        ByteArrayOutputStream headerStream = new ByteArrayOutputStream();
        DataOutputStream headerData = new DataOutputStream(headerStream);

        // future versions can put another header block here, we will skip this many bytes
        headerData.writeInt(0);

        if (dataSource.getMetaData() != null) {
            // write out the trial meta-data, this data is normalized across all threads (i.e. it applies to all threads)
            Map metaData = dataSource.getMetaData();
            headerData.writeInt(metaData.size());
            for (Iterator it2 = metaData.keySet().iterator(); it2.hasNext();) {
                String name = (String) it2.next();
                String value = (String) metaData.get(name);
                headerData.writeUTF(name);
                headerData.writeUTF(value);
            }
        } else {
            headerData.writeInt(0);
        }

        headerData.writeInt(dataSource.getAllThreads().size());
        for (Iterator it = dataSource.getAllThreads().iterator(); it.hasNext();) {
            Thread thread = (Thread) it.next();
            Map metaData = thread.getMetaData();
            headerData.writeInt(thread.getNodeID());
            headerData.writeInt(thread.getContextID());
            headerData.writeInt(thread.getThreadID());
            headerData.writeInt(metaData.size());
            for (Iterator it2 = metaData.keySet().iterator(); it2.hasNext();) {
                String name = (String) it2.next();
                String value = (String) metaData.get(name);
                headerData.writeUTF(name);
                headerData.writeUTF(value);
            }
        }
        headerData.close();

        p.writeInt(headerData.size());
        p.write(headerStream.toByteArray());

        // write out metric names
        p.writeInt(numMetrics);
        for (int i = 0; i < numMetrics; i++) {
            String metricName = dataSource.getMetricName(i);
            p.writeUTF(metricName);
        }

        // write out group names
        p.writeInt(numGroups);
        Group groups[] = new Group[numGroups];
        int idx = 0;
        for (Iterator it = dataSource.getGroups(); it.hasNext();) {
            Group group = (Group) it.next();
            String groupName = group.getName();
            p.writeUTF(groupName);
            groups[idx++] = group;
        }

        // write out function names
        Function functions[] = new Function[numFunctions];
        idx = 0;
        p.writeInt(numFunctions);
        for (Iterator it = dataSource.getFunctions(); it.hasNext();) {
            Function function = (Function) it.next();
            if (!function.isGroupMember(derived)) {

                functions[idx++] = function;
                p.writeUTF(function.getName());

                List thisGroups = function.getGroups();
                if (thisGroups == null) {
                    p.writeInt(0);
                } else {
                    p.writeInt(thisGroups.size());
                    for (int i = 0; i < thisGroups.size(); i++) {
                        Group group = (Group) thisGroups.get(i);
                        p.writeInt(findGroupID(groups, group));
                    }
                }
            }
        }

        // write out user event names
        UserEvent userEvents[] = new UserEvent[numUserEvents];
        idx = 0;
        p.writeInt(numUserEvents);
        for (Iterator it = dataSource.getUserEvents(); it.hasNext();) {
            UserEvent userEvent = (UserEvent) it.next();
            userEvents[idx++] = userEvent;
            p.writeUTF(userEvent.getName());
        }

        // write out the number of threads
        p.writeInt(dataSource.getAllThreads().size());

        // write out each thread's data
        for (Iterator it = dataSource.getAllThreads().iterator(); it.hasNext();) {
            Thread thread = (Thread) it.next();

            p.writeInt(thread.getNodeID());
            p.writeInt(thread.getContextID());
            p.writeInt(thread.getThreadID());

            // count (non-null) function profiles
            int count = 0;
            for (int i = 0; i < numFunctions; i++) {
                FunctionProfile fp = thread.getFunctionProfile(functions[i]);
                if (fp != null) {
                    count++;
                }
            }
            p.writeInt(count);

            // write out function profiles
            for (int i = 0; i < numFunctions; i++) {
                FunctionProfile fp = thread.getFunctionProfile(functions[i]);

                if (fp != null) {
                    p.writeInt(i); // which function (id)
                    p.writeDouble(fp.getNumCalls());
                    p.writeDouble(fp.getNumSubr());

                    for (int j = 0; j < numMetrics; j++) {
                        p.writeDouble(fp.getExclusive(j));
                        p.writeDouble(fp.getInclusive(j));
                    }
                }
            }

            // count (non-null) user event profiles
            count = 0;
            for (int i = 0; i < numUserEvents; i++) {
                UserEventProfile uep = thread.getUserEventProfile(userEvents[i]);
                if (uep != null) {
                    count++;
                }
            }

            p.writeInt(count); // number of user event profiles

            // write out user event profiles
            for (int i = 0; i < numUserEvents; i++) {
                UserEventProfile uep = thread.getUserEventProfile(userEvents[i]);

                if (uep != null) {
                    p.writeInt(i);
                    p.writeInt((int) uep.getNumSamples());
                    p.writeDouble(uep.getMinValue());
                    p.writeDouble(uep.getMaxValue());
                    p.writeDouble(uep.getMeanValue());
                    p.writeDouble(uep.getSumSquared());
                }
            }
        }

        p.close();
        gzip.close();
        ostream.close();

    }
    
    private static String xmlFixUp(String string) {
        string = string.replaceAll("&","&amp;");
        string = string.replaceAll(">","&gt;");
        string = string.replaceAll("<","&lt;");
        string = string.replaceAll("\n","&#xa;");
        return string;
    }

    private static void writeXMLSnippet(BufferedWriter bw, Map metaData) throws IOException {
        for (Iterator it2 = metaData.keySet().iterator(); it2.hasNext();) {
            String name = (String)it2.next();
            String value = (String)metaData.get(name);
            bw.write("<attribute><name>"+xmlFixUp(name)+"</name><value>"+xmlFixUp(value)+"</value></attribute>");
        }
    }
    private static void writeMetric(File root, DataSource dataSource, int metricID, Function[] functions, String[] groupStrings,
            UserEvent[] userEvents, List threads) throws IOException {

        int numMetrics = dataSource.getNumberOfMetrics();
        int numUserEvents = dataSource.getNumUserEvents();
        int numGroups = dataSource.getNumGroups();

        for (Iterator it = threads.iterator(); it.hasNext();) {
            Thread thread = (Thread) it.next();

            File file = new File(root + "/profile." + thread.getNodeID() + "." + thread.getContextID() + "."
                    + thread.getThreadID());

            FileOutputStream out = new FileOutputStream(file);
            OutputStreamWriter outWriter = new OutputStreamWriter(out);
            BufferedWriter bw = new BufferedWriter(outWriter);

            // count function profiles
            int count = 0;
            for (int i = 0; i < functions.length; i++) {
                FunctionProfile fp = thread.getFunctionProfile(functions[i]);
                if (fp != null) {
                    count++;
                }
            }

            if (dataSource.getNumberOfMetrics() == 1 && dataSource.getMetricName(metricID).equals("Time")) {
                bw.write(count + " templated_functions\n");
            } else {
                bw.write(count + " templated_functions_MULTI_" + dataSource.getMetricName(metricID) + "\n");
            }
            
            if (dataSource.getMetaData() != null) {
                bw.write("# Name Calls Subrs Excl Incl ProfileCalls<metadata>");
                writeXMLSnippet(bw, dataSource.getMetaData());
                writeXMLSnippet(bw, thread.getMetaData());
                bw.write("</metadata>\n");
                
            } else {
                bw.write("# Name Calls Subrs Excl Incl ProfileCalls\n");
            }
            
            // write out function profiles
            for (int i = 0; i < functions.length; i++) {
                FunctionProfile fp = thread.getFunctionProfile(functions[i]);

                if (fp != null) {
                    bw.write('"' + functions[i].getName() + "\" ");
                    bw.write((int) fp.getNumCalls() + " ");
                    bw.write((int) fp.getNumSubr() + " ");
                    bw.write(fp.getExclusive(metricID) + " ");
                    bw.write(fp.getInclusive(metricID) + " ");
                    bw.write("0 " + "GROUP=\"" + groupStrings[i] + "\"\n");
                }
            }

            bw.write("0 aggregates\n");

            // count user event profiles
            count = 0;
            for (int i = 0; i < numUserEvents; i++) {
                UserEventProfile uep = thread.getUserEventProfile(userEvents[i]);
                if (uep != null) {
                    count++;
                }
            }

            if (count > 0) {
                bw.write(count + " userevents\n");
                bw.write("# eventname numevents max min mean sumsqr\n");

                // write out user event profiles
                for (int i = 0; i < numUserEvents; i++) {
                    UserEventProfile uep = thread.getUserEventProfile(userEvents[i]);

                    if (uep != null) {
                        bw.write('"' + userEvents[i].getName() + "\" ");
                        bw.write(uep.getNumSamples() + " ");
                        bw.write(uep.getMaxValue() + " ");
                        bw.write(uep.getMinValue() + " ");
                        bw.write(uep.getMeanValue() + " ");
                        bw.write(uep.getSumSquared() + "\n");
                    }
                }
            }
            bw.close();
            outWriter.close();
            out.close();
        }

    }

    public static String createSafeMetricName(String name) {
        String ret = name.replace('/', '\\');
        return ret;
    }


    public static void writeProfiles(DataSource dataSource, File directory) throws IOException {
        writeProfiles(dataSource, directory, dataSource.getAllThreads());
    }

    public static void writeProfiles(DataSource dataSource, File directory, List threads) throws IOException {

        int numMetrics = dataSource.getNumberOfMetrics();
        int numUserEvents = dataSource.getNumUserEvents();
        int numGroups = dataSource.getNumGroups();

        int idx = 0;

        Group derived = dataSource.getGroup("TAU_CALLPATH_DERIVED");

        int numFunctions = 0;
        for (Iterator it = dataSource.getFunctions(); it.hasNext();) {
            Function function = (Function) it.next();
            if (function.isGroupMember(derived)) {
                continue;
            }
            numFunctions++;
        }

        // write out group names
        Group groups[] = new Group[numGroups];
        for (Iterator it = dataSource.getGroups(); it.hasNext();) {
            Group group = (Group) it.next();
            String groupName = group.getName();
            groups[idx++] = group;
        }

        Function functions[] = new Function[numFunctions];
        String groupStrings[] = new String[numFunctions];
        idx = 0;

        // write out function names
        for (Iterator it = dataSource.getFunctions(); it.hasNext();) {
            Function function = (Function) it.next();

            if (!function.isGroupMember(derived)) {
                functions[idx] = function;

                List thisGroups = function.getGroups();

                if (thisGroups == null) {
                    groupStrings[idx] = "";
                } else {
                    groupStrings[idx] = "";

                    for (int i = 0; i < thisGroups.size(); i++) {
                        Group group = (Group) thisGroups.get(i);
                        if (i == 0) {
                            groupStrings[idx] = group.getName();
                        } else {
                            groupStrings[idx] = groupStrings[idx] + " | " + group.getName();
                        }
                    }

                    groupStrings[idx] = groupStrings[idx].trim();
                }
                idx++;
            }
        }

        UserEvent userEvents[] = new UserEvent[numUserEvents];
        idx = 0;
        // collect user event names
        for (Iterator it = dataSource.getUserEvents(); it.hasNext();) {
            UserEvent userEvent = (UserEvent) it.next();
            userEvents[idx++] = userEvent;
        }

        if (numMetrics == 1) {
            writeMetric(directory, dataSource, 0, functions, groupStrings, userEvents, threads);
        } else {
            for (int i = 0; i < numMetrics; i++) {
                String name = "MULTI__" + createSafeMetricName(dataSource.getMetricName(i));
                boolean success = (new File(name).mkdir());
                if (!success) {
                    System.err.println("Failed to create directory: " + name);
                } else {
                    writeMetric(new File(directory + "/" + name), dataSource, i, functions, groupStrings, userEvents, threads);
                }
            }
        }
    }

}
