/*
 * Name: PSRunDataSource.java Author: Kevin Huck Description: Parse an psrun XML
 * data file.
 */

/*
 * To do:
 */

package edu.uoregon.tau.dms.dss;

import java.io.*;
import java.util.*;
import org.xml.sax.*;
import org.xml.sax.helpers.*;

class NoOpEntityResolver implements EntityResolver {
    public InputSource resolveEntity(String publicId, String systemId) {
        return new InputSource(new StringReader(""));
    }
}

public class PSRunDataSource extends DataSource {

    public PSRunDataSource(Object initializeObject) {
        super();
        this.setMetrics(new Vector());
        this.initializeObject = initializeObject;
    }

    private Object initializeObject;

	public void cancelLoad() {
		return;
	}
	public int getProgress() {
		return 0;
	}

    public void load() {
        boolean firstFile = true;
        try {
            v = (Vector) initializeObject;
            // create our XML parser
            XMLReader xmlreader = XMLReaderFactory.createXMLReader("org.apache.xerces.parsers.SAXParser");

            DefaultHandler handler = new PSRunLoadHandler();
            xmlreader.setContentHandler(handler);
            xmlreader.setErrorHandler(handler);

            xmlreader.setEntityResolver(new NoOpEntityResolver());
            for (Enumeration e = v.elements(); e.hasMoreElements();) {
                files = (File[]) e.nextElement();
                for (int i = 0; i < files.length; i++) {
                    System.out.println("Processing data file, please wait ......");
                    long time = System.currentTimeMillis();

                    StringTokenizer st = new StringTokenizer(files[i].getName(), ".");

                    if (st.countTokens() == 3) {
                        // increment the node counter - there's a file for each
                        // node
                        nodeID++;
                    } else {
                        String prefix = st.nextToken();

                        String tid = st.nextToken();
                        threadID = Integer.parseInt(tid);

                        String nid = st.nextToken();
                        Integer tmpID = (Integer) nodeHash.get(nid);
                        if (tmpID == null) {
                            nodeID = nodeHash.size();
                            nodeHash.put(nid, new Integer(nodeID));
                        } else {
                            nodeID = tmpID.intValue();
                        }
                    }

                    // parse the next file
                    xmlreader.parse(new InputSource(new FileInputStream(files[i])));

                    // initialize the thread/node
                    initializeThread();

                    // get the data, and put it in the mapping
                    PSRunLoadHandler tmpHandler = (PSRunLoadHandler) handler;
                    Hashtable metricHash = tmpHandler.getMetricHash();
                    for (Enumeration keys = metricHash.keys(); keys.hasMoreElements();) {
                        String key = (String) keys.nextElement();
                        String value = (String) metricHash.get(key);
                        processHardwareCounter(key, value);
                    }

                    // generate summary statistics
                    //Remove after testing is complete.
                    //this.setMeanDataAllMetrics(0);

                    if (UtilFncs.debug) {
                        System.out.println("The total number of threads is: "
                                + this.getNCT().getTotalNumberOfThreads());
                        System.out.println("The number of mappings is: "
                                + this.getTrialData().getNumFunctions());
                        System.out.println("The number of user events is: "
                                + this.getTrialData().getNumUserEvents());
                    }

                    time = (System.currentTimeMillis()) - time;
                    System.out.println("Done processing data file!");
                    System.out.println("Time to process file (in milliseconds): " + time);
                }
            }
            //Generate derived data.
            this.generateDerivedData();

        } catch (Exception e) {
            e.printStackTrace();
            UtilFncs.systemError(e, null, "SSD01");
        }
    }

    //####################################
    //Private Section.
    //####################################

    private void initializeThread() {
        // create the mapping, if necessary
        function = this.getTrialData().addFunction("Entire application", 1);

        // make sure we start at zero for all counters
        nodeID = (nodeID == -1) ? 0 : nodeID;
        contextID = (contextID == -1) ? 0 : contextID;
        threadID = (threadID == -1) ? 0 : threadID;

        //Get the node,context,thread.
        node = this.getNCT().getNode(nodeID);
        if (node == null)
            node = this.getNCT().addNode(nodeID);
        context = node.getContext(contextID);
        if (context == null)
            context = node.addContext(contextID);
        thread = context.getThread(threadID);
        if (thread == null) {
            thread = context.addThread(threadID);
            thread.setDebug(this.debug());
            thread.initializeFunctionList(this.getTrialData().getNumFunctions());
            thread.initializeUsereventList(this.getTrialData().getNumUserEvents());
        }

        functionProfile = thread.getFunctionProfile(function);
        if (functionProfile == null) {
            functionProfile = new FunctionProfile(function);
            thread.addFunctionProfile(functionProfile, function.getID());
        }
    }

    private void processHardwareCounter(String key, String value) {
        thread.incrementStorage();
        function.incrementStorage();
        functionProfile.incrementStorage();
        try {
            double eventValue = 0;
            eventValue = Double.parseDouble(value);

            metric = this.getNumberOfMetrics();
            //Set the metric name.
            Metric newMetric = this.addMetric(key);
            if (metric < this.getNumberOfMetrics()) {
                //Need to call increaseVectorStorage() on all objects that
                // require it.
                this.getTrialData().increaseVectorStorage();
            }
            metric = newMetric.getID();

            functionProfile.setExclusive(metric, eventValue);
            functionProfile.setInclusive(metric, eventValue);
            functionProfile.setInclusivePerCall(metric, eventValue);
            functionProfile.setNumCalls(1);
            functionProfile.setNumSubr(0);

            if ((function.getMaxExclusive(metric)) < eventValue) {
                function.setMaxExclusive(metric, eventValue);
                function.setMaxInclusive(metric, eventValue);
            }
            if (function.getMaxInclusivePerCall(metric) < eventValue)
                function.setMaxInclusivePerCall(metric, eventValue);
            function.setMaxNumCalls(1);
            function.setMaxNumSubr(0);

        } catch (Exception e) {
            System.out.println("An error occurred while parsing the XML data!");
            e.printStackTrace();
        }
    }

    //####################################
    //End - Private Section.
    //####################################

    //######
    //Frequently used items.
    //######
    private int metric = 0;
    private Function function = null;
    private FunctionProfile functionProfile = null;
    private Node node = null;
    private Context context = null;
    private edu.uoregon.tau.dms.dss.Thread thread = null;
    private int nodeID = -1;
    private int contextID = -1;
    private int threadID = -1;
    private String inputString = null;
    private String s1 = null;
    private String s2 = null;
    private String tokenString;
    private String groupNamesString = null;
    private StringTokenizer genericTokenizer;
    private Vector v = null;
    private File[] files = null;
    private BufferedReader br = null;
    boolean initialized = false;
    private Hashtable nodeHash = new Hashtable();
    //######
    //End - Frequently used items.
    //######
}