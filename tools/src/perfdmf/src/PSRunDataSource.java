/*
 * Name: PSRunDataSource.java 
 * Author: Kevin Huck 
 * Description: Parse an psrun XML
 * data file.
 */

/*
 * To do:
 */

package edu.uoregon.tau.perfdmf;

import java.io.File;
import java.io.FileInputStream;
import java.io.StringReader;
import java.util.*;

import org.xml.sax.EntityResolver;
import org.xml.sax.InputSource;
import org.xml.sax.XMLReader;
import org.xml.sax.helpers.DefaultHandler;
import org.xml.sax.helpers.XMLReaderFactory;

class NoOpEntityResolver implements EntityResolver {
    public InputSource resolveEntity(String publicId, String systemId) {
        return new InputSource(new StringReader(""));
    }
}

public class PSRunDataSource extends DataSource {
    private int metric = 0;
    private Function function = null;
    private FunctionProfile functionProfile = null;
    private Node node = null;
    private Context context = null;
    private Thread thread = null;
    private int nodeID = -1;
    private int contextID = -1;
    private int threadID = -1;
    private List v = null;
    boolean initialized = false;
    private Hashtable nodeHash = new Hashtable();
    private int threadCounter = 0;

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

    public void load() throws DataSourceException {
        try {
            boolean firstFile = true;
            v = (List) initializeObject;
            // create our XML parser
            XMLReader xmlreader = XMLReaderFactory.createXMLReader("org.apache.xerces.parsers.SAXParser");

            DefaultHandler handler = new PSRunLoadHandler(this);
            xmlreader.setContentHandler(handler);
            xmlreader.setErrorHandler(handler);

            xmlreader.setEntityResolver(new NoOpEntityResolver());
            for (Iterator e = v.iterator(); e.hasNext();) {
                File files[] = (File[]) e.next();
                for (int i = 0; i < files.length; i++) {
                    long time = System.currentTimeMillis();

                    StringTokenizer st = new StringTokenizer(files[i].getName(), ".");

                    if (st.countTokens() == 3) {
                        // increment the node counter - there's a file for each node
                        nodeID++;
                    } else {
                        String prefix = st.nextToken();

                        String tid = st.nextToken();
                        try {
                            threadID = Integer.parseInt(tid);
                            String nid = st.nextToken();
                            Integer tmpID = (Integer) nodeHash.get(nid);
                            if (tmpID == null) {
                                nodeID = nodeHash.size();
                                nodeHash.put(nid, new Integer(nodeID));
                            } else {
                                nodeID = tmpID.intValue();
                            }
                        } catch (NumberFormatException nfe) {
                            threadID = threadCounter++;
                        }
                    }

                    // initialize the thread/node
                    initializeThread();

                    // parse the next file
                    xmlreader.parse(new InputSource(new FileInputStream(files[i])));

                    PSRunLoadHandler tmpHandler = (PSRunLoadHandler) handler;

                    if (!tmpHandler.getIsProfile()) {
                        Hashtable metricHash = tmpHandler.getMetricHash();
                        for (Enumeration keys = metricHash.keys(); keys.hasMoreElements();) {
                            String key = (String) keys.nextElement();
                            String value = (String) metricHash.get(key);
                            processHardwareCounter(key, value);
                        }
                    }

                    time = (System.currentTimeMillis()) - time;
                    //System.out.println("Time to process file (in milliseconds): " + time);
                }
            }
            this.generateDerivedData();
        } catch (Exception e) {
            throw new DataSourceException(e);
        }
    }

    private void initializeThread() {
        function = this.addFunction("Entire application", 1);

        // make sure we start at zero for all counters
        nodeID = (nodeID == -1) ? 0 : nodeID;
        contextID = (contextID == -1) ? 0 : contextID;
        threadID = (threadID == -1) ? 0 : threadID;

        //Get the node,context,thread.
        node = this.getNode(nodeID);
        if (node == null)
            node = this.addNode(nodeID);
        context = node.getContext(contextID);
        if (context == null)
            context = node.addContext(contextID);
        thread = context.getThread(threadID);
        if (thread == null) {
            thread = context.addThread(threadID);
        }

        functionProfile = thread.getFunctionProfile(function);
        if (functionProfile == null) {
            functionProfile = new FunctionProfile(function);
            thread.addFunctionProfile(functionProfile);
        }
    }

    private void processHardwareCounter(String key, String value) {
        thread.addMetric();
        double eventValue = 0;
        eventValue = Double.parseDouble(value);

        metric = this.getNumberOfMetrics();
        //Set the metric name.
        Metric newMetric = this.addMetric(key);
        metric = newMetric.getID();

        functionProfile.setExclusive(metric, eventValue);
        functionProfile.setInclusive(metric, eventValue);
        //functionProfile.setInclusivePerCall(metric, eventValue);
        functionProfile.setNumCalls(1);
        functionProfile.setNumSubr(0);
    }

    public Thread getThread() {
        return thread;
    }

}