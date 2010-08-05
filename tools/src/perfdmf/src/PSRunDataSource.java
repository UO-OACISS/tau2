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

import java.io.*;
import java.net.URL;
import java.util.*;

import org.xml.sax.EntityResolver;
import org.xml.sax.InputSource;
import org.xml.sax.XMLReader;
import org.xml.sax.helpers.XMLReaderFactory;

class NoOpEntityResolver implements EntityResolver {
    public InputSource resolveEntity(String publicId, String systemId) {
        return new InputSource(new StringReader(""));
    }
}

public class PSRunDataSource extends DataSource {
    private Thread thread = null;
    private int nodeID = -1;
    private int contextID = -1;
    private int threadID = -1;
    private int threadCounter = 0;
    PSRunLoadHandler handler;
    private Hashtable<String, Integer> nodeHash = new Hashtable<String, Integer>();

    public PSRunDataSource(Object initializeObject) {
        super();
        this.setMetrics(new Vector<Metric>());
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
        //Debug.dataSource = this;

        try {
            List v = (List) initializeObject;
            XMLReader xmlreader = XMLReaderFactory.createXMLReader("org.apache.xerces.parsers.SAXParser");

            handler = new PSRunLoadHandler(this);
            xmlreader.setContentHandler(handler);
            xmlreader.setErrorHandler(handler);

            xmlreader.setEntityResolver(new NoOpEntityResolver());
            for (Iterator e = v.iterator(); e.hasNext();) {

                File files[] = (File[]) e.next();
                for (int i = 0; i < files.length; i++) {
                    long time = System.currentTimeMillis();

                    StringTokenizer st = new StringTokenizer(files[i].getName(), ".");

                    if (st.countTokens() == 2) {
                    	// this is a multireport.  
                        nodeID++;
                        threadID = threadCounter++;
                    } else if (st.countTokens() == 3) {
                        // increment the node counter - there's a file for each node
                        nodeID++;
                    } else {
                        st.nextToken(); // prefix

                        String tid = st.nextToken();
                        try {
                            threadID = Integer.parseInt(tid);
                            String nid = st.nextToken();
                            Hashtable<String, Integer> nodeHash = new Hashtable<String, Integer>();
                            Integer tmpID = nodeHash.get(nid);
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

                    // reset the thread
                    thread = null;
                    //Debug.foo("IT-BEFORE");

                    File file = files[i];
                    InputStream istream;
                    if (file.toString().toLowerCase().startsWith("http:")) {
                        // When it gets converted from a String to a File http:// turns into http:/
                        URL url = new URL("http://" + file.toString().substring(6).replace('\\', '/'));
                        istream = url.openStream();
                    }  else {
                        istream = new FileInputStream(file);
                    }
                    
                    
                    // parse the next file
                    xmlreader.parse(new InputSource(istream));
                    //Debug.foo("IT-AFTER");

                    if (!handler.getIsProfile()) {
                        Hashtable metricHash = handler.getMetricHash();
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
            processSnapshots();
            this.generateDerivedData();
            this.aggregateMetaData();
            this.buildXMLMetaData();
            this.setGroupNamesPresent(true);
        } catch (Exception e) {
            if (e instanceof DataSourceException) {
                throw (DataSourceException) e;
            } else {
                throw new DataSourceException(e);
            }
        }
    }

    private void processSnapshots() {
        for (Iterator it = getThreads().iterator(); it.hasNext();) {
            Thread thread = (Thread) it.next();
            for (int i = 1; i < thread.getNumSnapshots(); i++) {
                for (Iterator it2 = getFunctions(); it2.hasNext();) {
                    Function function = (Function) it2.next();
                    FunctionProfile fp = thread.getFunctionProfile(function);
                    if (fp != null) {
                        double prevEx = fp.getExclusive(i - 1, 0);
                        double prevIn = fp.getInclusive(i - 1, 0);
                        double currEx = fp.getExclusive(i, 0);
                        double currIn = fp.getInclusive(i, 0);
                        fp.setExclusive(i, 0, prevEx + currEx);
                        fp.setInclusive(i, 0, prevIn + currIn);
                    }
                }
            }
        }
    }

    private void initializeThread() {
        nodeID = (nodeID == -1) ? 0 : nodeID;
        contextID = (contextID == -1) ? 0 : contextID;
        threadID = (threadID == -1) ? 0 : threadID;

        // create a thread
        Node node = this.addNode(nodeID);
        Context context = node.addContext(contextID);
        thread = context.addThread(threadID);
    }

    public void incrementThread(String tid, String nid) {
        threadID = Integer.parseInt(tid);
        Integer tmpID = nodeHash.get(nid);
        if (tmpID == null) {
            nodeID = nodeHash.size();
            nodeHash.put(nid, new Integer(nodeID));
        } else {
            nodeID = tmpID.intValue();
        }
        thread = null;
    }
    
    public Thread getThread() {
        if (thread == null) {
            Map attribMap = handler.getAttributes();
            String nct = (String) attribMap.get("nct");
            if (nct != null) {
                String[] nums = nct.split(":");
                nodeID = Integer.parseInt(nums[0]);
                contextID = Integer.parseInt(nums[1]);
                threadID = Integer.parseInt(nums[2]);

            }
            initializeThread();
        }
        return thread;
    }

    private void processHardwareCounter(String metricName, String value) {
        Thread thread = getThread();
        double eventValue = Double.parseDouble(value);

        Metric metric = addMetric(metricName);
        int metricId = metric.getID();

        Function function = addFunction("Entire application");
        FunctionProfile functionProfile = thread.getFunctionProfile(function);
        if (functionProfile == null) {
            functionProfile = new FunctionProfile(function, getNumberOfMetrics());
        }

        thread.addFunctionProfile(functionProfile);
        functionProfile.setExclusive(metricId, eventValue);
        functionProfile.setInclusive(metricId, eventValue);
        functionProfile.setNumCalls(1);
        functionProfile.setNumSubr(0);
    }

}