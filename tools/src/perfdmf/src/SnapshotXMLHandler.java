package edu.uoregon.tau.perfdmf;

import java.util.*;

import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

/**
 * XML Handler for snapshot profiles, this is where all the work is done
 *
 * <P>CVS $Id: SnapshotXMLHandler.java,v 1.2 2007/01/04 01:34:36 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.2 $
 */
public class SnapshotXMLHandler extends DefaultHandler {

    private SnapshotDataSource dataSource;

    private Map threadMap = new HashMap();

    private ThreadData currentThread;
    private Snapshot currentSnapshot;
    
    private int currentMetrics[];
    
    StringBuffer accumulator = new StringBuffer();
    
    private static class ThreadData {
        public Thread thread;
        public Map metricMap = new HashMap();
        public Map eventMap = new HashMap();
    }

    SnapshotXMLHandler(SnapshotDataSource source) {
        this.dataSource = source;
    }

    public void startDocument() throws SAXException {
        //System.out.println("startDocument");
    }

    public void endDocument() throws SAXException {
        //System.out.println("endDocument");
    }

    private void handleMetric(Attributes attributes) {
        int id = Integer.parseInt(attributes.getValue("id"));
        String name = attributes.getValue("name");

        Metric metric = dataSource.addMetric(name);

        currentThread.metricMap.put(new Integer(id), metric);

        //System.out.println("metric definition, id = " + id);
        //dataSource.ad

    }
    private void handleEvent(Attributes attributes) {
        int id = Integer.parseInt(attributes.getValue("id"));
        String name = attributes.getValue("name");
        String groups = attributes.getValue("group");

        Function function = dataSource.addFunction(name);
        dataSource.addGroups(groups, function);
        currentThread.eventMap.put(new Integer(id), function);
    }
    private void handleThread(Attributes attributes) {
        String threadName = attributes.getValue("id");
        int nodeID = Integer.parseInt(attributes.getValue("node"));
        int contextID = Integer.parseInt(attributes.getValue("context"));
        int threadID = Integer.parseInt(attributes.getValue("thread"));

        ThreadData data = new ThreadData();
        data.thread = dataSource.addThread(nodeID, contextID, threadID);

        threadMap.put(threadName, data);

    }

    private void handleDefinitions(Attributes attributes) {
        String threadID = attributes.getValue("thread");
        currentThread = (ThreadData) threadMap.get(threadID);
    }
   
    private void handleProfile(Attributes attributes) {
        String threadID = attributes.getValue("thread");
        currentThread = (ThreadData) threadMap.get(threadID);
        if (currentThread.thread.getSnapshots().size() != 0) {
            //???
            // increase storage
            System.err.println("todo: add snapshot structures for 'Thread'");
            for (Iterator e6 = currentThread.thread.getFunctionProfiles().iterator(); e6.hasNext();) {
                FunctionProfile fp = (FunctionProfile) e6.next();
                if (fp != null) { // fp == null would mean this thread didn't call this function
                    fp.addSnapshot();
                }
            }
        }
        
        currentSnapshot = currentThread.thread.addSnapshot("");
        
    }
    
    public void startElement(String uri, String localName, String qName, Attributes attributes) throws SAXException {
        //System.out.println("startElement: uri:" + uri + ", localName:"+localName+", qName:"+qName);

        if (localName.equals("thread")) {
            handleThread(attributes);
        } else if (localName.equals("name")) {
            accumulator = new StringBuffer();
        } else if (localName.equals("definitions")) {
            handleDefinitions(attributes);
        } else if (localName.equals("metric")) {
            handleMetric(attributes);
        } else if (localName.equals("event")) {
            handleEvent(attributes);
        } else if (localName.equals("profile")) {
            handleProfile(attributes);
        } else if (localName.equals("interval_data")) {
            handleIntervalData(attributes);
            accumulator = new StringBuffer();
        }

    }

    
    private void handleIntervalData(Attributes attributes) {
        String metrics = attributes.getValue("metrics");
  
        StringTokenizer tokenizer = new StringTokenizer(metrics, " \t\n\r");

        currentMetrics = new int[tokenizer.countTokens()];
        int index = 0;
        while (tokenizer.hasMoreTokens()) {
            int metricID = Integer.parseInt(tokenizer.nextToken());
            currentMetrics[index++] = metricID;
        }
    }
    
    private void handleIntervalDataEnd() {
        String data = accumulator.toString();

        StringTokenizer tokenizer = new StringTokenizer(data, " \t\n\r");
        
        while (tokenizer.hasMoreTokens()) {
            int eventID = Integer.parseInt(tokenizer.nextToken());

            
            Function function = (Function) currentThread.eventMap.get(new Integer(eventID));
            
            FunctionProfile fp = currentThread.thread.getFunctionProfile(function);
            if (fp == null) {
                fp = new FunctionProfile(function, dataSource.getNumberOfMetrics(), currentThread.thread.getSnapshots().size());
                currentThread.thread.addFunctionProfile(fp);
            }

            double numcalls = Double.parseDouble(tokenizer.nextToken());
            double numsubr = Double.parseDouble(tokenizer.nextToken());

            for (int i = 0; i < currentMetrics.length; i++) {
                int metricID = currentMetrics[i];
                Metric metric = (Metric)currentThread.metricMap.get(new Integer(metricID));
                double exclusive = Double.parseDouble(tokenizer.nextToken());
                double inclusive = Double.parseDouble(tokenizer.nextToken());
                fp.setExclusive(metric.getID(), exclusive);
                fp.setInclusive(metric.getID(), inclusive);
            }
            
            
            fp.setNumCalls(numcalls);
            fp.setNumSubr(numsubr);
            
            //System.out.println("item = "+ item);
        }
        
    }
    public void endElement(String uri, String localName, String qName) throws SAXException {
        //System.out.println("endElement: uri:" + uri + ", localName:"+localName+", qName:"+qName);
        if (localName.equals("thread_definition")) {
            currentThread = null;
        } else if (localName.equals("name")) {
            System.out.println("reading snapshot: " + accumulator);
            currentSnapshot.setName(accumulator.toString());
        } else if (localName.equals("interval_data")) {
            handleIntervalDataEnd();
        }

    }

    public void characters(char[] ch, int start, int length) throws SAXException {
        accumulator.append(ch, start, length);
    }
}
