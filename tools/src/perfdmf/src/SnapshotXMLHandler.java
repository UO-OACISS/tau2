package edu.uoregon.tau.perfdmf;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.StringTokenizer;

import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.SAXParseException;
import org.xml.sax.helpers.DefaultHandler;

/**
 * XML Handler for snapshot profiles, this is where all the work is done
 *
 * <P>CVS $Id: SnapshotXMLHandler.java,v 1.15 2010/04/22 21:49:37 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.15 $
 */
public class SnapshotXMLHandler extends DefaultHandler {

    private SnapshotDataSource dataSource;

    private Map<String, ThreadData> threadMap = new HashMap<String, ThreadData>();

    private ThreadData currentThread;
    private Snapshot currentSnapshot;
    //private Date currentDate;

    private int currentMetrics[];

    private StringBuffer accumulator = new StringBuffer();

    private int currentId;
    private String currentName;
    private String currentValue;
    private String currentGroup;
    private long currentTimestamp;
    
    private boolean unified;
    private ThreadData unifiedDefinitions;

    private static class ThreadData {
        public Thread thread;
        public List<Metric> metricMap = new ArrayList<Metric>();
        public List<Function> eventMap = new ArrayList<Function>();
        public List<UserEvent> userEventMap = new ArrayList<UserEvent>();
    }

    public SnapshotXMLHandler(SnapshotDataSource source) {
        this.dataSource = source;
    }

    public void warning(SAXParseException e) throws SAXException {}

    public void error(SAXParseException e) throws SAXException {}

    public void fatalError(SAXParseException e) throws SAXException {
        throw e;
    }

    public void startDocument() throws SAXException {}

    public void endDocument() throws SAXException {}

    private void handleMetric(String name) {
        Metric metric = dataSource.addMetric(name);
        
        while (currentId >= currentThread.metricMap.size()) {
            currentThread.metricMap.add(null);
        }
        currentThread.metricMap.set(currentId, metric);
    }

    private void handleEvent(String name, String groups) {
        int id = currentId;

        Function function = dataSource.addFunction(name);
        dataSource.addGroups(groups, function);
        
        while (id >= currentThread.eventMap.size()) {
            currentThread.eventMap.add(null);
        }
        currentThread.eventMap.set(id, function);
    }

    private void handleUserEvent(String name) {
        int id = currentId;
        UserEvent userEvent = dataSource.addUserEvent(name);
        while (id >= currentThread.userEventMap.size()) {
            currentThread.userEventMap.add(null);
        }
        currentThread.userEventMap.set(id, userEvent);
    }

    private void handleThread(Attributes attributes) {
        String threadName = attributes.getValue("id");
        int nodeID = Integer.parseInt(attributes.getValue("node"));
        int contextID = Integer.parseInt(attributes.getValue("context"));
        int threadID = Integer.parseInt(attributes.getValue("thread"));

        ThreadData data = new ThreadData();
        if (unified) {
            data.eventMap = unifiedDefinitions.eventMap;
            data.userEventMap = unifiedDefinitions.userEventMap;
        }
        data.thread = dataSource.addThread(nodeID, contextID, threadID);
        threadMap.put(threadName, data);
        currentThread = data;
    }

    private void handleDefinitions(Attributes attributes) {
        String threadID = attributes.getValue("thread");
        if (threadID.equals("*")) {
            System.out.println("Unified format found!");
            unified = true;
            unifiedDefinitions = new ThreadData();
            currentThread = unifiedDefinitions;
        } else {
            currentThread = threadMap.get(threadID);
        }
    }

    private void handleProfile(Attributes attributes) {
        String threadID = attributes.getValue("thread");
        currentThread = threadMap.get(threadID);
        currentSnapshot = currentThread.thread.addSnapshot("");
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

    private void handleAtomicDataEnd() {
        String data = accumulator.toString();

        StringTokenizer tokenizer = new StringTokenizer(data, " \t\n\r");

        while (tokenizer.hasMoreTokens()) {
            int eventID = Integer.parseInt(tokenizer.nextToken());
            UserEvent userEvent = currentThread.userEventMap.get(eventID);

            UserEventProfile uep = currentThread.thread.getUserEventProfile(userEvent);
            if (uep == null) {
                uep = new UserEventProfile(userEvent, currentThread.thread.getSnapshots().size());
                currentThread.thread.addUserEventProfile(uep);
            }

            double numSamples = Double.parseDouble(tokenizer.nextToken());
            double max = Double.parseDouble(tokenizer.nextToken());
            double min = Double.parseDouble(tokenizer.nextToken());
            double mean = Double.parseDouble(tokenizer.nextToken());
            double sumSqr = Double.parseDouble(tokenizer.nextToken());
            uep.setNumSamples(numSamples);
            uep.setMaxValue(max);
            uep.setMinValue(min);
            uep.setMeanValue(mean);
            uep.setSumSquared(sumSqr);
        }
    }

    private void handleIntervalDataEnd() {
        String data = accumulator.toString();

        StringTokenizer tokenizer = new StringTokenizer(data, " \t\n\r");

        while (tokenizer.hasMoreTokens()) {
            int eventID = Integer.parseInt(tokenizer.nextToken());

            Function function = currentThread.eventMap.get(eventID);

            FunctionProfile fp = currentThread.thread.getFunctionProfile(function);
            if (fp == null) {
                fp = new FunctionProfile(function, dataSource.getNumberOfMetrics(), currentThread.thread.getSnapshots().size());
                currentThread.thread.addFunctionProfile(fp);
            }

            double numcalls = Double.parseDouble(tokenizer.nextToken());
            double numsubr = Double.parseDouble(tokenizer.nextToken());

            for (int i = 0; i < currentMetrics.length; i++) {
                int metricID = currentMetrics[i];
                Metric metric = currentThread.metricMap.get(metricID);
                double exclusive = Double.parseDouble(tokenizer.nextToken());
                double inclusive = Double.parseDouble(tokenizer.nextToken());
                fp.setExclusive(metric.getID(), exclusive);
                fp.setInclusive(metric.getID(), inclusive);
            }

            fp.setNumCalls(numcalls);
            fp.setNumSubr(numsubr);
        }
    }

    public void startElement(String uri, String localName, String qName, Attributes attributes) throws SAXException {
        //System.out.println("startElement: uri:" + uri + ", localName:"+localName+", qName:"+qName);

        if (localName.equals("thread")) {
            handleThread(attributes);
        } else if (localName.equals("name")) {
            accumulator = new StringBuffer();
        } else if (localName.equals("value")) {
            accumulator = new StringBuffer();
        } else if (localName.equals("group")) {
            accumulator = new StringBuffer();
        } else if (localName.equals("utc_date")) {
            accumulator = new StringBuffer();
        } else if (localName.equals("timestamp")) {
            accumulator = new StringBuffer();
        } else if (localName.equals("definitions")) {
            handleDefinitions(attributes);
        } else if (localName.equals("metric")) {
            currentId = Integer.parseInt(attributes.getValue("id"));
        } else if (localName.equals("event")) {
            currentId = Integer.parseInt(attributes.getValue("id"));
        } else if (localName.equals("userevent")) {
            currentId = Integer.parseInt(attributes.getValue("id"));
        } else if (localName.equals("profile")) {
            handleProfile(attributes);
        } else if (localName.equals("interval_data")) {
            handleIntervalData(attributes);
            accumulator = new StringBuffer();
        } else if (localName.equals("atomic_data")) {
            accumulator = new StringBuffer();
        }

    }

    public void endElement(String uri, String localName, String qName) throws SAXException {
        //System.out.println("endElement: uri:" + uri + ", localName:"+localName+", qName:"+qName);
        if (localName.equals("thread_definition")) {
            currentThread = null;
        } else if (localName.equals("name")) {
            currentName = accumulator.toString();
        } else if (localName.equals("value")) {
            currentValue = accumulator.toString();
        } else if (localName.equals("utc_date")) {
            try {
                //currentDate = 
                	DataSource.dateTime.parse(accumulator.toString());
            } catch (java.text.ParseException e) {}
        } else if (localName.equals("timestamp")) {
            currentTimestamp = Long.parseLong(accumulator.toString());
        } else if (localName.equals("group")) {
            currentGroup = accumulator.toString();
        } else if (localName.equals("profile")) {
            currentSnapshot.setName(currentName);
            currentSnapshot.setTimestamp(currentTimestamp);
        } else if (localName.equals("metric")) {
            handleMetric(currentName);
        } else if (localName.equals("event")) {
            handleEvent(currentName, currentGroup);
        } else if (localName.equals("userevent")) {
            handleUserEvent(currentName);
        } else if (localName.equals("attribute")) {
            currentThread.thread.getMetaData().put(currentName, currentValue);
        } else if (localName.equals("interval_data")) {
            handleIntervalDataEnd();
        } else if (localName.equals("atomic_data")) {
            handleAtomicDataEnd();
        }

    }

    public void characters(char[] ch, int start, int length) throws SAXException {
        accumulator.append(ch, start, length);
    }
}
