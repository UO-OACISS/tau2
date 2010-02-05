package edu.uoregon.tau.perfdmf;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

/*** SAX Handler which creates SQL to load a xml document into the database. ***/

public class PSRunLoadHandler extends DefaultHandler {
    private StringBuffer accumulator = new StringBuffer();

    protected String currentElement;
    protected String metricName;
    protected String metricValue;
    protected Hashtable metricHash = new Hashtable();

    private String fileName;
    private String functionName;
    private String lineno;

    private boolean isProfile;

    private PSRunDataSource dataSource;

    private double totalProfileValue;
    private Map attribMap = new HashMap();

    private String pid = "-1";
    private String thread = "-1";
    private Map metadata = new HashMap();
    private List metaPrefix = new ArrayList();
    private boolean inMetadata = false;
    private int cacheIndex = 0;

    private int sequence;

    private double wallticks = 1;
    private double clockspeed = 1;
    private double wallclock = 1;
    private double totalsamples = 1;
    
    private boolean itimer;

    public PSRunLoadHandler(PSRunDataSource dataSource) {
        super();
        this.dataSource = dataSource;
    }

    public void startDocument() throws SAXException {}

    private String getInsensitiveValue(Attributes attributes, String key) {
        for (int i = 0; i < attributes.getLength(); i++) {
            if (attributes.getLocalName(i).equalsIgnoreCase(key)) {
                return attributes.getValue(i);
            }
        }
        return null;
    }

    public void endDocument() throws SAXException {
        super.endDocument();
        if (isProfile) {

            Function function = dataSource.getFunction("Entire application");
            if (function == null) {
                function = dataSource.addFunction("Entire application", 1);
            }

            FunctionProfile fp = new FunctionProfile(function, dataSource.getNumberOfMetrics(),
                    dataSource.getThread().getNumSnapshots());
            dataSource.getThread().addFunctionProfile(fp);
            fp.setExclusive(0, 0);
            fp.setInclusive(0, totalProfileValue);
        }
    }

    public void startElement(String url, String name, String qname, Attributes attrList) throws SAXException {
        accumulator.setLength(0);

        if (name.equalsIgnoreCase("hwpcprofile")) {
            isProfile = true;
            inMetadata = false;
        } else if (name.equalsIgnoreCase("hwpcprofiledata")) {
            throw new DataSourceException(
                    "<html><center>This PerfSuite XML file contains unprocessed profile data.<br>Please use `psprocess -x` (from PerfSuite) to generate a processed XML file.</center></html>");
        } else if (name.equalsIgnoreCase("hwpcevent")) {
            currentElement = "hwpcevent";
            metricName = attrList.getValue("name");
        } else if (name.equalsIgnoreCase("file")) {
            fileName = getInsensitiveValue(attrList, "name");
        } else if (name.equalsIgnoreCase("function")) {
            functionName = getInsensitiveValue(attrList, "name");
        } else if (name.equalsIgnoreCase("line")) {
            lineno = getInsensitiveValue(attrList, "lineno");

            // these are for processing multi files
        } else if (name.equalsIgnoreCase("multihwpcprofilereport")) {
            isProfile = true;
        } else if (name.equalsIgnoreCase("hwpcprofilereport")) {
            isProfile = true;
        } else if (name.equalsIgnoreCase("executioninfo")) {
            inMetadata = true;
            cacheIndex = 0;
        } else if (name.equalsIgnoreCase("cache") && inMetadata) {
            cacheIndex++;
            metaPrefix.add(name + cacheIndex + ": ");
            StringBuffer buf = new StringBuffer();
            for (int j = 0; j < metaPrefix.size(); j++) {
                buf.append(metaPrefix.get(j));
            }
            for (int i = 0; i < attrList.getLength(); i++) {
                String tmp = attrList.getQName(i);
                metadata.put(buf.toString() + tmp, attrList.getValue(tmp).trim());
            }
        } else if (name.equalsIgnoreCase("pid")) {} else if (name.equalsIgnoreCase("thread")) {} else if (inMetadata) {
            metaPrefix.add(name + ": ");
            StringBuffer buf = new StringBuffer();
            for (int j = 0; j < metaPrefix.size(); j++) {
                buf.append(metaPrefix.get(j));
            }
            for (int i = 0; i < attrList.getLength(); i++) {
                String tmp = attrList.getQName(i);
                metadata.put(buf.toString() + tmp, attrList.getValue(tmp).trim());
            }
        }
    }

    public void characters(char[] ch, int start, int length) throws SAXException {
        accumulator.append(ch, start, length);
    }

    private void processAnnotation(String annotation) {
        //        System.out.println("process annotation: " + annotation);
        String magic = "TAU^";
        if (annotation.indexOf(magic) == -1) {
            return;
        }
        int index = annotation.indexOf(magic) + magic.length();
        while (index != -1) {
            int next = annotation.indexOf("^", index) + 1;
            int next2 = annotation.indexOf("^", next) + 1;
            if (next <= 0 || next2 <= 0) {
                break;
            }
            String key = annotation.substring(index, next - 1);
            String value = annotation.substring(next, next2 - 1);
            attribMap.put(key, value);
            index = next2;
        }

        String seqString = (String) attribMap.get("seq");
        sequence = Integer.parseInt(seqString);
        Thread thread = dataSource.getThread();
        thread.addSnapshots(sequence + 1);

        String snapName = (String) attribMap.get("phase");
        Snapshot snapshot = (Snapshot) thread.getSnapshots().get(sequence);
        snapshot.setName(snapName);
        long timestamp = Long.parseLong((String) attribMap.get("timestamp"));
        snapshot.setTimestamp(timestamp);
        thread.getMetaData().put("Starting Timestamp", attribMap.get("start"));
    }

    public void endElement(String url, String name, String qname) {

        if (name.equalsIgnoreCase("wallclock")) {
            String foo = accumulator.toString();
            wallticks = Double.parseDouble(foo);
//            System.out.println("wallticks is " + wallticks);
        }

        if (name.equalsIgnoreCase("totalsamples")) {
            String foo = accumulator.toString();
            totalsamples = Double.parseDouble(foo);
//            System.out.println("totalsamples is " + totalsamples);
        }

        if (name.equalsIgnoreCase("clockspeed")) {
            String foo = accumulator.toString();
            clockspeed = Double.parseDouble(foo);
//            System.out.println("clockspeed is " + clockspeed);
            wallclock = wallticks / clockspeed;
//            System.out.println("wallclock is " + wallclock);
        }

        if (name.equalsIgnoreCase("annotation")) {
            String annotation = accumulator.toString();
            //System.out.println("Got annotation: " + annotation);
            processAnnotation(annotation);

        } else if (name.equalsIgnoreCase("hwpcevent")) {
            if (metricName.compareTo("ITIMER_PROF") == 0) { 
                itimer = true;
            }
            if (!isProfile) {
                metricValue = accumulator.toString();
                metricHash.put(metricName, new String(metricValue));
            } else {
                dataSource.addMetric(metricName);
            }
        } else if (name.equalsIgnoreCase("line")) {
            double value = Double.parseDouble(accumulator.toString());
            if (itimer) {
                value = (value / totalsamples) * wallclock;
            }

            totalProfileValue += value;
            Function function = dataSource.addFunction(functionName + ":" + fileName + ":" + lineno);
            FunctionProfile fp = dataSource.getThread().getFunctionProfile(function);
            if (fp == null) {
                fp = new FunctionProfile(function, dataSource.getNumberOfMetrics(), dataSource.getThread().getNumSnapshots());
            }
            dataSource.getThread().addFunctionProfile(fp);
            fp.setExclusive(sequence, 0, value);
            fp.setInclusive(sequence, 0, value);
        } else if (name.equalsIgnoreCase("pid")) {
            pid = accumulator.toString();
            metadata.put(name, pid);
        } else if (name.equalsIgnoreCase("thread")) {
            thread = accumulator.toString();
            dataSource.incrementThread(thread, pid);
            metadata.put(name, thread);
        } else if (name.equalsIgnoreCase("hwpcprofilereport")) {
            Function function = dataSource.getFunction("Entire application");
            if (function == null) {
                function = dataSource.addFunction("Entire application", 1);
            }

            FunctionProfile fp = new FunctionProfile(function, dataSource.getNumberOfMetrics(),
                    dataSource.getThread().getNumSnapshots());
            dataSource.getThread().addFunctionProfile(fp);
            fp.setExclusive(0, 0);
            fp.setInclusive(0, totalProfileValue);
            // end of the report
            Iterator iter = metadata.keySet().iterator();
            String mName;
            String value;
            // get the explicit values in the file
            while (iter.hasNext()) {
                mName = (String) iter.next();
                value = (String) metadata.get(mName);
                dataSource.getThread().getMetaData().put(mName, value.trim());
            }
        } else if (name.equalsIgnoreCase("multihwpcprofilereport")) {} else if (name.equalsIgnoreCase("executioninfo")) {
            // end of the metadata
            inMetadata = false;
            metaPrefix.clear();
        } else if (name.equalsIgnoreCase("machineinfo")) {
            // end of the metadata
            inMetadata = false;
            metaPrefix.clear();
        } else if (inMetadata) {
            metaPrefix.remove(metaPrefix.size() - 1);
            StringBuffer tmp = new StringBuffer();
            for (int i = 0; i < metaPrefix.size(); i++) {
                tmp.append(metaPrefix.get(i));
            }
            metadata.put(tmp + name, accumulator.toString().trim());
        }

    }

    public Map getAttributes() {
        return attribMap;
    }

    public Hashtable getMetricHash() {
        return metricHash;
    }

    public boolean getIsProfile() {
        return isProfile;
    }
}
