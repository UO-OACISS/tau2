package edu.uoregon.tau.perfdmf;

import java.util.HashMap;
import java.util.Hashtable;
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

    private int sequence;

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
        if (name.equalsIgnoreCase("annotation")) {
            String annotation = accumulator.toString();
            //System.out.println("Got annotation: " + annotation);
            processAnnotation(annotation);

        } else if (name.equalsIgnoreCase("hwpcevent")) {
            if (!isProfile) {
                metricValue = accumulator.toString();
                metricHash.put(metricName, new String(metricValue));
            } else {
                dataSource.addMetric(metricName);
            }
        } else if (name.equalsIgnoreCase("line")) {
            double value = Double.parseDouble(accumulator.toString());
            totalProfileValue += value;
            Function function = dataSource.addFunction(functionName + ":" + fileName + ":" + lineno);
            FunctionProfile fp = dataSource.getThread().getFunctionProfile(function);
            if (fp == null) {
                fp = new FunctionProfile(function, dataSource.getNumberOfMetrics(),
                        dataSource.getThread().getNumSnapshots());
            }
            dataSource.getThread().addFunctionProfile(fp);
            fp.setExclusive(sequence, 0, value);
            fp.setInclusive(sequence, 0, value);
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
