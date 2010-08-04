package edu.uoregon.tau.perfdmf;

import java.util.HashMap;
import java.util.Map;

import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

public class PeriXMLHandler extends DefaultHandler {
    private StringBuffer accumulator = new StringBuffer();

    private PeriXMLDataSource periXMLDataSource;

    private String procedure;
    private String type;
    private int startline;
    private String module;
    private String name;
    private String description;
    private int id;

    private Thread thread;
    private FunctionProfile functionProfile;
    private Map<Integer, Function> functionMap = new HashMap<Integer, Function>();
    private Map<Integer, Metric> metricMap = new HashMap<Integer, Metric>();

    public PeriXMLHandler(PeriXMLDataSource source) {
        super();
        this.periXMLDataSource = source;
    }

    public int getProgress() {
        return 0;
    }

    public void startDocument() throws SAXException {}

    public void endDocument() throws SAXException {}

    private int getId(Attributes attributes) {
        return Integer.parseInt(getInsensitiveValue(attributes, "id"));
    }

    public void startElement(String uri, String localName, String qName, Attributes attributes) throws SAXException {
        // clear the accumulator
        accumulator.setLength(0);
        if (localName.equalsIgnoreCase("eventDef")) {
            id = getId(attributes);
        } else if (localName.equalsIgnoreCase("metric")) {
            id = getId(attributes);
        } else if (localName.equalsIgnoreCase("process")) {
            id = getId(attributes);
            thread = periXMLDataSource.addThread(id, 0, 0);
        } else if (localName.equalsIgnoreCase("eventInstance")) {
            id = getId(attributes);
            Function function = functionMap.get(new Integer(id));
            functionProfile = new FunctionProfile(function);
            thread.addFunctionProfile(functionProfile);
        } else if (localName.equalsIgnoreCase("metricDef")) {
            id = getId(attributes);
        }

    }

    public void characters(char[] ch, int start, int length) throws SAXException {
        accumulator.append(ch, start, length);
    }

    public void endElement(String uri, String localName, String qName) throws SAXException {

        if (localName.equalsIgnoreCase("name")) {
            name = accumulator.toString();
        } else if (localName.equalsIgnoreCase("metric")) {
            double value = Double.parseDouble(accumulator.toString());
            value = value * 1e6;
            Metric metric = metricMap.get(new Integer(id));
            functionProfile.setExclusive(metric.getID(), value);
            functionProfile.setInclusive(metric.getID(), value);
        } else if (localName.equalsIgnoreCase("description")) {
            name = accumulator.toString();
        } else if (localName.equalsIgnoreCase("procedure")) {
            procedure = accumulator.toString();
        } else if (localName.equalsIgnoreCase("type")) {
            type = accumulator.toString();
        } else if (localName.equalsIgnoreCase("startline")) {
            startline = -1;
            if (!accumulator.toString().equals("")) {
                startline = Integer.parseInt(accumulator.toString());
            }
        } else if (localName.equalsIgnoreCase("module")) {
            module = accumulator.toString();
        } else if (localName.equalsIgnoreCase("metricDef")) {
            Metric metric = periXMLDataSource.addMetric(name);
            metricMap.put(new Integer(id), metric);
        } else if (localName.equalsIgnoreCase("eventDef")) {
            String functionName = procedure + " (" + type + ") " + module;
            Function function = periXMLDataSource.addFunction(functionName);
            // add to map
            functionMap.put(new Integer(id), function);
        }

    }

    private String getInsensitiveValue(Attributes attributes, String key) {
        for (int i = 0; i < attributes.getLength(); i++) {
            if (attributes.getLocalName(i).equalsIgnoreCase(key)) {
                return attributes.getValue(i);
            }
        }
        return null;
    }

}
