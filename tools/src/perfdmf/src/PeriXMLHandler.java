package edu.uoregon.tau.perfdmf;

import java.util.HashMap;
import java.util.Map;

import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

public class PeriXMLHandler extends DefaultHandler {
    private StringBuffer accumulator;

    private PeriXMLDataSource periXMLDataSource;

    private int value;
    private String procedure;
    private String type;
    private int startline;
    private String module;
    private String name;
    private String description;
    private String uniqueId;

    private Map functionMap = new HashMap();
    private Map metricMap = new HashMap();

    public PeriXMLHandler(PeriXMLDataSource source) {
        super();
        this.periXMLDataSource = source;
    }

    public int getProgress() {
        return 0;
    }

    public void startDocument() throws SAXException {}

    public void endDocument() throws SAXException {}

    public void startElement(String uri, String localName, String qName, Attributes attributes) throws SAXException {
        // clear the accumulator
        accumulator.setLength(0);
    }

    public void characters(char[] ch, int start, int length) throws SAXException {
        accumulator.append(ch, start, length);
    }

    public void endElement(String uri, String localName, String qName) throws SAXException {

        if (localName.equalsIgnoreCase("value")) {
            value = Integer.parseInt(accumulator.toString());
        } else if (localName.equalsIgnoreCase("name")) {
            name = accumulator.toString();
        } else if (localName.equalsIgnoreCase("description")) {
            name = accumulator.toString();
        } else if (localName.equalsIgnoreCase("uniqueId")) {
            uniqueId = accumulator.toString();
        } else if (localName.equalsIgnoreCase("procedure")) {
            procedure = accumulator.toString();
        } else if (localName.equalsIgnoreCase("type")) {
            type = accumulator.toString();
        } else if (localName.equalsIgnoreCase("startline")) {
            startline = Integer.parseInt(accumulator.toString());
        } else if (localName.equalsIgnoreCase("module")) {
            module = accumulator.toString();
        } else if (localName.equalsIgnoreCase("metricDef")) {
            Metric metric = periXMLDataSource.addMetric(name);
            metricMap.put(uniqueId, metric);
        } else if (localName.equalsIgnoreCase("eventDef")) {

            String functionName = procedure + " (" + type + ") " + module;

            Function function = periXMLDataSource.addFunction(functionName);

            // add to map
            functionMap.put(new Integer(value), function);
        }

    }

}
