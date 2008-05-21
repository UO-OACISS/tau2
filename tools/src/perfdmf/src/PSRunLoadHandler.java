package edu.uoregon.tau.perfdmf;

import java.util.Hashtable;

import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

/*** SAX Handler which creates SQL to load a xml document into the database. ***/

public class PSRunLoadHandler extends DefaultHandler {
    private StringBuffer accumulator = new StringBuffer();

    protected String currentElement = "";
    protected String metricName = "";
    protected String metricValue = "";
    protected Hashtable metricHash = new Hashtable();

    private String fileName = "";
    private String functionName = "";
    private String lineno = "";

    private boolean isProfile;

    private PSRunDataSource dataSource;

    private double totalProfileValue;

    public PSRunLoadHandler(PSRunDataSource dataSource) {
        super();
        this.dataSource = dataSource;
    }

    public void startDocument() throws SAXException {
    // nothing needs to be done here.
    }

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

            FunctionProfile fp = new FunctionProfile(function);
            dataSource.getThread().addFunctionProfile(fp);
            fp.setExclusive(0, 0);
            fp.setInclusive(0, totalProfileValue);
        }
    }

    public void startElement(String url, String name, String qname, Attributes attrList) throws SAXException {
        accumulator.setLength(0);

        if (name.equalsIgnoreCase("hwpcprofile")) {
            isProfile = true;
        } else if (name.equalsIgnoreCase("hwpcevent")) {
            currentElement = "hwpcevent";
            metricName = attrList.getValue("name");
            // } else if( name.equalsIgnoreCase("other") ) {
            // currentElement = "other";
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

    public void endElement(String url, String name, String qname) {
        if (name.equalsIgnoreCase("hwpcevent")) {
            if (!isProfile) {
                metricValue = accumulator.toString();
                metricHash.put(metricName, new String(metricValue));
            } else {
                dataSource.addMetric(metricName);
            }
        } else if (name.equalsIgnoreCase("line")) {
            double value = Double.parseDouble(accumulator.toString());
            totalProfileValue += value;
            //metricHash.put(fileName + ":" + functionName + ":" + lineno, new String(metricValue));

            //Function function = dataSource.addFunction(fileName + ":" + functionName + ":" + lineno);
            Function function = dataSource.addFunction(functionName + ":" + fileName + ":" + lineno);

            FunctionProfile fp = new FunctionProfile(function);
            dataSource.getThread().addFunctionProfile(fp);
            fp.setExclusive(0, value);
            fp.setInclusive(0, value);

        }

    }

    public Hashtable getMetricHash() {
        return metricHash;
    }

    public boolean getIsProfile() {
        return isProfile;
    }
}
