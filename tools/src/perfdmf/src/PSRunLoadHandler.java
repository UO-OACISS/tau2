package edu.uoregon.tau.perfdmf;

import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;
import java.util.Hashtable;

/*** SAX Handler which creates SQL to load a xml document into the database. ***/

public class PSRunLoadHandler extends DefaultHandler {

	protected String currentElement = "";
	protected String metricName = "";
	protected String metricValue = "";
	protected Hashtable metricHash = new Hashtable();

	public PSRunLoadHandler() {
		super();
	}

	/*** Initialize the document table when begining loading a XML document.*/

	public void startDocument() throws SAXException{
    	// nothing needs to be done here.
	}

	/*** Handle element, attributes, and the connection from this element to its parent. ***/

	public void startElement(String url, String name, String qname, Attributes attrList) throws SAXException {
		if( name.equalsIgnoreCase("hwpcevent") ) {
			currentElement = "hwpcevent";
			metricName = attrList.getValue("name");
		// } else if( name.equalsIgnoreCase("other") ) {
			// currentElement = "other";
		}       
	}

	/**
 	* Handle character data regions.
 	*/
	public void characters(char[] chars, int start, int length) {
		// check if characters is whitespace, if so, return
		boolean isWhitespace = true;
		for (int i = start; i < start+length; i++) {		
			if (! Character.isWhitespace(chars[i])) {
				isWhitespace = false;
				break;
			}
		}
		if (isWhitespace == true) {
			return;
		}
		String tempstr = new String(chars, start, length);
		if (currentElement.equals("hwpcevent")) this.metricValue = tempstr;
		// else if (currentElement.equals("other")) other = tempstr;
	}

	public void endElement(String url, String name, String qname) {
		if (name.equalsIgnoreCase("hwpcevent")){
			// save the metric value
			metricHash.put(metricName, new String(metricValue));
		}
	}

	public Hashtable getMetricHash() {
		return metricHash;
	}
}
