package edu.uoregon.tau.perfdmf;

import java.io.ByteArrayInputStream;
import java.io.IOException;

import org.xml.sax.*;
import org.xml.sax.helpers.DefaultHandler;
import org.xml.sax.helpers.XMLReaderFactory;

import edu.uoregon.tau.common.MetaDataMap;
import edu.uoregon.tau.common.XMLCleanWrapInputStream;
import edu.uoregon.tau.common.MetaDataMap.MetaDataKey;

public class MetaDataParser {

    private static class XMLParser extends DefaultHandler {
        private StringBuffer accumulator = new StringBuffer();
        private Thread thread = null;

        //private String currentName = "";
        private MetaDataKey key = null;

        //private Map<String, String> metadataMap;
        private MetaDataMap map = null;

/*        public XMLParser(Map<String, String> metadataMap) {
            this.metadataMap = metadataMap;
        }
*/
        public XMLParser(Thread thread) {
            this.map = new MetaDataMap();
            this.thread = thread;
        }
        
        public XMLParser(MetaDataMap map, Thread thread) {
            this.map = map;
            this.thread = thread;
        }
        
        
        public void startDocument() throws SAXException {
        }

        public void endDocument() throws SAXException {
        }

        public void startElement(String uri, String localName, String qName, Attributes attributes) throws SAXException {
            accumulator = new StringBuffer();
        }

        public void endElement(String uri, String localName, String qName) throws SAXException {
            if (localName.equals("name") || localName.equals("tau:name")) {
            	key = map.newKey(accumulator.toString().trim());
            } else if (localName.equals("value") || localName.equals("tau:value")) {
                String currentValue = accumulator.toString().trim();
                map.put(key, currentValue);
//                key = null;
            } else if (localName.equals("timer_context") || localName.equals("tau:timer_context")) {
            	key.timer_context = accumulator.toString().trim();
            } else if (localName.equals("call_number") || localName.equals("tau:call_number")) {
            	key.call_number = Integer.parseInt(accumulator.toString().trim());
            } else if (localName.equals("timestamp") || localName.equals("tau:timestamp")) {
            	key.timestamp = Long.parseLong(accumulator.toString().trim());
            }
        }

        public void characters(char[] ch, int start, int length) throws SAXException {
            accumulator.append(ch, start, length);
        }
        
        public MetaDataMap getMap() {
        	return this.map;
        }
    }

    public static MetaDataMap parse(MetaDataMap map, String string, Thread thread) {
        try {
            XMLReader xmlreader = XMLReaderFactory.createXMLReader("org.apache.xerces.parsers.SAXParser");
            XMLParser parser = new XMLParser(thread);
            xmlreader.setContentHandler(parser);
            xmlreader.setErrorHandler(parser);
            ByteArrayInputStream input = new ByteArrayInputStream(string.getBytes());
            xmlreader.parse(new InputSource(new XMLCleanWrapInputStream(input)));
            map = parser.getMap();
        } catch (SAXException saxe) {
       		throw new RuntimeException(saxe);
        } catch (IOException ioe) {
            throw new RuntimeException(ioe);
        }
        return map;
    }
    
    public static MetaDataMap parse(String string, Thread thread) {
    	MetaDataMap map = new MetaDataMap();
    	return parse(map, string, thread);
    }

}
