package edu.uoregon.tau.perfdmf;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.Map;

import org.xml.sax.*;
import org.xml.sax.helpers.DefaultHandler;
import org.xml.sax.helpers.XMLReaderFactory;

import edu.uoregon.tau.common.XMLCleanWrapInputStream;

public class MetaDataParser {

    private static class XMLParser extends DefaultHandler {
        private StringBuffer accumulator = new StringBuffer();

        private String currentName = "";

        private Map metadataMap;

        public XMLParser(Map metadataMap) {
            this.metadataMap = metadataMap;
        }

        public void startDocument() throws SAXException {
        //            System.out.println("startDocument");
        }

        public void endDocument() throws SAXException {
        //            System.out.println("endDocument");
        }

        public void startElement(String uri, String localName, String qName, Attributes attributes) throws SAXException {
            accumulator = new StringBuffer();
        }

        public void endElement(String uri, String localName, String qName) throws SAXException {
            if (localName.equals("name")) {
                currentName = accumulator.toString().trim();
            } else if (localName.equals("value")) {
                String currentValue = accumulator.toString().trim();
                metadataMap.put(currentName, currentValue);
            }
        }

        public void characters(char[] ch, int start, int length) throws SAXException {
            accumulator.append(ch, start, length);
        }
    }

  

    public static void parse(Map metadataMap, String string) {
        //        System.out.println("parse: " + string);
        try {

            XMLReader xmlreader = XMLReaderFactory.createXMLReader("org.apache.xerces.parsers.SAXParser");

            XMLParser parser = new XMLParser(metadataMap);

            xmlreader.setContentHandler(parser);
            xmlreader.setErrorHandler(parser);

            ByteArrayInputStream input = new ByteArrayInputStream(string.getBytes());

            xmlreader.parse(new InputSource(new XMLCleanWrapInputStream(input)));

        } catch (SAXException saxe) {
            throw new RuntimeException(saxe);
        } catch (IOException ioe) {
            throw new RuntimeException(ioe);
        }

    }

}
