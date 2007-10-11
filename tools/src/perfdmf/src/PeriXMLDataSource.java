package edu.uoregon.tau.perfdmf;

import java.io.*;
import java.sql.SQLException;

import org.xml.sax.InputSource;
import org.xml.sax.SAXException;
import org.xml.sax.XMLReader;
import org.xml.sax.helpers.XMLReaderFactory;

public class PeriXMLDataSource extends DataSource {

    private File file;
    
    private volatile PeriXMLHandler handler = new PeriXMLHandler(this);
    
    public PeriXMLDataSource(File file) {
        this.file = file;
    }
    
    public void cancelLoad() {
        // TODO Auto-generated method stub
        
    }

   

    public void load() throws FileNotFoundException, IOException, DataSourceException, SQLException {
        // TODO Auto-generated method stub
        try {
            long time = System.currentTimeMillis();

            XMLReader xmlreader = XMLReaderFactory.createXMLReader("org.apache.xerces.parsers.SAXParser");

            
            xmlreader.setContentHandler(handler);
            xmlreader.setErrorHandler(handler);
            xmlreader.parse(new InputSource(new FileInputStream(file)));
            
            
            this.setGroupNamesPresent(true);

            this.generateDerivedData();

            time = (System.currentTimeMillis()) - time;
            //System.out.println("Time to process (in milliseconds): " + time);

        } catch (SAXException e) {
            throw new DataSourceException(e);
        }
    }
    
    public int getProgress() {
        int value = 0;
        if (handler != null) {
            value = handler.getProgress();
        }
        return value;
    }

}
