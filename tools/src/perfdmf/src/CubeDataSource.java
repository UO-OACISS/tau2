package edu.uoregon.tau.perfdmf;

import java.io.*;
import java.sql.SQLException;
import java.util.Iterator;

import org.xml.sax.InputSource;
import org.xml.sax.SAXException;
import org.xml.sax.XMLReader;
import org.xml.sax.helpers.DefaultHandler;
import org.xml.sax.helpers.XMLReaderFactory;

/**
 * Reader for cube data
 *
 *
 * @see <a href="http://www.fz-juelich.de/zam/kojak/">
 * http://www.fz-juelich.de/zam/kojak/</a> for more information about cube
 * 
 * <P>CVS $Id: CubeDataSource.java,v 1.2 2006/03/29 20:14:38 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.2 $
 */
public class CubeDataSource extends DataSource {

    private File file;
    private volatile CubeXMLHandler handler = new CubeXMLHandler(this);

    /**
     * Constructor for CubeDataSource
     * @param file      file containing cube data
     */
    public CubeDataSource(File file) {
        this.file = file;
    }

    
    public void load() throws FileNotFoundException, IOException, DataSourceException, SQLException {
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

    public void cancelLoad() {
        // TODO Auto-generated method stub

    }

}
