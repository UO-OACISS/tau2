package edu.uoregon.tau.dms.dss;

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
 * <P>CVS $Id: CubeDataSource.java,v 1.1 2005/06/07 01:25:32 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.1 $
 */
public class CubeDataSource extends DataSource {

    private File file;

    /**
     * Constructor for CubeDataSource
     * @param file      file containing cube data
     */
    public CubeDataSource(File file) {
        this.file = file;
    }

    
    
    
    
    public void load() throws FileNotFoundException, IOException, DataSourceException, SQLException {
        try {

            XMLReader xmlreader = XMLReaderFactory.createXMLReader("org.apache.xerces.parsers.SAXParser");

            
            DefaultHandler handler = new CubeXMLHandler(this);
            xmlreader.setContentHandler(handler);
            xmlreader.setErrorHandler(handler);

            // parse the next file
            xmlreader.parse(new InputSource(new FileInputStream(file)));

            
            if (CallPathUtilFuncs.checkCallPathsPresent(this.getFunctions())) {
                setCallPathDataPresent(true);
            }

            this.generateDerivedData();
            
        } catch (SAXException e) {
            throw new DataSourceException(e);
        }
    }

    public int getProgress() {
        // TODO Auto-generated method stub
        return 0;
    }

    public void cancelLoad() {
        // TODO Auto-generated method stub

    }

}
