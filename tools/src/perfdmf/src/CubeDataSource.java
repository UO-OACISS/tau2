package edu.uoregon.tau.perfdmf;

import java.io.*;
import java.sql.SQLException;

import org.xml.sax.InputSource;
import org.xml.sax.SAXException;
import org.xml.sax.XMLReader;
import org.xml.sax.helpers.XMLReaderFactory;

import edu.uoregon.tau.common.TrackerInputStream;
import java.util.zip.GZIPInputStream;

/**
 * Reader for cube data
 *
 *
 * @see <a href="http://www.fz-juelich.de/zam/kojak/">
 * http://www.fz-juelich.de/zam/kojak/</a> for more information about cube
 * 
 * <P>CVS $Id: CubeDataSource.java,v 1.4 2009/10/06 07:17:55 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.4 $
 */
public class CubeDataSource extends DataSource {

    private File file;
    private volatile CubeXMLHandler handler = new CubeXMLHandler(this);
    private volatile TrackerInputStream tracker;

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


	    FileInputStream fis = new FileInputStream(file);
	    tracker = new TrackerInputStream(fis);
	    
	    InputStream input;
	    
	    // see if it is gzip'd, if not, read directly
	    try {
		GZIPInputStream gzip = new GZIPInputStream(tracker);
		input = gzip;
	    } catch (IOException ioe) {
		fis.close();
		fis = new FileInputStream(file);
		tracker = new TrackerInputStream(fis);
		input = tracker;
	    }


	    xmlreader.parse(new InputSource(new BufferedInputStream(input)));

            
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
