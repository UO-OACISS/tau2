package edu.uoregon.tau.perfdmf;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.sql.SQLException;

import org.xml.sax.InputSource;
import org.xml.sax.SAXException;
import org.xml.sax.XMLReader;
import org.xml.sax.helpers.XMLReaderFactory;

/**
 * Snapshot data reader, the real work is done in the XML Handler
 *
 * <P>CVS $Id: SnapshotDataSource.java,v 1.2 2007/01/04 01:34:36 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.2 $
 */
public class SnapshotDataSource extends DataSource {

    public void cancelLoad() {
        // TODO Auto-generated method stub

    }

    public int getProgress() {
        // TODO Auto-generated method stub
        return 0;
    }

    public void load() throws FileNotFoundException, IOException, DataSourceException, SQLException {
        try {
            XMLReader xmlreader = XMLReaderFactory.createXMLReader("org.apache.xerces.parsers.SAXParser");

            SnapshotXMLHandler handler = new SnapshotXMLHandler(this);

            xmlreader.setContentHandler(handler);
            xmlreader.setErrorHandler(handler);
            //xmlreader.parse(new InputSource(new FileInputStream("/home/amorris/profile.xml")));
            xmlreader.parse(new InputSource(new FileInputStream("/home/amorris/snapshot.0.0.0")));
            xmlreader.parse(new InputSource(new FileInputStream("/home/amorris/snapshot.1.0.0")));
            xmlreader.parse(new InputSource(new FileInputStream("/home/amorris/snapshot.2.0.0")));
            xmlreader.parse(new InputSource(new FileInputStream("/home/amorris/snapshot.3.0.0")));

            this.generateDerivedData();
            
            
        } catch (SAXException e) {
            e.printStackTrace();
            throw new DataSourceException(e);

        }

    }

    public static void main(String args[]) {

        
        SnapshotDataSource dataSource = new SnapshotDataSource();
        try {
            dataSource.load();
            
            Function f = dataSource.getFunction("main");

            Thread zero = dataSource.getThread(0,0,0);
            FunctionProfile fp = zero.getFunctionProfile(f);
            System.out.println("main exclusive = " + fp.getExclusive(0));
            System.out.println("main inclusive = " + fp.getInclusive(0));
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
