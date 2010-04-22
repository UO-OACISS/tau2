package edu.uoregon.tau.perfdmf;

import java.io.*;
import java.sql.SQLException;
import java.util.zip.GZIPInputStream;

import org.xml.sax.InputSource;
import org.xml.sax.SAXException;
import org.xml.sax.XMLReader;
import org.xml.sax.helpers.XMLReaderFactory;

import edu.uoregon.tau.common.TrackerInputStream;
import edu.uoregon.tau.common.XMLCleanWrapInputStream;
import edu.uoregon.tau.common.XMLRootWrapInputStream;

/**
 * Snapshot data reader, the real work is done in the XML Handler
 *
 * <P>CVS $Id: SnapshotDataSource.java,v 1.14 2010/04/22 22:01:58 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.14 $
 */
public class SnapshotDataSource extends DataSource {

    private volatile long totalBytes = 0;
    private volatile long bytesRead = 0;
    private volatile TrackerInputStream tracker;
    private File files[];

    public SnapshotDataSource(File files[]) {
        this.files = files;
    }

    public void cancelLoad() {}

    public int getProgress() {
        if (totalBytes != 0 && tracker != null) {
            return (int) ((float) (bytesRead + tracker.byteCount()) / (float) totalBytes * 100);
        }
        return 0;
    }

    public void load() throws FileNotFoundException, IOException, DataSourceException, SQLException {
        String currentFile = null;
        try {
            long time = System.currentTimeMillis();
            XMLReader xmlreader = XMLReaderFactory.createXMLReader("org.apache.xerces.parsers.SAXParser");

            totalBytes = 0;
            for (int i = 0; i < files.length; i++) {
                totalBytes += files[i].length();
            }

            for (int i = 0; i < files.length; i++) {
                FileInputStream fis = new FileInputStream(files[i]);
                currentFile = files[i].toString();
                tracker = new TrackerInputStream(fis);

                InputStream input;

                // see if it is gzip'd, if not, read directly
                try {
                    GZIPInputStream gzip = new GZIPInputStream(tracker);
                    input = gzip;
                } catch (IOException ioe) {
                    fis.close();
                    fis = new FileInputStream(files[i]);
                    tracker = new TrackerInputStream(fis);
                    input = tracker;
                }

                SnapshotXMLHandler handler = new SnapshotXMLHandler(this);
                xmlreader.setContentHandler(handler);
                xmlreader.setErrorHandler(handler);
                xmlreader.parse(new InputSource(new XMLRootWrapInputStream(new XMLCleanWrapInputStream(new BufferedInputStream(input)))));
                bytesRead += files[i].length();
            }

            time = System.currentTimeMillis() - time;
            System.out.println("Snapshot reading took " + time + " ms");
            //System.out.println("found " + this.getThread(0,0,0).getNumSnapshots() + " snapshots");
            this.generateDerivedData();
            this.aggregateMetaData();

        } catch (SAXException e) {
            e.printStackTrace();
            throw new DataSourceException(e, currentFile);
        }
    }

}
