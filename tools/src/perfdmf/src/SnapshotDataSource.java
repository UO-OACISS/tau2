package edu.uoregon.tau.perfdmf;

import java.io.*;
import java.sql.SQLException;
import java.util.List;

import org.xml.sax.InputSource;
import org.xml.sax.SAXException;
import org.xml.sax.XMLReader;
import org.xml.sax.helpers.XMLReaderFactory;

/**
 * Snapshot data reader, the real work is done in the XML Handler
 *
 * <P>CVS $Id: SnapshotDataSource.java,v 1.5 2007/01/18 02:56:08 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.5 $
 */
public class SnapshotDataSource extends DataSource {

    /**
     * Wraps an InputStream with root tags to make it a well-formed document
     * There's probably an easier way to do this 
     */
    private static class RootWrap extends InputStream {
        private InputStream stream;
        private static String before = "<root>";
        private static String after = "</root>";

        private int position = 0;

        private static int BEFORE = 0;
        private static int DURING = 1;
        private static int AFTER = 2;

        private int state = BEFORE;

        public RootWrap(InputStream stream) {
            this.stream = stream;
        }

        public int read() throws IOException {
            if (state == DURING) {
                int retval = stream.read();
                if (retval == -1) {
                    state = AFTER;
                    position = 0;
                }
                return retval;
            } else if (state == BEFORE) {
                int retval = before.charAt(position++);
                if (position == before.length()) {
                    state = DURING;
                }
                return retval;
            } else if (state == AFTER) {
                if (position < after.length()) {
                    return after.charAt(position++);
                } else {
                    return -1;
                }
            }
            return -1;
        }

        public int read(byte[] b) throws IOException {
            if (state == DURING) {
                int retval = stream.read(b);
                if (retval == -1) {
                    state = AFTER;
                    position = 0;
                }
                return retval;
            } else {
                return super.read(b);
            }
        }

        public int read(byte[] b, int off, int len) throws IOException {
            if (state == DURING) {
                int retval = stream.read(b, off, len);
                if (retval == -1) {
                    state = AFTER;
                    position = 0;
                }
                return retval;
            } else {
                return super.read(b, off, len);
            }
        }

    }

    /**
     * A stream wrapper that tracks progress
     */
    class TrackerInputStream extends FilterInputStream {
        private long count;

        public TrackerInputStream(InputStream in) {
            super(in);
        }

        public long byteCount() {
            return count;
        }

        public int read() throws IOException {
            ++count;
            return super.read();
        }

        public int read(byte[] buf) throws IOException {
            count += buf.length;
            return read(buf, 0, buf.length);
        }

        public int read(byte[] buf, int off, int len) throws IOException {
            int actual = super.read(buf, off, len);
            if (actual > 0)
                count += actual;
            return actual;
        }
    }

    private volatile long totalBytes = 0;
    private volatile long bytesRead = 0;
    private volatile TrackerInputStream tracker;

    private File files[];

    public SnapshotDataSource(File files[]) {
        this.files = files;
    }

    public void cancelLoad() {
        // TODO Auto-generated method stub

    }

    public int getProgress() {
        if (totalBytes != 0) {
            return (int) ((float) tracker.byteCount() / (float) totalBytes * 100);
        }
        return 0;
    }

    public void load() throws FileNotFoundException, IOException, DataSourceException, SQLException {
        try {
            long time = System.currentTimeMillis();
            XMLReader xmlreader = XMLReaderFactory.createXMLReader("org.apache.xerces.parsers.SAXParser");

            totalBytes = 0;
            for (int i = 0; i < files.length; i++) {
                totalBytes += files[i].length();
            }

            for (int i = 0; i < files.length; i++) {
                FileInputStream fis = new FileInputStream(files[i]);
                tracker = new TrackerInputStream(fis);
                SnapshotXMLHandler handler = new SnapshotXMLHandler(this);
                xmlreader.setContentHandler(handler);
                xmlreader.setErrorHandler(handler);
                xmlreader.parse(new InputSource(new RootWrap(new BufferedInputStream(tracker))));
            }

            time = System.currentTimeMillis() - time;
            //System.out.println("Snapshot reading took " + time + " ms");
            //System.out.println("found " + this.getThread(0,0,0).getNumSnapshots() + " snapshots");
            this.generateDerivedData();

        } catch (SAXException e) {
            e.printStackTrace();
            throw new DataSourceException(e);
        }
    }

    //    public static void main(String args[]) {
    //
    //        SnapshotDataSource dataSource = new SnapshotDataSource();
    //        try {
    //            dataSource.load();
    //
    //            Function f = dataSource.getFunction("main");
    //
    //            Thread zero = dataSource.getThread(0, 0, 0);
    //            FunctionProfile fp = zero.getFunctionProfile(f);
    //            System.out.println("main exclusive = " + fp.getExclusive(0));
    //            System.out.println("main inclusive = " + fp.getInclusive(0));
    //
    //        } catch (Exception e) {
    //            e.printStackTrace();
    //        }
    //    }
}
