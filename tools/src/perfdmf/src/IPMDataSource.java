/*
 * Name: IPMDataSource.java 
 * Author: Kevin Huck 
 * Description: Parse an psrun XML
 * data file.
 */

/*
 * To do:
 */

package edu.uoregon.tau.perfdmf;

import java.io.File;
import java.io.FileInputStream;
import java.io.StringReader;
import java.util.*;

import org.xml.sax.EntityResolver;
import org.xml.sax.InputSource;
import org.xml.sax.XMLReader;
import org.xml.sax.helpers.DefaultHandler;
import org.xml.sax.helpers.XMLReaderFactory;

public class IPMDataSource extends DataSource {
    private int metric = 0;
    private Function function = null;
    private FunctionProfile functionProfile = null;
    private Node node = null;
    private Context context = null;
    private Thread thread = null;
    private int nodeID = -1;
    private int contextID = -1;
    private int threadID = -1;
    private List v = null;
    boolean initialized = false;
    private Hashtable nodeHash = new Hashtable();
    private int threadCounter = 0;
	private File file = null;

    public IPMDataSource(File file) {
        super();
        this.setMetrics(new Vector<Metric>());
		this.file = file;
    }

    public void cancelLoad() {
        return;
    }

    public int getProgress() {
        return 0;
    }

    public void load() throws DataSourceException {
        try {
            // create our XML parser
            XMLReader xmlreader = XMLReaderFactory.createXMLReader("org.apache.xerces.parsers.SAXParser");
            DefaultHandler handler = new IPMXMLHandler(this);
            xmlreader.setContentHandler(handler);
            xmlreader.setErrorHandler(handler);
            xmlreader.setEntityResolver(new NoOpEntityResolver());
            long time = System.currentTimeMillis();

			nodeID = 0;
    		contextID = 0;
			threadID = 0;

            // parse the next file
            xmlreader.parse(new InputSource(new FileInputStream(file)));

            IPMXMLHandler tmpHandler = (IPMXMLHandler) handler;

            time = (System.currentTimeMillis()) - time;
            //System.out.println("Time to process file (in milliseconds): " + time);
            this.generateDerivedData();
    		this.aggregateMetaData();
    		this.buildXMLMetaData();
			this.setGroupNamesPresent(true);
    	} catch (Exception e) {
            if (e instanceof DataSourceException) {
                throw (DataSourceException)e;
            } else {
                throw new DataSourceException(e);
            }
        }
    }

    public void initializeThread(boolean increment) {
		if (increment) {
			nodeID++;
		} else {
        	// make sure we start at zero for all counters
        	nodeID = (nodeID == -1) ? 0 : nodeID;
        	contextID = (contextID == -1) ? 0 : contextID;
        	threadID = (threadID == -1) ? 0 : threadID;
		}

        //Get the node,context,thread.
        node = this.getNode(nodeID);
        if (node == null)
            node = this.addNode(nodeID);
        context = node.getContext(contextID);
        if (context == null)
            context = node.addContext(contextID);
        thread = context.getThread(threadID);
        if (thread == null) {
            thread = context.addThread(threadID);
        }
    }

    public Thread getThread() {
        return thread;
    }

}