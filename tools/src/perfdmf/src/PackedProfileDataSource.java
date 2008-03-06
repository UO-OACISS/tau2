package edu.uoregon.tau.perfdmf;

import java.io.*;
import java.net.URL;
import java.sql.SQLException;
import java.util.TreeMap;
import java.util.zip.GZIPInputStream;

import edu.uoregon.tau.common.TrackerInputStream;

/**
 * Reads the ParaProf Packed Format (.ppk)
 *    
 * TODO : nothing, this class is complete
 *
 * <P>CVS $Id: PackedProfileDataSource.java,v 1.17 2008/03/06 01:25:56 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.17 $
 */
public class PackedProfileDataSource extends DataSource {

    private File file;
    private volatile boolean abort = false;
    private volatile long totalBytes = 0;
    private volatile long bytesRead = 0;
    private volatile TrackerInputStream tracker;

    /**
     * Constructor for PackedProfileDataSource
     * @param file      file contained packed profile data (.ppk)
     */
    public PackedProfileDataSource(File file) {
        this.file = file;
    }

    /**
     * Cancel load command, causes the load to abort.  Note that most DataSource implementations 
     * do nothing when cancelLoad is called, it is merely a hint.
     * 
     * @see edu.uoregon.tau.perfdmf.DataSource#cancelLoad()
     */
    public void cancelLoad() {
        abort = true;
        return;
    }

    /**
     * Returns the progress of the load as a percentage (0-100)
     * @return          progress
     */
    public int getProgress() {
        if (totalBytes != 0) {
            return (int) ((float) tracker.byteCount() / (float) totalBytes * 100);
        }
        return 0;
    }

    /**
     * Load the data specified at construction.
     * 
     * @see edu.uoregon.tau.perfdmf.DataSource#load()
     */
    public void load() throws FileNotFoundException, IOException, DataSourceException, SQLException {
        long time = System.currentTimeMillis();

        InputStream istream;
        if (file.toString().toLowerCase().startsWith("http:")) {
            // When it gets converted from a String to a File http:// turns into http:/
            URL url = new URL("http://" + file.toString().substring(6).replace('\\', '/'));
            istream = url.openStream();
        }  else {
            istream = new FileInputStream(file);
        }

        tracker = new TrackerInputStream(istream);
        GZIPInputStream gzip = new GZIPInputStream(tracker);
        BufferedInputStream bis = new BufferedInputStream(gzip);
        DataInputStream p = new DataInputStream(bis);

        totalBytes = file.length();

        // read and check magic cookie
        char cookie1 = p.readChar();
        char cookie2 = p.readChar();
        char cookie3 = p.readChar();

        if (!(cookie1 == 'P' && cookie2 == 'P' && cookie3 == 'K')) {
            throw new DataSourceException("This doesn't look like a packed profile");
        }

        // read file version
        int version = p.readInt();

        // read lowest compatibility version
        int compatible = p.readInt();

        if (compatible > 2) {
            throw new DataSourceException("This packed profile is not compatible, please upgrade\nVersion: " + compatible + " > "
                    + "2");
        }

        metaData = new TreeMap();
        if (version >= 2) {
            int metadataHeaderSize = p.readInt(); // older versions will skip over this many bytes

            // skip over next section (future capability)
            int bytesToSkip = p.readInt();
            p.skipBytes(bytesToSkip);
            
            int numTrialMetaData = p.readInt();
            for (int i = 0; i < numTrialMetaData; i++) {
                String name = p.readUTF();
                String value = p.readUTF();
                metaData.put(name,value);
            }
            
            // process thread meta-data
            int numThreads = p.readInt();
            for (int i = 0; i < numThreads; i++) {
                int nodeID = p.readInt();
                int contextID = p.readInt();
                int threadID = p.readInt();

                Thread thread = addThread(nodeID, contextID, threadID);
                int numMetadata = p.readInt();
                for (int j = 0; j < numMetadata; j++) {
                    String name = p.readUTF();
                    String value = p.readUTF();
                    thread.getMetaData().put(name, value);
                    uncommonMetaData.put(name, value);
                }
            }
        } else {
            // skip over header
            int bytesToSkip = p.readInt();
            p.skipBytes(bytesToSkip);
        }

        // process metrics
        int numMetrics = p.readInt();
        for (int i = 0; i < numMetrics; i++) {
            String metricName = p.readUTF();
            this.addMetric(metricName);
        }

        // process groups
        int numGroups = p.readInt();
        if (numGroups != 0) {
            this.setGroupNamesPresent(true);
        }
        Group groups[] = new Group[numGroups];
        for (int i = 0; i < numGroups; i++) {
            groups[i] = this.addGroup(p.readUTF());
        }

        // process functions
        int numFunctions = p.readInt();
        Function functions[] = new Function[numFunctions];
        for (int i = 0; i < numFunctions; i++) {
            String functionName = p.readUTF();
            Function function = this.addFunction(functionName, numMetrics);
            functions[i] = function;

            int numThisGroups = p.readInt();
            for (int j = 0; j < numThisGroups; j++) {
                int thisGroup = p.readInt();
                function.addGroup(groups[thisGroup]);
            }
        }

        // process user events
        int numUserEvents = p.readInt();
        UserEvent userEvents[] = new UserEvent[numUserEvents];
        for (int i = 0; i < numUserEvents; i++) {
            String userEventName = p.readUTF();
            UserEvent userEvent = this.addUserEvent(userEventName);
            userEvents[i] = userEvent;
        }

        // process thread data
        int numThreads = p.readInt();
        for (int i = 0; i < numThreads; i++) {
            int nodeID = p.readInt();
            int contextID = p.readInt();
            int threadID = p.readInt();

            Thread thread = addThread(nodeID, contextID, threadID);
            //((ArrayList)thread.getFunctionProfiles()).ensureCapacity(numFunctions);

            // get function profiles
            int numFunctionProfiles = p.readInt();
            for (int j = 0; j < numFunctionProfiles; j++) {
                int functionID = p.readInt();
                FunctionProfile fp = new FunctionProfile(functions[functionID], numMetrics);
                thread.addFunctionProfile(fp);
                fp.setNumCalls(p.readDouble());
                fp.setNumSubr(p.readDouble());
                for (int k = 0; k < numMetrics; k++) {
                    fp.setExclusive(k, p.readDouble());
                    fp.setInclusive(k, p.readDouble());
                }
            }

            // get user event profiles
            int numUserEventProfiles = p.readInt();
            for (int j = 0; j < numUserEventProfiles; j++) {
                int userEventID = p.readInt();
                UserEventProfile uep = new UserEventProfile(userEvents[userEventID]);

                uep.setNumSamples(p.readInt());
                uep.setMinValue(p.readDouble());
                uep.setMaxValue(p.readDouble());
                uep.setMeanValue(p.readDouble());
                uep.setSumSquared(p.readDouble());
                uep.updateMax();
                thread.addUserEventProfile(uep);
            }
        }

        // close streams
        p.close();
        gzip.close();
        tracker.close();
        istream.close();

        if (numUserEvents > 0) {
            setUserEventsPresent(true);
        }

        this.generateDerivedData(); // mean, percentages, etc.
        this.buildXMLMetaData();

        time = (System.currentTimeMillis()) - time;
        //System.out.println("Time to process (in milliseconds): " + time);
    }
}
