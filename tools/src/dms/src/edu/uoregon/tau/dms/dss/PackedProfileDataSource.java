package edu.uoregon.tau.dms.dss;

import java.io.*;
import java.sql.SQLException;
import java.util.zip.GZIPInputStream;

public class PackedProfileDataSource extends DataSource {

    class TrackerInputStream extends FilterInputStream {
        private int count;

        public TrackerInputStream(InputStream in) {
            super(in);
        }

        public int byteCount() {
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

    private File file;
    private volatile boolean abort = false;
    private volatile long totalBytes = 0;
    private volatile long bytesRead = 0;
    TrackerInputStream tracker;

    public PackedProfileDataSource(File file) {
        this.file = file;
    }

    public void cancelLoad() {
        abort = true;
        return;
    }

    public int getProgress() {
        if (totalBytes != 0)
            return (int) ((float) tracker.byteCount() / (float) totalBytes * 100);
        return 0;
    }

    public void load() throws FileNotFoundException, IOException, DataSourceException, SQLException {
        long time = System.currentTimeMillis();

        FileInputStream istream = new FileInputStream(file);
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
        
        if (compatible != 1) {
            throw new DataSourceException("This packed profile is not compatible, please upgrade\nVersion: " + compatible + " > " + "1");
        }
        
        // skip over header
        int bytesToSkip = p.readInt();
        
        p.skipBytes(bytesToSkip);
        
        
        int numMetrics = p.readInt();

        for (int i = 0; i < numMetrics; i++) {
            String metricName = p.readUTF();
            this.addMetric(metricName);
        }

        int numGroups = p.readInt();
        if (numGroups != 0) {
            this.setGroupNamesPresent(true);
        }
            
        Group groups[] = new Group[numGroups];
        for (int i = 0; i < numGroups; i++) {
            groups[i] = this.addGroup(p.readUTF());
        }
        
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

        int numUserEvents = p.readInt();
        UserEvent userEvents[] = new UserEvent[numUserEvents];
        for (int i = 0; i < numUserEvents; i++) {
            String userEventName = p.readUTF();
            UserEvent userEvent = this.addUserEvent(userEventName);
            userEvents[i] = userEvent;
        }

        int numThreads = p.readInt();


        for (int i = 0; i < numThreads; i++) {

            int nodeID = p.readInt();
            int contextID = p.readInt();
            int threadID = p.readInt();

            Node node = this.addNode(nodeID);
            Context context = node.addContext(contextID);
            Thread thread = context.addThread(threadID, numMetrics);


            int numFunctionProfiles = p.readInt();

            for (int j = 0; j < numFunctionProfiles; j++) {
                int functionID = p.readInt();
                FunctionProfile fp = new FunctionProfile(functions[functionID], numMetrics);
                fp.setNumCalls(p.readDouble());
                fp.setNumSubr(p.readDouble());
                for (int k = 0; k < numMetrics; k++) {
                    fp.setExclusive(k, p.readDouble());
                    fp.setInclusive(k, p.readDouble());
                }
                thread.addFunctionProfile(fp);
            }

            int numUserEventProfiles = p.readInt();

            for (int j = 0; j < numUserEventProfiles; j++) {
                int userEventID = p.readInt();
                UserEventProfile uep = new UserEventProfile(userEvents[userEventID]);

                uep.setUserEventNumberValue(p.readInt());
                uep.setUserEventMinValue(p.readDouble());
                uep.setUserEventMaxValue(p.readDouble());
                uep.setUserEventMeanValue(p.readDouble());
                uep.setUserEventSumSquared(p.readDouble());
                uep.updateMax();

                thread.addUserEvent(uep);
            }
        }

        
        p.close();
        gzip.close();
        tracker.close();
        istream.close();

        if (numUserEvents > 0) {
            setUserEventsPresent(true);
        }

        this.generateDerivedData(); // mean, percentages, etc.

        if (CallPathUtilFuncs.checkCallPathsPresent(this.getFunctions())) {
            setCallPathDataPresent(true);
        }

        //time = (System.currentTimeMillis()) - time;
        //System.out.println("Time to process (in milliseconds): " + time);

    }
}
