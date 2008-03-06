package edu.uoregon.tau.common;

import java.io.FilterInputStream;
import java.io.IOException;
import java.io.InputStream;

public class TrackerInputStream extends FilterInputStream {
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
