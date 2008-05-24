package edu.uoregon.tau.common;

import java.io.IOException;
import java.io.InputStream;

/**
 * Wraps an InputStream with root tags to make it a well-formed XML document
 * There's probably an easier way to do this 
 */
public class XMLRootWrapInputStream extends InputStream {
    private InputStream stream;
    private static String before = "<root>";
    private static String after = "</root>";

    private int position = 0;

    private static int BEFORE = 0;
    private static int DURING = 1;
    private static int AFTER = 2;

    private int state = BEFORE;

    public XMLRootWrapInputStream(InputStream stream) {
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
