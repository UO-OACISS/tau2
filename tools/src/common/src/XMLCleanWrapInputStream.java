package edu.uoregon.tau.common;

import java.io.IOException;
import java.io.InputStream;

public class XMLCleanWrapInputStream extends InputStream {
    private InputStream stream;

    public static int stripNonValidXMLCharacters(int current) {

        if ((current == 0x9) || (current == 0xA) || (current == 0xD) || ((current >= 0x20) && (current <= 0xD7FF))
                || ((current >= 0xE000) && (current <= 0xFFFD)) || ((current >= 0x10000) && (current <= 0x10FFFF))) {
            return current;
        } else {
            System.out.println("warning: bad char found " + current);
            return ' ';
        }
    }

    public static byte stripNonValidXMLCharacters(byte in) {
        int ret = stripNonValidXMLCharacters((int) in);
        return (byte) ret;
    }

    public XMLCleanWrapInputStream(InputStream stream) {
        this.stream = stream;
    }

    public int read() throws IOException {
        int retval = stripNonValidXMLCharacters(stream.read());
        return retval;
    }

    public int read(byte[] b) throws IOException {
        int retval = stream.read(b);
        for (int i = 0; i < b.length; i++) {
            b[i] = stripNonValidXMLCharacters(b[i]);
        }
        return retval;
    }

    public int read(byte[] b, int off, int len) throws IOException {
        int retval = stream.read(b, off, len);
        for (int i = 0; i < b.length; i++) {
            b[i] = stripNonValidXMLCharacters(b[i]);
        }
        return retval;
    }
}
