package edu.uoregon.tau.common;

import java.io.*;

public class Common {

    // Copies src file to dst file.
    // If the dst file does not exist, it is created
    public static void copy(String src, String dst) throws IOException {
        InputStream in = new FileInputStream(src);
        OutputStream out = new FileOutputStream(dst);

        // Transfer bytes from in to out
        byte[] buf = new byte[1024];
        int len;
        while ((len = in.read(buf)) > 0) {
            out.write(buf, 0, len);
        }
        in.close();
        out.close();
    }

    public static void deltree(File infile)  {
        if (infile.isDirectory()) {
            String[] files = infile.list();
            for (int i = 0; i < files.length; i++) {
                deltree(new File(infile, files[i]));
            }
        } else {
            if (infile.delete()) {
                // you may want to keep track of deleted files here 
            } else {
                // you may want to provide error logging here 
            }
        }
    }
}
