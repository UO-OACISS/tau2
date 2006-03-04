package edu.uoregon.tau.common;

import java.io.*;
import java.util.zip.GZIPInputStream;

public class Gzip {

    public static void gunzip(String input, String output) throws FileNotFoundException, IOException {
        GZIPInputStream gis = new GZIPInputStream(new BufferedInputStream(new FileInputStream(input)));
        BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(output));
        int c;
        while ((c = gis.read()) != -1) {
            bos.write((byte) c);
        }
        gis.close();
        bos.close();
    }

    
    
    
}
