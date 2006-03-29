package edu.uoregon.tau.common;

import java.awt.Graphics2D;
import java.io.*;
import java.util.Calendar;
import java.util.TimeZone;

public class EPSWriter {

    private FileOutputStream out;
    private OutputStreamWriter outWriter;
    private BufferedWriter bw;
    private Graphics2D lastG2d;

    // set this to true to enable EPS comments (useful for debugging)
    private static boolean commentsEnabled = false;
    
    public EPSWriter(String name, File file, int width, int height) {
        try {

            out = new FileOutputStream(file);
            outWriter = new OutputStreamWriter(out);
            bw = new BufferedWriter(outWriter);

            Calendar cal = Calendar.getInstance(TimeZone.getDefault());

            String DATE_FORMAT = "yyyy-MM-dd HH:mm:ss";
            java.text.SimpleDateFormat sdf = new java.text.SimpleDateFormat(DATE_FORMAT);
            sdf.setTimeZone(TimeZone.getDefault());

            output("%!PS-Adobe-3.0 EPSF-3.0\n");
            output("%%Creator: TAU\n");
            output("%%Title: " + name + "\n");
            output("%%CreationDate: " + sdf.format(cal.getTime()) + "\n");
            output("%%BoundingBox: 0 0 " + width + " " + height + "\n");
            output("%%DocumentData: Clean7Bit\n");
            output("%%DocumentProcessColors: Black\n");
            output("%%ColorUsage: Color\n");
            output("%%Origin: 0 0\n");
            output("%%Pages: 1\n");
            output("%%Page: 1 1\n");
            output("%%EndComments\n\n");

            output("gsave\n");

            output("-0.0 " + height + " translate\n");
            output("/dialog findfont 12 scalefont setfont\n");

        } catch (Exception e) {
            throw new RuntimeException(e);
        }

    }

    private void output(String str) {
        try {
            bw.write(str);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
    
    public void comment(String str) {
        try {
            if (commentsEnabled) {
                bw.write("% " + str);
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public void write(Graphics2D g2d, String str) {
        if (lastG2d != null && g2d != lastG2d) {
            lastG2d = g2d;
            comment("--------------= graphics mismatch, resetting state BEGIN =--------------\n");
            output("grestore\n");
            g2d.setClip(g2d.getClip());
            g2d.setStroke(g2d.getStroke());
            g2d.setColor(g2d.getColor());
            g2d.setFont(g2d.getFont());
            output("gsave\n");
            comment("--------------= graphics mismatch, resetting state END   =--------------\n");
        }
        output(str);
    }

    public void finish() throws IOException {
        output("grestore\n");
        output("showpage\n");
        output("\n%%EOF");
        bw.close();
        outWriter.close();
        out.close();
    }

}
