/************************************************************
 *
 *           File : InstListWriter.java
 *         Author : Tyrel Datwyler
 *
 *    Description : Utility class for generating selective
 *                  instrumentation list files from abstract
 *                  InstList classes for use by the 
 *                  tau_instrumentor utility.
 *
 ************************************************************/

package TauIL.util;

import java.io.FileWriter;
import java.io.BufferedWriter;
import java.io.IOException;

/**
 * This is a utility class that generates TAU selective instrumentation
 * list files for use by the tau_instrumentor utility. The list is
 * generated from the TauIL {@link InstList} internal representation.
 */
public class InstListWriter {
    private static final String BEGIN = "BEGIN_", END = "END_", LIST = "_LIST", FILE = "FILE_";
    private static final String HEADER = "# Selective insturmentation: Specify an exclude/include list.";

    private FileWriter fout;
    private BufferedWriter fbuf;

    public InstListWriter() { }

    /**
     * Generates a selective instrumentation list file from an {@link InstList}.
     *
     * @param list an InstList to generate the file from.
     */
    public void writeList(InstList list) throws IOException {
	fout = new FileWriter(list.fname);
	fbuf = new BufferedWriter(fout);

	writeLine(HEADER);
	fbuf.newLine();

	String type = InstList.literals[list.list_type].toUpperCase();

	write(BEGIN);
	write(FILE);
	write(type);
	writeLine(LIST);

	String [] files = (String []) list.file_list.toArray(new String[0]);

	for (int i = 0; i < files.length; i++)
	    writeLine(files[i]);

	write(END);
	write(FILE);
	write(type);
	writeLine(LIST);

	fbuf.newLine();
	fbuf.flush();

	write(BEGIN);
	write(type);
	writeLine(LIST);

	String [] events = (String []) list.event_list.toArray(new String[0]);

	for (int i = 0; i < events.length; i++)
	    writeLine(events[i]);

	write(END);
	write(type);
	write(LIST);

	fbuf.flush();
	fbuf.close();
    }

    /* Helper methods for writing lines and strings to a file output buffer. */
    private void writeLine(String s) throws IOException {
	fbuf.write(s, 0, s.length());
	fbuf.newLine();
    }

    private void write(String s) throws IOException {
	fbuf.write(s, 0, s.length());
    }
}
