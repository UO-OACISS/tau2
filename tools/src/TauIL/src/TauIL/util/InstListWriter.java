package TauIL.util;

import java.io.FileWriter;
import java.io.BufferedWriter;
import java.io.IOException;

public class InstListWriter {
    private static final String BEGIN = "BEGIN_", END = "END_", LIST = "_LIST", FILE = "FILE_";
    private static final String HEADER = "# Selective insturmentation: Specify an exclude/include list.";

    private FileWriter fout;
    private BufferedWriter fbuf;

    public InstListWriter() { }

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

    private void writeLine(String s) throws IOException {
	fbuf.write(s, 0, s.length());
	fbuf.newLine();
    }

    private void write(String s) throws IOException {
	fbuf.write(s, 0, s.length());
    }
}

/*
import TauIL.absyn.IncludeDec;

import java.io.PrintStream;
import java.io.OutputStream;
import java.io.IOException;
import java.util.Vector;

public class InstListWriter {

    public static final int INCLUDE = IncludeDec.INCLUDE, EXCLUDE = IncludeDec.EXCLUDE;
    public static final String [] literals = IncludeDec.literals;
    
    private static final String BEGIN = "BEGIN_", END = "END_", LIST = "_LIST";
    private static final String HEADER = "# Selective insturmentation: Specify an exclude/include list.";

    private PrintStream out;
    private int flag = EXCLUDE;

    public InstListWriter(OutputStream out, int flag) throws IOException {
	if (out instanceof PrintStream)
	    this.out = (PrintStream) out;
	else
	    this.out = new PrintStream(out);
	this.flag = flag;

	writeHeader();
    }

    public InstListWriter(OutputStream out) throws IOException {
	this(out, EXCLUDE);
    }

    public void writeList(Vector list) throws IOException {
	String [] events = (String []) list.toArray(new String[0]);

	for (int i = 0; i < events.length; i++)
	    out.println(events[i]);

	if (out.checkError())
	    throw new IOException("I/O error occurred while writing list.");
    }

    public void close() throws IOException {
	writeFooter();
 	if (out != System.out)
	    out.close();

	if (out.checkError())
	    throw new IOException("I/O error occurred while closing list writer.");
    }

    private void writeHeader() throws IOException {
	out.println(HEADER);
	out.println();
	out.println(BEGIN + literals[flag].toUpperCase() + LIST);

	if (out.checkError())
	    throw new IOException("I/O error occurred while writing list header.");
    }

    private void writeFooter() throws IOException {
	out.println(END + literals[flag].toUpperCase() + LIST);

	if (out.checkError())
	    throw new IOException("I/O error occurred while writing list footer.");
    }
}
*/
