package TauIL.util;

import TauIL.absyn.IncludeDec;

import java.util.Vector;

public class InstList {
    public static final int INCLUDE = IncludeDec.INCLUDE, EXCLUDE = IncludeDec.EXCLUDE;
    public static final String [] literals = IncludeDec.literals;

    public String fname = "inst.sel";

    public Vector file_list;
    public Vector event_list;

    public int list_type = EXCLUDE;

    public InstList(Vector files, Vector events) {
	file_list = new Vector(files);
	event_list = new Vector(events);
    }

    public InstList(Vector files, Vector events, int list_type) {
	this(files, events);
	
	this.list_type = list_type;
    }

    public InstList(String fname, Vector files, Vector events) {
	this(files, events);

	this.fname = fname;
    }

    public InstList(String fname, Vector files, Vector events, int list_type) {
	this(files, events, list_type);

	this.fname = fname;
    }
}
