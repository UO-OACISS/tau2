/************************************************************
 *
 *           File : InstList.java
 *         Author : Tyrel Datwyler
 *
 *    Description : Abstract representation of a TAU instrumentation
 *                  list used by the tau_instrumentor utility.
 *
 ************************************************************/

package TauIL.util;

import TauIL.absyn.IncludeDec;

import java.util.Vector;

/**
 * This classes is intended to be an abstract representation of
 * the selective instrumentation list format used by TAU's
 * tau_instrumentor utility. The default behavior is exclusion if
 * not specified, and the default filename for the list is "inst.sel"
 * if not specified. The utility class {@link InstListWriter} can be
 * used to actually generate the selective instrumentation list file
 * from an InstList class.
 */
public class InstList {
    public static final int INCLUDE = IncludeDec.INCLUDE, EXCLUDE = IncludeDec.EXCLUDE;
    public static final String [] literals = IncludeDec.literals;

    public String fname = "inst.sel";

    public Vector file_list;
    public Vector event_list;

    public int list_type = EXCLUDE;

    /**
     * Creates a list of files and a list of events to be excluded. The
     * list is given the default filename "inst.sel".
     *
     * @param files a Vector of files, each element is assumed to be a String.
     * @param events a Vector of events, each element is assumed to be a String.
     */
    public InstList(Vector files, Vector events) {
	file_list = new Vector(files);
	event_list = new Vector(events);
    }

    /**
     * Creates a list of files and a list of events that will either be
     * excluded or included by the instrumentor. The list is givent the 
     * default filename "inst.sel".
     *
     * @param files a Vector of files, each element is assumed to be a String.
     * @param events a Vector of events, each element is assumed to be a String.
     * @param list_type flag specifing inclusion behavior, either InstList.INCLUDE or InstList.EXCLUDE.
     */
    public InstList(Vector files, Vector events, int list_type) {
	this(files, events);
	
	this.list_type = list_type;
    }

    /**
     * Creates a list of files and a list of events to be excluded with the 
     * given file name.
     *
     * @param fname filename for the instrumentation list.
     * @param files a Vector of files, each element is assumed to be a String.
     * @param events a Vector of events, each element is assumed to be a String.
     */
    public InstList(String fname, Vector files, Vector events) {
	this(files, events);

	this.fname = fname;
    }

    /**
     * Creates a list of files and a list of events that will either be
     * excluded or included by the instrumentor with the given filename.
     *
     * @param fname filename for the instrumentation list.
     * @param files a Vector of files, each element is assumed to be a String.
     * @param events a Vector of events, each element is assumed to be a String.
     * @param list_type flag specifing inclusion behavior, either InstList.INCLUDE or InstList.EXCLUDE.
     */
    public InstList(String fname, Vector files, Vector events, int list_type) {
	this(files, events, list_type);

	this.fname = fname;
    }
}
