/************************************************************
 *
 *           File : GetOpt.java
 *         Author : Tyrel Datwyler
 *
 *    Description : Used to define and parse command line
 *                  switches. Similar to gnu/posix getopt.
 *
 ************************************************************/

package TauIL.util;

import java.util.Vector;
import java.util.StringTokenizer;
import java.io.StringReader;
import java.io.IOException;

/**
 * This class is intended for defining a set of command line
 * option switches and parsing a set of string arguments against
 * those switches. It is desinged to behave in a similar fashion
 * to the GNU and POSIX getopt implementations. Long options are
 * not currently supported, and the parsing behavior is slightly
 * different.
 *
 * <p>Switches can either be boolean or non-boolean. The non-boolean
 * case always assumes that an argument will be supplied with the
 * switch. Switches are single character and defined similar to GNU
 * POSIX short switches.</p>
 *
 * <p> Example : &nbsp;&nbsp;&nbsp; "cvf:ho:"</p>
 *
 * <p> This example defines five switches. Two of which, namely "f" and "o"
 * are non-boolean and are assumed to take an argument.</p>
 *
 * <p> The format that switches are given on the command line is once
 * again similar to that of the GNU and POSIX getopt implementation.
 * switches must be immediatlly preceded with a "-" and several options
 * may be grouped together. If more than one option precedes the dash
 * all but the last option must be boolean switches. Otherwise, any options
 * immediatlly following a non-boolean option with no whitespace would be
 * assumed to represent the argument to that non-boolean option.</p>
 *
 * <p> Bad example : &nbsp;&nbsp;&nbsp; "-c -vfho test.out"</p>
 *
 * <p> In this example instead of "test.out" being the argument attached to
 * the "o" switch it would be assumed that "ho" is the argument to the "f"
 * switch, and that "test.out" is merely a parameter being passed directly
 * to the program. This is assuming the switches defined in the earlier example.
 * Note that currently parameters are simply grouped into a list, and that
 * command line parsing doesn't halt when a parameter is encouterd.</p>
 *
 * <p> It should be noted that this is a naive implemenation and thus it is 
 * the programmers responsibility to call the optarg() method everytime a 
 * non-boolean switch is returned otherwise the arguments will no longer by 
 * in sync with the remaining options.</p>
 */
public class GetOpt {
    /** End of command line options indicator. **/
    public static final char OPT_EOF = '\0';

    private Vector optlist = new Vector();
    private Vector opts = new Vector();
    private Vector args = new Vector();
    private Vector params = new Vector();
    private int opt_count = 0;
    private int arg_count = 0;
    private int param_count = 0;
    private boolean first_opt = false;

    /**
     * Constructor for creating an instance that will parse a set of
     * String arguments with respect to a set of command line options.
     * Option switches are required to be a single character. By default
     * switches are assumed to be boolean unless followed by a colon.
     * <p>example : &nbsp;&nbsp;&nbsp "cvf:"</p>
     * <p>In this example "c" and "v" are boolean while "f" is non-boolean
     * and thus assumed to take an argument.
     *
     * @param sargs array of String arguments to be parsed.
     * @param flags string definition of option switches.
     */
    public GetOpt(String [] sargs, String flags) throws BadOptException {
	int size = sargs.length;

	try {
	    StringReader sreader = new StringReader(flags);
	    for (int i = 0; i < flags.length(); i++) {
		int opt = sreader.read();
		
		if (opt == ':')
		   ((OptPair) optlist.elementAt(optlist.size() - 1)).bool = false;
		else
		    optlist.add(new OptPair((char) opt, true));
	    }
	    
	    sreader.close();
	} catch (IOException e) {
	    System.err.println("Bizarre error situation!");
	    throw new BadOptException("I/O problem encountered manipulating strings with a StringReader!");
	}

	//	for (int i = 0; i < optlist.size(); i++)
	//	    System.out.println(optlist.elementAt(i));

	for (int i = 0; i < sargs.length; i++) {
	    if (sargs[i].startsWith("-")) {
		try {
		    StringReader sreader = new StringReader(sargs[i].substring(1));

		    int opt = -1;
		    while ((opt = sreader.read()) != -1) {
			boolean found = false;
			boolean bool = true;
			for (int j = 0; j < optlist.size(); j++) {
			    OptPair temp = (OptPair) optlist.elementAt(j);
			    if (temp.option == (char) opt) {
				found = true;
				bool = temp.bool;
				break;
			    }
			}

			if (found) {
			    opts.add(new Character((char) opt));
			    if (!bool) {
				args.add(sargs[++i]);
				break;
			    }
			} else {
			    throw new BadOptException((char) opt + ": is an invalid switch option.");
			}
		    }
		} catch (IOException e) {
		    System.err.println("Bizarre error situation!");
		    throw new BadOptException("I/O problem encountered manipulating strings with a StringReader!");
		}
	    } else {
		params.add(sargs[i]);
	    }
	}
    }

    /**
     * Returns the next command line option. In the event that no more options
     * are left the OPT_EOF value is returned.
     *
     * @return the character representing the next command line option found.
     */
    public char getopt() {
	if (opt_count >= opts.size())
	    return OPT_EOF;
	return ((Character) (opts.elementAt(opt_count++))).charValue();
    }

    /**
     * Returns the next non-boolean command line switch argument.
     *
     * @return a String argument corresponding to a command line switch.
     */
    public String optarg() {
	if (arg_count >= args.size())
	    return null;
	return (String) args.elementAt(arg_count++);
    }

    /**
     * Returns the next parameter passed on the command line.
     *
     * @return a parameter not associated with any command line switches.
     */
    public String param() {
	if (param_count >= params.size())
	    return null;
	return (String) params.elementAt(param_count++);
    }
}

/* Simple class for representing a command line option.
   Options can be boolean or require a value. */
class OptPair {
    protected char option;
    protected boolean bool = true;

    OptPair(char option, boolean bool) {
	this.option = option;
	this.bool = bool;
    }

    public String toString() {
	return new String("[ " + option + ", " + bool + " ]");
    }
}
