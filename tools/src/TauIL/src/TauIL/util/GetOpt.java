package TauIL.util;

import java.util.Vector;
import java.util.StringTokenizer;
import java.io.StringReader;
import java.io.IOException;

public class GetOpt {
    public static final char OPT_EOF = '\0';

    private Vector optlist = new Vector();
    private Vector opts = new Vector();
    private Vector args = new Vector();
    private Vector params = new Vector();
    private int opt_count = 0;
    private int arg_count = 0;
    private int param_count = 0;
    private boolean first_opt = false;

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



    public char getopt() {
	if (opt_count >= opts.size())
	    return OPT_EOF;
	return ((Character) (opts.elementAt(opt_count++))).charValue();
    }

    public String optarg() {
	if (arg_count >= args.size())
	    return null;
	return (String) args.elementAt(arg_count++);
    }

    public String param() {
	if (param_count >= params.size())
	    return null;
	return (String) params.elementAt(param_count++);
    }

    /* --------------------- OLD BUGGY CONSTRUCTOR ------------------------------

    public GetOpt(String [] sargs, String flags) throws BadOptException {
	int size = sargs.length;
	
	StringTokenizer opt_toks = new StringTokenizer(flags, ":");

	while (opt_toks.hasMoreTokens()) {
	    String token = opt_toks.nextToken();
	    int length = token.length();
	    
	    try {
		StringReader sreader = new StringReader(token);
		for (int i = 0; i < length - 1; i++) 
		    optlist.add(new OptPair((char) sreader.read(), true));
		
		optlist.add(new OptPair((char) sreader.read(), false));
		sreader.close();
	    } catch (IOException e) {
		System.err.println("Bizarre error situation!");
		e.printStackTrace();
		throw new BadOptException("I/O problem encountered manipulating strings with a StringReader!");
	    }
	} // while

	for (int i = 0; i < sargs.length; i++) {
	    if (sargs[i].startsWith("-")) {
		first_opt = true;
		try {
		    StringReader sreader = new StringReader(sargs[i].substring(1));

		    int opt = -1;
		    while ((opt = sreader.read()) != -1) { 
			boolean found = false;
			for (int j = 0; j < optlist.size(); j++) {
			    OptPair temp = (OptPair) optlist.elementAt(j);
			    if (temp.option == (char) opt) {
				found = true;
				break;
			    }
			}
			if (found)  
			    opts.add(new Character((char) opt));
			else
			    throw new BadOptException((char) opt + ": is an invalid switch option.");
		    } // while
		    sreader.close();
		} catch (IOException e) {
		    System.err.println("Bizarre error situation!");
		    System.err.println(e.getMessage());
		    e.printStackTrace();
		    throw new BadOptException("I/O problem encountered manipulating strings!");
		}
	    } else {
		if (first_opt) {
		    char opt = ((Character) opts.elementAt(opts.size() - 1)).charValue();
		    for (int j = 0; j < optlist.size(); j++) {
			OptPair temp = (OptPair) optlist.elementAt(j);
			if (temp.option == opt) {
			    if (temp.bool)
				throw new BadOptException(opt + ": this switch takes no arguments.");
			    else
				break;
			} // if
		    } // for
		    args.add(sargs[i]);
		} else 
		    params.add(sargs[i]);
	    } // if
	} // for
    } // GetOpt Constructor

    */
}

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
