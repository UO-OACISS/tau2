package TauIL;

import TauIL.util.GetOpt;
import TauIL.util.BadOptException;
import TauIL.lexer.Lexer;
import TauIL.absyn.AbstractSyntaxTree;
import TauIL.parser.Parser;
import TauIL.interpreter.Interpreter;
import TauIL.util.InstList;
import TauIL.util.InstListWriter;

import java.io.InputStream;
import java.io.FileInputStream;

import java.util.Vector;
import java.util.ListIterator;
import java.util.HashMap;

public class Tool {
    public static final char [] version = { '0', '8', 'a' };

    private static String fname = null;
    private static boolean debug = false;
    private static boolean verbose = false;

    private static HashMap macros = new HashMap();

    public static boolean parseCommandLine(String [] args) {
	String define;

	try {
	    GetOpt opts = new GetOpt(args, "dhiuvD:");

	    char ch;
	    while ((ch = opts.getopt()) != GetOpt.OPT_EOF) {
		//		System.out.println(ch);
		switch(ch) {
		case 'i' :
		    verbose = true;
		    break;
		case 'D' :
		    define = opts.optarg();
		    int index = define.indexOf("=");
		    String key = define.substring(0, index);
		    String value = define.substring(index + 1);
		    macros.put(key, Lexer.makeMacroValue(value));
		    break;
		case 'd' :
		    debug = true;
		    break;
		case 'v' :
		    version();
		    System.exit(0);
		case 'h' :
		case 'u':
		    return false;
		}
	    }

	    if ((fname = opts.param()) == null)
		if ((fname = opts.optarg()) == null)
		    return false;
	} catch (BadOptException e) {
	    return false;
	}

	return true;
    }

    public static void usage() {
	System.err.println("Usage: Tool [-hiuv] [-D macro[=value]] inst-spec");
	System.err.println("Options:");
	System.err.println(" -h               : display usage dialogue");
	System.err.println(" -i               : prints step by step processing information");
	System.err.println(" -u               : same function as -h option");
	System.err.println(" -v               : display version information");
	System.err.println(" -D macro[=value] : defines a macro with an optional value");
	System.err.println(" inst-spec        : file name of insturmentation specification");
	System.err.println();
	version();
	System.exit(0);
    }

    public static void version() {
	System.err.println("Tool v" + version[0] + "." + version[1] + version[2]);
	TauIL.version();
	System.err.println();
	System.err.println("Written by Tyrel Datwyler, 2003");
    }

    public static void main(String [] args) {
	if (!parseCommandLine(args))
	    usage();

	try {
	    InputStream in = new FileInputStream(fname);
	    Lexer lexer = new Lexer(in);
	    lexer.defineMacros(macros);
	    lexer.setDebugMode(debug);

	    Parser parser = new Parser(lexer);

	    AbstractSyntaxTree ast;

	    if (debug)
	     	ast = (AbstractSyntaxTree) parser.debug_parse().value; 
	    else
		ast = (AbstractSyntaxTree) parser.parse().value;

	    Interpreter interp = new Interpreter(ast);
	    interp.setVerboseMode(verbose);
	    interp.setDebugMode(debug);

	    interp.interpret();

	    InstListWriter lwriter = new InstListWriter();

	    Vector lists = interp.getInstLists();
	    ListIterator iterator = lists.listIterator();

	    while (iterator.hasNext()) {
		lwriter.writeList((InstList) iterator.next());
	    }

	} catch (Error e) {
	    // If an error arises die with a simple error.
	    System.err.println("Unhandled error has been thrown.");
	    e.printStackTrace();
	    System.exit(1);
	} catch (Exception e) {
	    // If an exception arises die with a simple error.
	    System.err.println("Unhandled exception has been thrown.");
	    e.printStackTrace();
	    System.exit(1);
	}
	
	System.exit(0);	    
    }
}
