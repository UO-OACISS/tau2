package TauIL;

import TauIL.absyn.AbstractSyntaxTree;
import TauIL.absyn.DirectiveList;
import TauIL.interpreter.Interpreter;
import TauIL.lexer.Lexer;
import TauIL.parser.Parser;
import TauIL.util.BadOptException;
import TauIL.util.GetOpt;
import TauIL.util.InstListWriter;

import java.io.*;
import java.util.Vector;
import java.util.ListIterator;

public class jReduce {

    public static final String DEFAULT_RULE = "numcalls > 1000000 & usecs/call < 2";

    private static boolean debug = false, print = false, default_rules = true, verbose = false;
    private static String rulefile = null, outputfile = null;
    
    /**************************************************
     *
     * parseCommandLine : Parses command line options
     *                    and returns false on erroneous
     *                    input.
     *
     **************************************************/
    public static boolean parseCommandLine(String [] args) {
	try {
	    GetOpt opts = new GetOpt(args, "dpr:o:vh");
	
	    char ch;
	    while ((ch = opts.getopt()) != GetOpt.OPT_EOF) {
		switch(ch) {
		case 'd' :
		    debug = true;
		    break;
		case 'p' :
		    print = true;
		    break;
		case 'r' :
		    rulefile = opts.optarg();
		    default_rules = false;
		    break;
		case 'o' :
		    outputfile = opts.optarg();
		    break;
		case 'v' :
		    verbose = true;
		    break;
		case 'h' :
		default :
		    return false;
		}
	    }
	} catch (BadOptException e) {
	    return false;
	}

	return true;
    }

    /**************************************************
     *
     * usage : Prints usage instructions to stderr.
     *
     **************************************************/
    public static void usage() {
	System.err.println("usage: jReduce [-hpv] [-r rulefile] [-o outputfile]");
	System.err.println(" -r rulefile   : specify filename of rule file");
	System.err.println(" -p            : print out all functions with their attributes");
	System.err.println(" -v            : verbose mode");
	System.err.println(" -o outputfile : specify filename of exclude list (defaults to stdout)");
	System.err.println(" -h            : display this usage dialogue");
	System.err.println("\nTauIL v0.2\njReduce v0.3");
	System.err.println("\nWritten by Tyrel Datwyler, 2003");
	System.err.println("Based on the tau_reduce utility written by Nick Trebon, 2002"); 
	System.exit(-1);
    }

    /**************************************************
     *
     * main : The main entry point to the jReduce tool.
     *
     **************************************************/
    public static void main(String [] args) {
	
	// Parse command line options
	if (!parseCommandLine(args))
	    usage();

	try {
	    Lexer lexer;

	    // If in verbose mode identify the rule file to be used or if one was not
	    // given identify the default rule that is being used.
	    if (verbose)
		if (default_rules)
		    System.out.println("Applying default rule : " + DEFAULT_RULE);
		else
		    System.out.println("Applying rule file : " + rulefile);
	    
	    // Initialize the lexical scanner with the proper input stream according
	    // to whether the rules are located in a file or a the default rule,
	    // which is a string, is being used.
	    if (rulefile != null) {
		InputStream in = new FileInputStream(rulefile);
		lexer = new Lexer(in);
	    } else {
		StringReader in = new StringReader(DEFAULT_RULE);
		lexer = new Lexer(in);
	    }
	   
//	    if (debug)
//		lexer.setDebugMode(true);

	    // Intialize the parser with the lexical scanner.
	    Parser parser = new Parser(lexer);

	    // Parse the rules and retrieve the resulting Abstract Syntax Tree from
	    // the parser.
	    if (verbose)
		System.out.println("Parsing rules...");

	    AbstractSyntaxTree ast;

	    if (debug)
	     	ast = (AbstractSyntaxTree) parser.debug_parse().value; 
	    else
		ast = (AbstractSyntaxTree) parser.parse().value;

	    // Initialize the interpreter with the verbose mode flag.
	    Interpreter interp = new Interpreter(ast);
	    
	    // If in debug mode regenerate the rules from the Abstract Syntax.
	    if (debug) {
		//		interp.prettyPrint(ast);
		interp.setDebugMode(true);
	    }

	    // Interpret the Abstract Syntax Tree.
	    if (verbose)
		System.out.println("Interpreting intermidiate rule representation...");
	    interp.interpret();
	    /*
	    InstListWriter lout;
	    OutputStream out;

	    // Determine if the output will be written to a file or stdout.
	    if (outputfile == null)
		out = System.out;
	    else
		out = new FileOutputStream(outputfile);

	    // Generate exclude list
	    if (verbose)
		System.out.println("Generating exclude list...");
	    lout = new InstListWriter(out);
	    lout.writeList(interp.getExcludeList(0));
	    lout.close();
	    */
	    //	    ListIterator ex_lists = interp.getExcludeLists();
	    //      while (ex_lists.hasNext()) 
	    //      lout.writeList((Vector) ex_list.next());
		
	    //	    ListIterator inc_lists = interp.getIncludeLists();
	    //      while (inc_lists.hasNext())
	    //      lout.writeList((Vector) inc_list.next());

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
