import java.io.*;

import TauIL.absyn.AbstractSyntaxTree;
import TauIL.absyn.DirectiveList;
import TauIL.lexer.Lexer;
import TauIL.parser.Parser;
import TauIL.interpreter.Interpreter;

public class Test {
	
	public static void main(String [] args) {
		try {
			InputStream in = new FileInputStream(args[0]);
			Lexer lexer = new Lexer(in);
			Parser parser = new Parser(lexer);
			AbstractSyntaxTree ast = new AbstractSyntaxTree((DirectiveList) parser.parse().value);
			Interpreter interp = new Interpreter();
			interp.prettyPrint(ast);
		} catch (Exception e) {
			System.err.println("Exception caught.");
			e.printStackTrace();
			System.exit(1);
		}
	}
}
