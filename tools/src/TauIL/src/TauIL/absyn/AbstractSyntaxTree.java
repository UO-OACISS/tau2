/***********************************************************************
 *
 * File        : AbstractSyntaxTree.java
 * Author      : Tyrel Datwyler
 *
 * Description : Class representing the root of an abstract syntax tree.
 *
 ***********************************************************************/

package TauIL.absyn;

import java.util.Date;

public class AbstractSyntaxTree implements AbstractSyntax {
    public DecList declarations;
    public InstrumentationList root;

    public AbstractSyntaxTree(DecList declarations, InstrumentationList tree) {
	this.declarations = declarations;
	root = tree;
    }

    public String generateSyntax() {
	String syntax = "// Instrumentation rules generated from TauIL abstract syntax. " + new Date() + "\n\n";
	if (declarations != null) {
	    syntax = syntax + "// Global Declarations\n";
	    syntax = syntax + "declarations\n{:\n";
	    syntax = syntax + declarations.generateSyntax();
	}

	syntax = syntax + "// Instrumentation Scenarios\n";
	syntax = syntax + root.generateSyntax();

	return syntax;
    }
}
