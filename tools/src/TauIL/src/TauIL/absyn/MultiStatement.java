/***********************************************************************
 *
 * File        : MultiStatement.java
 * Author      : Tyrel Datwyler
 *
 * Description : Not to be confused with a list of statements. This
 *               class represents a concatenation of statements that
 *               form a single analysis.
 *
 ***********************************************************************/

package TauIL.absyn;

public class MultiStatement extends SyntaxList implements Statement {
	public Group group;

	public MultiStatement(Statement head) {
		this(head, null, NO_GROUP);
	}

	public MultiStatement(Statement head, MultiStatement tail) {
		this(head, tail, NO_GROUP);
	}

	public MultiStatement(Statement head, MultiStatement tail, Group group) {
		super(head, tail);
		this.group = group;
	}

	public void setGroup(Group group) {
		this.group = group;
	}

	public String generateSyntax() {
		String syntax = "";

		if (group != NO_GROUP)
			syntax = group.generateSyntax() + ": ";

		syntax = syntax + head.generateSyntax(); 
		if (tail != null)
			syntax = syntax + " & " + tail.generateSyntax();

		return syntax;
	}
}
