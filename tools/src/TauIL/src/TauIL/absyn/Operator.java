/***********************************************************************
 *
 * File        : Operator.java
 * Author      : Tyrel Datwyler
 *
 * Description : Represents a binary operator.
 *
 ***********************************************************************/

package TauIL.absyn;

public class Operator implements SyntaxAttribute {
    public static final int EQ = 0, LT = 1, LTEQ = 2, GT = 3, GTEQ = 4, NEQ = 5;
    public static final String [] literals = { "=", "<", "<=", ">", ">=", "!=" };

    public int op;

    public Operator(int op) {
	this.op = op;
    }

    public String generateSyntax() {
	return literals[op];
    }
}
