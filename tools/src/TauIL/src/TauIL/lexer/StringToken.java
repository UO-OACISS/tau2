/************************************************************
 *
 *           File : StringToken.java
 *         Author : Tyrel Datwyler
 *
 *    Description : Lexical token for representing a string.
 *
 ************************************************************/

package TauIL.lexer;

public class StringToken extends Token {

    /** String value represented by this token. */
    public String string = "";

    /**
     * Constructs a token that represents a string value with the given line
     * and column numbers.
     *
     * @param line line position of token.
     * @param left left column position of token.
     * @param right right column position of token.
     * @param string string value that this token represents.
     */
    public StringToken(int line, int left, int right, String string) {
	super(line, left, right);
	this.string = string;
    }
}
