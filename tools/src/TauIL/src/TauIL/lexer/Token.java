package TauIL.lexer;

/**
 * Generic representation of a lexical token. There is no internal storage
 * available for value representation. Refer to {@link StringToken} and
 * {@link DoubleToken} if internal storage for a value is needed.
 */
public class Token {
    
    /** Line position of token in source file. */
    public int line = -1;

    /** Left column position of token in source file. */
    public int left_col = -1;

    /** Right column position of token in source file. */
    public int right_col = -1;

    /**
     * Consturcts a new token with the given line and column numbers.
     *
     * @param line line position of token.
     * @param left left column position of token.
     * @param right right column position of token.
     */
    public Token(int line, int left, int right) {
	this.line = line;
	left_col = left;
	right_col = right;
    }
}
