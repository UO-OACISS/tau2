package TauIL.lexer;

/**
 * Generic lexical token with internal storage for a double value.
 */
public class DoubleToken extends Token {

    /** Double value represented by this token. */
    public Double value = new Double(0.0);

    /**
     * Constructs a token that represents a double value with the given line
     * and column numbers.
     *
     * @param value double value that this token represents.
     * @param line line position of token.
     * @param left left column position of token.
     * @param right right column position of token.
     */
    public DoubleToken(int line, int left, int right, Double value) {
	super(line, left, right);
	this.value = value;
    }
}
