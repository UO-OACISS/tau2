package TauIL.absyn;

public class DecList extends SyntaxList implements Declaration {
	public DecList(Declaration head) {
		super(head);
	}

	public DecList(Declaration head, DecList tail) {
		super(head, tail);
	}
}
