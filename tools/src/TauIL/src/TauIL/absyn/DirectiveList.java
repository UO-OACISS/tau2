package TauIL.absyn;

public class DirectiveList extends SyntaxList implements SyntaxElement {
	public DirectiveList(Directive head) {
		super(head);
	}

	public DirectiveList(Directive head, DirectiveList tail) {
		super(head, tail);
	}
}
