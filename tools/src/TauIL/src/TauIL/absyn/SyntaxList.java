package TauIL.absyn;

abstract public class SyntaxList implements AbstractSyntax {	
	public AbstractSyntax head;
	public SyntaxList tail;

	public SyntaxList(AbstractSyntax head) {
		this(head, null);
	}

	public SyntaxList(AbstractSyntax head, SyntaxList tail) {
		this.head = head;
		this.tail = tail;
	}

	public String generateSyntax() {
		String syntax = head.generateSyntax();
		if (tail != null) 
			syntax = syntax + "\n" + tail.generateSyntax();

		return syntax;
	}
}
