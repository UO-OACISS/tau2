package TauIL.absyn;

public class StatementList extends SyntaxList implements Statement {
	public StatementList(Statement head) {
		super(head);
	}

	public StatementList(Statement head, StatementList tail) {
		super(head, tail);
	}

	public void setGroup(Group group) {
		((StatementList) tail).setGroup(group);
	}
}
