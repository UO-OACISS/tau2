package TauIL.absyn;

public class GroupList extends SyntaxList implements SyntaxAttribute {
	public GroupList(Group head) {
		super(head);
	}

	public GroupList(Group head, GroupList tail) {
		super(head, tail);
	}
}
