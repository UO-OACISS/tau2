package TauIL.absyn;

public class EntityList extends SyntaxList implements SyntaxAttribute {
	public EntityList(Entity head) {
		super(head);
	}

	public EntityList(Entity head, EntityList tail) {
		super(head, tail);
	}
}
