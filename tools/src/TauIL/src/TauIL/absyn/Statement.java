package TauIL.absyn;

public interface Statement extends SyntaxElement {
	public static final Group NO_GROUP = null;

	public void setGroup(Group group);
}
