/***********************************************************************
 *
 * File        : Statement.java
 * Author      : Tyrel Datwyler
 *
 * Description : Interface defining for an abstract syntax statement.
 *               See Decleration.java for a discussion of why this is
 *               an interface due to the limitations of Java.
 *
 ***********************************************************************/

package TauIL.absyn;

public interface Statement extends SyntaxElement {
	public static final Group NO_GROUP = null;

	public void setGroup(Group group);
}
