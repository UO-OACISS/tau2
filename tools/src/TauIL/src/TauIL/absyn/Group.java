/***********************************************************************
 *
 * File        : Group.java
 * Author      : Tyrel Datwyler
 *
 * Description : Represents a TAU group.
 *
 ***********************************************************************/

package TauIL.absyn;

public class Group implements SyntaxAttribute {
    public String name;

    public Group(String name) {
	this.name = name;
    }

    public String generateSyntax() {
	return name;
    }
}
