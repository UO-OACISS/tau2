/***********************************************************************
 *
 * File        : DirectiveList.java
 * Author      : Tyrel Datwyler
 *
 * Description : Class to represent a directive list.
 *
 ***********************************************************************/

package TauIL.absyn;

public class DirectiveList extends SyntaxList implements SyntaxElement {
	public DirectiveList(Directive head) {
		super(head);
	}

	public DirectiveList(Directive head, DirectiveList tail) {
		super(head, tail);
	}
}
