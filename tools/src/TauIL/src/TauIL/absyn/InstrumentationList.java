/***********************************************************************
 *
 * File        : InstrumentationList.java
 * Author      : Tyrel Datwyler
 *
 * Description : Representation of a list of instrumentation scenarios.
 *
 ***********************************************************************/

package TauIL.absyn;

public class InstrumentationList extends SyntaxList implements SyntaxElement {
	public InstrumentationList(Instrumentation head) {
		super(head);
	}

	public InstrumentationList(Instrumentation head, InstrumentationList tail) {
		super(head, tail);
	}
}
