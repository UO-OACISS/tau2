/***********************************************************************
 *
 * File        : Decleration.java
 * Author      : Tyrel Datwyler
 *
 * Description : Decleration interface. Javas lack of multiple
 *               inheritance requires this approach so we can preform
 *               runtime type checking while walking the syntax tree.
 *               Java 1.5 introduces generic types and thus a cleaner
 *               approach that can be used to replace this.
 *
 ***********************************************************************/

package TauIL.absyn;

public interface Declaration extends SyntaxElement {

}
