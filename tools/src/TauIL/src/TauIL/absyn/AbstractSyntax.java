/***********************************************************************
 *
 * File        : AbstractSyntax.java
 * Author      : Tyrel Datwyler
 *
 * Description : Abstract syntax interface. Requires a method for 
 *               generating the respective concrete syntax.
 *
 ***********************************************************************/

package TauIL.absyn;

public interface AbstractSyntax {
	public String generateSyntax();
}
