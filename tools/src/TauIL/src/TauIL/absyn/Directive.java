/***********************************************************************
 *
 * File        : Directive.java
 * Author      : Tyrel Datwyler
 *
 * Description : Class to represent special directive or properties to
 *               use during data analysis. Data location and format is
 *               just one example.
 *
 ***********************************************************************/

package TauIL.absyn;

public class Directive implements SyntaxElement {
	public static final int TARGET = 0, TYPE = 1, USE = 2;

	public static final int INCLUDE = IncludeDec.INCLUDE, EXCLUDE = IncludeDec.EXCLUDE;
	public static final int TAU_PROFILE = 0, PDT = 1;
	public static final int FILE = 0, DB = 1;

	public int directive;
	public int flag;
	public String arg;

	public Directive(int directive, int flag, String argument) {
		this.directive = directive;
		this.flag = flag;
		this.arg = argument;
	}

	public Directive(int directive, int flag) {
		this(directive, flag, "");
	}

	public void setArgument(String arg) {
		this.arg = arg;
	}

	public String generateSyntax() {
		return "";
	}
}
