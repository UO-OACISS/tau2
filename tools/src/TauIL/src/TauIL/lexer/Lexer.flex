package TauIL.lexer;

import TauIL.error.*;
import TauIL.parser.sym;
import java_cup.runtime.Symbol;
import java.io.InputStream;
import java.io.Reader;
import java.util.HashMap;

%%

/* Classname to give generated lexical scanner. */
%public
%class Lexer

/* Lexer options  */
%pack
%unicode

/* Generate lexer with CUP parser compatability. */
%cup

/* Generate main method for standalone testing. */
%cupdebug

/* Enable line and column counting. */
%line
%column

/* User supplied code */
%{
	private MessageManager errors;

	private boolean extra_comment_line = false;
        private boolean debug = false ;

	private int return_state = YYINITIAL;

        private StringBuffer string = new StringBuffer();
        private HashMap table = new HashMap();

	/**
	 * Constructs a scanner that records error messages in the
	 * given {@link ErrorMessage} object.
	 *
	 * @param in the java.io.InputStream to read input from.
	 * @param errors the object to use for recording error messages.
	 */
	public Lexer(InputStream in, MessageManager errors) {
		this(in);
		this.errors = errors;
	}

	/**
	 * Constructs a scanner that records error messages in the
	 * given {@link ErrorMessage} object.
	 *
	 * @param in the java.io.Reader to read input from.
	 * @param errors the object to use for recording error messages.
	 */
	public Lexer(Reader in, MessageManager errors) {
		this(in);
		this.errors = errors;
	}
	
	/**
	 * Turns on and off lexer debug mode.
	 * 
	 * @param debug true value enables and a false value disables debuging messages.
	 */
        public void setDebugMode(boolean debug) {
                this.debug = debug;
        }

	public void defineMacros(HashMap macros) {
		table = macros;
	}

	private Symbol yysym(int sym, Token val) {
		return new Symbol(sym, val.left_col, val.right_col, val);
	}

	private Symbol token(int sym) {
		Token value = new Token(yyline, yycolumn, yycolumn + yylength());
		return yysym(sym, value);
	}

	private Symbol token(int sym, Double num) {
		Token value = new DoubleToken(yyline, yycolumn, yycolumn + yylength(), num);
		return yysym(sym, value);
	}

	private Symbol token(int sym, String str) {
		Token value = new StringToken(yyline, yycolumn, yycolumn + yylength(), str);
		return yysym(sym, value);
	}

	public static Symbol makeMacroValue(String str) {
		Token value = new StringToken(-1, -1, 0, str);
		return new Symbol(sym.STRING, -1, -1, value);
	}	

        private void debug(String text) {
	        if (debug)
		        System.out.println(text);
        }

%}

%state STRING
%xstate COMMENT

%xstate PREPROCESS

%state ENVIRONMENT
%xstate IGNORE
%state EXPAND
%state MACROVAL

/* Regular Expression Macros */
LineTerminator	=	\r|\n|\r\n|\u2028|\u2029|\u000B|\u000C|\u0085
WhiteSpace	=	{LineTerminator} | [ \t\f]

NonDigit        =       [a-zA-Z_.]
Digit           =       [0-9]
Identifier      =       {NonDigit} ({NonDigit} | {Digit})*

StringChar      =       [^\r\n\"\\]

NLit1		= 	[0-9]+ \. [0-9]*
NLit2		= 	\. [0-9]+
NLit3		= 	[0-9]+
Exp		= 	[eE] [+-]? [0-9]+

NumLiteral	= 	({NLit1} | {NLit2} | {NLit3}) ({Exp})?

%%

        "$"			{ return_state = yystate(); debug("Expanding Macro"); yybegin(EXPAND); }

<YYINITIAL, MACROVAL> {
	"include"		{ return token(sym.INCLUDE); }
	"exclude"		{ return token(sym.EXCLUDE); }
	
	"declarations"		{ return token(sym.DECS); }
	"directives"		{ return token(sym.DIRS); }
	"conditions"		{ return token(sym.CONDS); }
	"anti-conditions"	{ return token(sym.ANTICONDS); }

	"instrument with"	{ return token(sym.INSTRUMENT); }
	"static"		{ return token(sym.STATIC); }
	"profile"		{ return token(sym.PROFILE); }
	"runtime"		{ return token(sym.RUNTIME); }
	"as"			{ return token(sym.AS); }
	"end"			{ return token(sym.END); }

	"use"			{ return token(sym.USE); }
	"target"		{ return token(sym.TARGET); }
	"type"			{ return token(sym.TYPE); }

	"db"			{ return token(sym.DB); }
	"file"			{ return token(sym.FILE); }
	"event"			{ return token(sym.EVENT); }
	"group"			{ return token(sym.GROUP); }

	"tau_profile"		{ return token(sym.TAU_PROFILE); }
	"pdt"			{ return token(sym.PDT); }

	"{:"			{ return token(sym.LCOLBRACE); }
	":}"			{ return token(sym.RCOLBRACE); }

	"{"			{ return token(sym.LBRACE); }
	"}"			{ return token(sym.RBRACE); }
	
	":"			{ return token(sym.COLON); }
	"&"			{ return token(sym.AND); }

	"<="			{ return token(sym.LTEQ); }
	">="			{ return token(sym.GTEQ); }
	"<"			{ return token(sym.LT); }
	">"			{ return token(sym.GT); }
	"!="			{ return token(sym.NEQ); }
	"="			{ return token(sym.EQ); }

	"numcalls"		{ return token(sym.NUMCALLS); }
	"numsubrs"		{ return token(sym.NUMSUBRS); }
	"percent"		{ return token(sym.PERCENT); }
	"usec"			{ return token(sym.USEC); }
	"cumusec"		{ return token(sym.CUMUSEC); }
	"count"			{ return token(sym.COUNT); }
	"totalcount"		{ return token(sym.TOTCOUNT); }
	"usecs/call"		{ return token(sym.USECS_CALL); }
	"counts/call"		{ return token(sym.COUNTS_CALL); }
	"stddev"		{ return token(sym.STDDEV); }

	\"			{ debug("Found a String"); yybegin(STRING); string.setLength(0); }
	{Identifier}		{ return token(sym.ID, yytext()); }

	"*"			{ /* return token(sym.KLEENE); */ }
	"+"			{ /* return token(sym.NON_EMPTY); */ }
	"|"			{ /* return token(sym.UNION); */ }

	{NumLiteral}		{ return token(sym.NUM, new Double(yytext())); }
}

<YYINITIAL> {
	"#"			{ yybegin(PREPROCESS); }
	"//"		     	{ yybegin(COMMENT); }
	{WhiteSpace}		{ /* Ignore all whitespace */ }
}

<MACROVAL> {
        [ \t\f]                 { }
        .                       { yypushback(yylength()); return null; }
}

<STRING> {
	{StringChar}+		{ string.append(yytext()); }
	\"			{ yybegin(YYINITIAL);
				  return token(sym.STRING, string.toString()); }
	\\.                     { errors.report(new ErrorMessage(yyline + ":Illegal escape sequence \"" + yytext() + "\"")); }
}

<COMMENT> {
	{LineTerminator}	{ if (extra_comment_line)
					extra_comment_line = false;
				  else
					yybegin(YYINITIAL); }
	"\\"			{ extra_comment_line = true; }
	.			{ /* Eat up comment text up to end of line */ }
}

<PREPROCESS> {
	"ENV_"			{ yybegin(ENVIRONMENT); }
	"define " {Identifier}	{ String key = yytext().substring(7);
	                          yybegin(MACROVAL);
				  debug("Calling next_token");
                                  Symbol value = next_token();
				  table.put(key, value);
	                          yybegin(COMMENT); 
				  debug("Processed define."); }
	"ifdef $" {Identifier}	{ String key = yytext().substring(7);
				  System.out.println("key : " + key);
	                          if (table.containsKey(key))
				    yybegin(COMMENT);
				  else
				    yybegin(IGNORE);
	                          debug("Processed ifdef."); }
	"ifndef $" {Identifier}	{ String key = yytext().substring(8);
	                          if (table.containsKey(key))
				    yybegin(IGNORE);
				  else 
					System.out.println("Starting comment");
				    yybegin(COMMENT);
	                          debug("Processed ifndef."); }
	.                       { yybegin(COMMENT); 
				  yypushback(yylength()); }
}

<ENVIRONMENT> {
	"version[" [0-9.]* "]"	{ debug("Processed version variable.");
				  debug("VERSION = " + yytext().substring(8, yylength() - 1)); 
				  yybegin(COMMENT); }
}

<IGNORE> {
	"#endif"		{ yybegin(COMMENT); 
                        	  debug("Processed endif."); }
	{WhiteSpace}		{ }	
	.			{ /* Do Nothing until endif directive found */ }
}

<EXPAND> {
	{Identifier}		{ yybegin(return_state);
				  debug("MACRO = $" + yytext());
		                  debug("MACRO TABLE = " + table.toString());
				  Symbol macro = (Symbol) table.get(yytext()); 
				  if (macro != null) 
				    return (Symbol) table.get(yytext()); }
}

