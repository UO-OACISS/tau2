/************************************************************
 *
 *           File : Parser.y
 *         Author : Tyrel Datwyler
 *
 *    Description : Grammar specification for the TauIL
 *                  parser.
 *
 ************************************************************/

package TauIL.parser;

import TauIL.lexer.*;
import TauIL.absyn.*;
import TauIL.error.*;

import java_cup.runtime.*;

/* action code {: :} */

parser code {: 
	ErrorMessage errors;

	public Parser(Scanner s, ErrorMessage errors) {
		super(s);
		this.errors = errors;
	}
:}

/* Terminals */
terminal StringToken ID, STRING;
terminal DoubleToken NUM;
terminal Token COLON, EQ, LT, GT, GTEQ, LTEQ, NEQ, AND;
terminal Token LBRACE, RBRACE, LCOLBRACE, RCOLBRACE;
terminal Token NUMCALLS, NUMSUBRS, PERCENT, USEC, CUMUSEC, COUNT, TOTCOUNT;
terminal Token STDDEV, USECS_CALL, COUNTS_CALL;
terminal Token DECS, DIRS, CONDS, ANTICONDS;
terminal Token INSTRUMENT, STATIC, PROFILE, RUNTIME, AS, END;
terminal Token EXCLUDE, INCLUDE, USE, TARGET, TYPE;
terminal Token FILE, EVENT, GROUP, DB, TAU_PROFILE, PDT;

/* Non-Terminals */
non terminal AbstractSyntaxTree inst_spec;
non terminal DecList decs;
non terminal ListManager dec_list;
non terminal Declaration dec;
non terminal Integer dec_type;
non terminal Entity entity;
non terminal Integer entity_type;
non terminal ListManager entity_list;
non terminal EntityList entities;
non terminal ListManager inst_list;
non terminal Instrumentation inst_block;
non terminal Instrumentation inst_body;
non terminal Integer data_type;
non terminal DirectiveList directives;
non terminal StatementList conditions;
non terminal StatementList anti_conditions;
non terminal ListManager dir_list;
non terminal Directive direct;
non terminal Integer dir_type, dir_use;
non terminal ListManager cond_list;
non terminal MultiStatement cond;
non terminal ListManager statement;
non terminal Group group;
non terminal OperatorStatement op_statement; 
non terminal Field field;
non terminal Operator operator;
non terminal empty;

/* Precedences
precedence left AND, COLON;
precedence nonassoc EQ, LT, LTEQ, GT, GTEQ, NEQ;
*/

/* Grammar Productions */
start with inst_spec;

inst_spec	::=	  decs:dec inst_list:inst		{: RESULT = new AbstractSyntaxTree(dec, (InstrumentationList) inst.retrieve()); :};

decs		::=	  DECS LCOLBRACE dec_list:decls RCOLBRACE	{: RESULT = (DecList) decls.retrieve(); :}
                        | empty;

dec_list	::=	  dec:dec  				{: RESULT = new ListManager(new DecList(dec), ListManager.QUEUE); :}
			| dec_list:decls dec:dec		{: decls.insert(new DecList(dec)); RESULT = decls; :};

dec		::=	  dec_type:dt entity_type:et entities:ents      {: RESULT = new IncludeDec(dt.intValue(), et.intValue(), ents); :};

dec_type	::=	  INCLUDE				{: RESULT = new Integer(IncludeDec.INCLUDE); :}
			| EXCLUDE				{: RESULT = new Integer(IncludeDec.EXCLUDE); :};

entity_type	::=	  FILE					{: RESULT = new Integer(Entity.FILE); :}
			| EVENT					{: RESULT = new Integer(Entity.EVENT); :}
			| GROUP					{: RESULT = new Integer(Entity.GROUP); :};

entities	::=	  entity:ent				{: RESULT = new EntityList(ent); :}
			| LBRACE entity_list:entls RBRACE       {: RESULT = (EntityList) entls.retrieve(); :};

entity_list	::=	  entity:ent				{: RESULT = new ListManager(new EntityList(ent), ListManager.QUEUE); :}
			| entity_list:entls entity:ent	       	{: entls.insert(new EntityList(ent)); RESULT = entls; :};

entity		::=	  ID:id					{: RESULT = new Entity(id.string); :};

inst_list	::=	  inst_block:instb			{: RESULT = new ListManager(new InstrumentationList(instb), ListManager.STACK); :}
			| inst_block:instb inst_list:instls	{: instls.insert(new InstrumentationList(instb)); RESULT = instls; :};

inst_block	::=	  INSTRUMENT data_type:dt AS STRING:str inst_body:inst END	{: inst.setDataType(dt.intValue()); 
											   inst.setFileName(str.string); 
											   RESULT = inst; :};

data_type	::=	  PROFILE				{: RESULT = new Integer(Instrumentation.PROFILE); :}
			| STATIC				{: RESULT = new Integer(Instrumentation.STATIC); :}
			| RUNTIME				{: RESULT = new Integer(Instrumentation.RUNTIME); :};

inst_body	::=	  directives:dir decs:dec conditions:co anti_conditions:aco	{: RESULT = new Instrumentation(dir, dec, co, aco); :};

directives	::=	  DIRS LCOLBRACE dir_list:dirls RCOLBRACE	{: RESULT = (DirectiveList) dirls.retrieve(); :}
                        | DIRS LCOLBRACE error RCOLBRACE                {: System.out.println("ERROR IN DIRECTIVES BLOCK"); RESULT = null; :}
                        | empty;

dir_list	::=	  direct:dir				{: RESULT = new ListManager(new DirectiveList(dir), ListManager.QUEUE); :}
			| dir_list:dirls direct:dir		{: dirls.insert(new DirectiveList(dir)); RESULT = dirls; :};

direct		::=	  TARGET dec_type:dt			{: RESULT = new Directive(Directive.TARGET, dt.intValue()); :}
			| TYPE dir_type:dt			{: RESULT = new Directive(Directive.TYPE, dt.intValue()); :}
			| USE dir_use:du STRING:str		{: Directive d = new Directive(Directive.USE, du.intValue());
								   d.setArgument(str.string); 
								   RESULT = d; :};

dir_type	::=	  TAU_PROFILE				{: RESULT = new Integer(Directive.TAU_PROFILE); :}
			| PDT					{: RESULT = new Integer(Directive.PDT); :};

dir_use		::=	  FILE					{: RESULT = new Integer(Directive.FILE); :}
			| DB					{: RESULT = new Integer(Directive.DB); :};

conditions	::=	  CONDS LCOLBRACE cond_list:cols RCOLBRACE	{: RESULT = (StatementList) cols.retrieve(); :};

cond_list	::=	  cond:co				{: RESULT = new ListManager(new StatementList(co), ListManager.QUEUE); :}
			| cond_list:cols cond:co		{: cols.insert(new StatementList(co)); RESULT = cols; :};

cond		::=	  statement:st				{: RESULT = (MultiStatement) st.retrieve(); :}
			| group:g statement:st			{: MultiStatement list = (MultiStatement) st.retrieve(); list.setGroup(g); RESULT = list; :};

group		::=	  ID:id COLON				{: RESULT = new Group(id.string); :};

statement	::=	  op_statement:opst			{: RESULT = new ListManager(new MultiStatement(opst), ListManager.QUEUE); :}
			| statement:st AND op_statement:opst	{: st.insert(new MultiStatement(opst)); RESULT = st;  :};

op_statement    ::=       field:f operator:op NUM:num		{: RESULT = new OperatorStatement(f, op, new Literal(num.value.doubleValue())); :};

operator	::=	  EQ					{: RESULT = new Operator(Operator.EQ); :}
			| LT					{: RESULT = new Operator(Operator.LT); :}
			| GT					{: RESULT = new Operator(Operator.GT); :}
			| GTEQ					{: RESULT = new Operator(Operator.GTEQ); :}
			| LTEQ					{: RESULT = new Operator(Operator.LTEQ); :}
			| NEQ					{: RESULT = new Operator(Operator.NEQ); :};

field		::=	  NUMCALLS     				{: RESULT = new Field(Field.NUMCALLS); :}
			| NUMSUBRS     				{: RESULT = new Field(Field.NUMSUBRS); :}
			| PERCENT				{: RESULT = new Field(Field.PERCENT); :}
			| USEC					{: RESULT = new Field(Field.USEC); :}
			| CUMUSEC				{: RESULT = new Field(Field.CUMUSEC); :}
			| COUNT					{: RESULT = new Field(Field.COUNT); :}
			| TOTCOUNT				{: RESULT = new Field(Field.TOTCOUNT); :}
			| STDDEV				{: RESULT = new Field(Field.STDDEV); :}
			| USECS_CALL				{: RESULT = new Field(Field.USECS_CALL); :}
			| COUNTS_CALL				{: RESULT = new Field(Field.COUNTS_CALL); :};

anti_conditions	::=	  ANTICONDS LCOLBRACE RCOLBRACE		{: RESULT = new StatementList(null); :}
                        | empty;

empty           ::=       /* empty transition */;
