package TauIL.absyn;

public class Instrumentation implements SyntaxElement {
	public static final int PROFILE = 0, STATIC = 1, RUNTIME = 2;

        public static final String [] literals = { "profile", "static", "runtime" };

        public DirectiveList directives;
	public DecList declarations;
	public StatementList conditions;
        public StatementList anti_conditions;

	public String fname = "inst.sel";
	public int type = PROFILE;
	
	public Instrumentation(DirectiveList directives, DecList declarations, 
				StatementList conditions, StatementList anti_conditions) {
		this.directives = directives;
		this.declarations = declarations;
		this.conditions = conditions;
		this.anti_conditions = anti_conditions;
	}

	public Instrumentation(DecList declarations, StatementList conditions,
				StatementList anti_conditions) {
		this(null, declarations, conditions, anti_conditions);
	}

	public Instrumentation(DirectiveList directives, StatementList conditions,
				StatementList anti_conditions) {
		this(directives, null, conditions, anti_conditions);
	}

	public Instrumentation(DirectiveList directives, DecList declarations,
				StatementList conditions) {
		this(directives, declarations, conditions, null);
	}

	public Instrumentation(DirectiveList directives, StatementList conditions) {
		this(directives, null, conditions, null);
	}

	public Instrumentation(DecList declarations, StatementList conditions) {
		this(null, declarations, conditions, null);
	}

	public Instrumentation(StatementList conditions, StatementList anti_conditions) {
		this(null, null, conditions, anti_conditions);
	}

	public Instrumentation(StatementList conditions) {
		this(null, null, conditions, null);
	}

	public void setDataType(int type) {
		this.type = type;
	}

	public void setFileName(String fname) {
		this.fname = fname;
	}

	public String generateSyntax() {
		String syntax = "instrument with " + literals[type] + " as " + fname + "\n";
		
		if (directives != null) {
			syntax = "directives\n{:\n";
			syntax = directives.generateSyntax() + "\n:}\n";
		}
		if (declarations != null) {
			syntax = "declarations\n{:\n";
			syntax = declarations.generateSyntax() + "\n:}\n";
		}	
		
		syntax = "conditions\n{:\n";
		syntax = syntax + conditions.generateSyntax() + "\n:}\n";


		if (anti_conditions != null) {
			syntax = "anti-conditions\n{:\n";
			syntax = syntax + anti_conditions.generateSyntax() + "\n:}\n";
		}

		syntax += "end\n\n";

		return syntax;
	}
}
