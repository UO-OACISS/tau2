package TauIL.absyn;

public class IncludeDec implements Declaration {
    public static final int INCLUDE = 0, EXCLUDE = 1;
    public static final String [] literals = { "include", "exclude" };

    public int include_flag;
    public int entity_flag;
    public EntityList list;

    public IncludeDec(int in_or_ex_flag, int entity_type, EntityList event_list) {
	include_flag = in_or_ex_flag;
	entity_flag = entity_type;
	list = event_list;
    }

    public String generateSyntax() {
	String syntax = literals[include_flag] + " {\n";
	syntax = syntax + list.generateSyntax();
	syntax = syntax + "\n}";

	return syntax;
    }
}
