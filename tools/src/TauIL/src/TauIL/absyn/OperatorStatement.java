package TauIL.absyn;

public class OperatorStatement implements Statement {
    public Operator op;
    public Literal val;
    public Field field;
    public Group group;
	
    public OperatorStatement(Group group, Field field, Operator op, Literal val) {
	this.group = group;
	this.field = field;
	this.op = op;
	this.val = val;
    }

    public OperatorStatement(Field field, Operator op, Literal val) {
	this(NO_GROUP, field, op, val);
    }

    public void setGroup(Group group) {
	this.group = group;
    }

    public String generateSyntax() {
	String syntax;

	if (group != NO_GROUP)
		syntax = group.generateSyntax() + ": ";

	syntax = field.generateSyntax() + " " + op.generateSyntax() + " " + val.generateSyntax();

	return syntax;
    }
}
