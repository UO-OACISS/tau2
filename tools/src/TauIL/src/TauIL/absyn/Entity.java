package TauIL.absyn;

public class Entity implements SyntaxAttribute {
    public static final int FILE = 0, EVENT = 1, GROUP = 2;
    public static final int MAX_ENTITY_VALUE = GROUP;

    public String name;

    public Entity(String name) {
	this.name = name;
    }

    public String generateSyntax() {
	return name;
    }
}
