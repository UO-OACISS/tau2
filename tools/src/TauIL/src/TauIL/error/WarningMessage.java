package TauIL.error;

public class WarningMessage extends Message {

    public WarningMessage(String message) {
	super("Warning : " + message);
    }
}
