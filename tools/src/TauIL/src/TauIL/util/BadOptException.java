package TauIL.util;

public class BadOptException extends Exception {
    private String errmsg;

    public BadOptException(String error) {
	super(error);
    }

    public String getMessage() {
	return errmsg;
    }
}
