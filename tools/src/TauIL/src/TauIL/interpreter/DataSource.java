package TauIL.interpreter;

abstract class DataSource {
    abstract protected void load();
    abstract protected boolean hasNext();
    abstract protected void next();
    abstract protected void reset();
    abstract protected String getEventName();
}
