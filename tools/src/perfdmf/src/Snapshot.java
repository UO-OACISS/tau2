package edu.uoregon.tau.perfdmf;

/**
 * Snapshot object representing a Snapshot
 *
 * <P>CVS $Id: Snapshot.java,v 1.1 2006/12/28 03:05:59 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.1 $
 */
public class Snapshot {

    private String name;
    private int id;

    public Snapshot(String name, int id) {
        this.name = name;
        this.id = id;
    }

    public String toString() {
        return name;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

}
