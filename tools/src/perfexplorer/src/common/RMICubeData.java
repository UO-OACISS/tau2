package common;

import java.io.Serializable;
import java.util.List;
import java.util.ArrayList;

/**
 * This object contains the data necessary to generate an OpenGL 3D cube of
 * data.  This data is usually correlation data.  It can reasonably have up 
 * to 4 dimensions (x,y,z,color).
 *
 * <P>CVS $Id: RMICubeData.java,v 1.2 2005/09/27 19:50:17 khuck Exp $</P>
 * @author khuck
 * @version 0.1
 * @since   0.1
 *
 */
public class RMICubeData implements Serializable {
	int dimensions = 0;

	String[] names = null;
	List values = null;

	/**
	 * Constructor.
	 * 
	 * @param dimensions
	 */
	public RMICubeData (int dimensions) {
		this.names = new String[dimensions];
		this.values = new ArrayList();
	}

	/**
	 * Set the name for this dimension.
	 * 
	 * @param dimension
	 * @param name
	 */
	public void setName (int dimension, String name) {
		this.names[dimension] = name;
	}

	/**
	 * Set the names.
	 * 
	 * @param names
	 */
	public void setNames (String[] names) {
		this.names = names;
	}

	/**
	 * Add a set of values for one point.
	 * 
	 * @param values
	 */
	public void addValues(float[] values) {
		this.values.add(values);
	}

	/**
	 * Get the set of values for one point.
	 * 
	 * @param index
	 * @return
	 */
	public float[] getValues(int index) { 
		return (float[])values.get(index); 
	}

	/**
	 * Get all points.
	 * 
	 * @return
	 */
	public float[][] getValues() {
		float v[][] = new float[values.size()][this.dimensions];
		for (int i = 0 ; i < values.size() ; i++) {
			v[i] = (float[])values.get(i);
		}
		return v;
	}

	/**
	 * Get the names for the dimensions.
	 * 
	 * @param dimension
	 * @return
	 */
	public String getNames(int dimension) { 
		return names[dimension]; 
	}

	public String[] getNames() { 
		return names; 
	}

}
