/**
 * 
 */
package edu.uoregon.tau.common;

import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.TreeMap;

/**
 * @author khuck
 *
 */


public class MetaDataMap {
	public class MetaDataKey implements Comparable {
		/* (non-Javadoc)
		 * @see java.lang.Object#hashCode()
		 */
		@Override
		public int hashCode() {
			final int prime = 31;
			int result = 1;
//			result = prime * result + getOuterType().hashCode();
			result = prime * result + ((name == null) ? 0 : name.hashCode());
			if (timer_context != null) {
				result = prime * result + call_number;
				result = prime * result + ((timer_context == null) ? 0 : timer_context.hashCode());
				result = prime * result + (int) (timestamp ^ (timestamp >>> 32));
			}
			return result;
		}

		/* (non-Javadoc)
		 * @see java.lang.Object#equals(java.lang.Object)
		 */
		@Override
		public boolean equals(Object obj) {
			if (this == obj) {
				return true;
			}
			if (obj == null) {
				return false;
			}
			if (!(obj instanceof MetaDataKey)) {
				return false;
			}
			MetaDataKey other = (MetaDataKey) obj;
			if (call_number != other.call_number) {
				return false;
			}
			if (name == null) {
				if (other.name != null) {
					return false;
				}
			} else if (!name.equals(other.name)) {
				return false;
			}
			if (timer_context == null) {
				if (other.timer_context != null) {
					return false;
				}
			} else if (!timer_context.equals(other.timer_context)) {
				return false;
			}
			if (timestamp != other.timestamp) {
				return false;
			}
			return true;
		}

		public String name;
		public String timer_context;
		public int call_number;
		public long timestamp;
		
		public MetaDataKey(String name) {
			this.name = name;
			this.timer_context = null;
			this.call_number = 0;
			this.timestamp = 0;
		}

		public int compareTo(Object arg0) {
			if (!(arg0 instanceof MetaDataKey)) {
				return 1;
			}
			MetaDataKey other = (MetaDataKey) arg0;
			StringBuffer lhs = new StringBuffer();
			lhs.append(this.name);
			if (this.timer_context != null) {
				lhs.append(":");
				lhs.append(this.timer_context);
				lhs.append(":");
				lhs.append(this.call_number);
				lhs.append(":");
				lhs.append(this.timestamp);
			}
			StringBuffer rhs = new StringBuffer();
			rhs.append(other.name);
			if (other.timer_context != null) {
				rhs.append(":");
				rhs.append(other.timer_context);
				rhs.append(":");
				rhs.append(other.call_number);
				rhs.append(":");
				rhs.append(other.timestamp);
			}
			return (lhs.toString().compareTo(rhs.toString()));
		}

		@Override
		public String toString() {
			
			if(timestamp==0&&timer_context==null&&call_number==0){
				return name;
			}
			
			return "name=" + name + ", timer_context="
					+ timer_context + ", call_number=" + call_number
					+ ", timestamp=" + timestamp;
		}
		
		

	}
	
	/* the object can be one of:
	 * String
	 * Number
	 * Array
	 * Boolean
	 * Null
	 * Object (another name/value pair in the map)
	 */
	public class MetaDataValue {
		@Override
		public String toString() {
			return value.toString();
		}
		/* (non-Javadoc)
		 * @see java.lang.Object#equals(java.lang.Object)
		 */
		@Override
		public boolean equals(Object obj) {
			if (this == obj) {
				return true;
			}
			if (obj == null) {
				return false;
			}
			if (!(obj instanceof MetaDataValue)) {
				return false;
			}
			MetaDataValue other = (MetaDataValue) obj;
			if (name == null) {
				if (other.name != null) {
					return false;
				}
			} else if (!name.equals(other.name)) {
				return false;
			}
			if (value == null) {
				if (other.value != null) {
					return false;
				}
			} else if (!value.equals(other.value)) {
				return false;
			}
			return true;
		}
		public String name;
		public Object value;
		public MetaDataValue(String name, Object value) {
			this.name = name;
			this.value = value;
		}
	}

	public Map<MetaDataKey, MetaDataValue> theMap;
	
	public MetaDataMap() {
		if (theMap == null) {
			theMap = new TreeMap<MetaDataKey, MetaDataValue>();
		}
	}

	public MetaDataKey newKey(String name) {
		return new MetaDataKey(name);
	}
	
	public MetaDataValue put(MetaDataKey key, MetaDataValue value) {
		return theMap.put(key, (MetaDataValue)value);
	}
	
	public MetaDataValue put(MetaDataKey key, Object value) {
		MetaDataValue tmp = new MetaDataValue(key.name, value);
		return theMap.put(key, tmp);
	}
	
	public MetaDataValue get(MetaDataKey key) {
		return theMap.get(key);
	}
	
	// backwards compatability
	public String get(String name) {
		String nullValue = null;
		MetaDataKey key = new MetaDataKey(name);
		MetaDataValue value = theMap.get(key);
		if (value != null) {
			return value.value.toString();
		} else {
		// iterate? this is a horrible hack, but it works. IF you have performance
		// problems, fix this somehow.
			for (MetaDataKey key2 : theMap.keySet()) {
				if (key2.name.equals(name)) {
					value = theMap.get(key2);
					return value.value.toString();
				}
			}
		}
		return nullValue;
	}
	
	public String get(Object obj) {
		return get((String)(obj));
	}
	
	public String put(String name, String value) {
		MetaDataKey k = new MetaDataKey(name);
		MetaDataValue v = new MetaDataValue(name, value);
		theMap.put(k, v);
		return v.value.toString();
	}
	
	public void remove(String name) {
		MetaDataKey key = new MetaDataKey(name);
		theMap.remove(key);
	}
	
	public int size() {
		return theMap.size();
	}
	
	public Set<MetaDataKey> keySet() {
		return theMap.keySet();
	}
	
	public boolean containsKey(String name) {
		MetaDataKey key = new MetaDataKey(name);
		return theMap.containsKey(key);
	}
	
	public MetaDataValue remove(MetaDataKey key) {
		return theMap.remove(key);
	}
	
	public Set<Entry<MetaDataKey, MetaDataValue>> entrySet() {
		return theMap.entrySet();
	}

	public void putAll(MetaDataMap metaData) {
		theMap.putAll(metaData.theMap);
	}

}
