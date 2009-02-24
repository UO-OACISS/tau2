/**
 * 
 */
package edu.uoregon.tau.perfexplorer.persistence;

//import com.db4o.ObjectContainer;
//import com.db4o.Db4o;
//import com.db4o.ObjectSet;


/**
 * @author khuck
 *
 */
public class DB4OUtilities {

	public final static String DB4OFILENAME="perfexplorer.db4o";
/*
	
	public static void saveObject(Object object) {
		//	 accessDb4o
		ObjectContainer db=Db4o.openFile(DB4OFILENAME);
		try {
		    // do something with db4o
			db.set(object);
		}
		finally {
		    db.close();
		}
	}

	public static void listAll(Object type) {
		//	 accessDb4o
		ObjectContainer db=Db4o.openFile(DB4OFILENAME);
		try {
		    // do something with db4o
			retrieveAll(db, type);
		}
		finally {
		    db.close();
		}
	}

	public static void listResult(ObjectSet result) {
	    System.out.println(result.size());
	    while(result.hasNext()) {
	        System.out.println(result.next());
	    }
	}
	
	public static void listResult(java.util.List result){
		System.out.println(result.size());
		for(int x = 0; x < result.size(); x++)
			System.out.println(result.get(x));
	}
	
	public static void listRefreshedResult(ObjectContainer container,ObjectSet result,int depth) {
	    System.out.println(result.size());
	    while(result.hasNext()) {
	        Object obj = result.next();
	        container.ext().refresh(obj, depth);
	        System.out.println(obj);
	    }
	}
	
	public static void retrieveAll(ObjectContainer db){
	    ObjectSet result=db.get(new Object());
	    listResult(result);
	}
	
	public static void retrieveAll(ObjectContainer db, Object object){
	    ObjectSet result=db.get(object);
	    listResult(result);
	}

	public static void deleteAll(ObjectContainer db) {
	    ObjectSet result=db.get(new Object());
	    while(result.hasNext()) {
	        db.delete(result.next());
	    }
	}
*/
}
