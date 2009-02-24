package edu.uoregon.tau.perfexplorer.persistence;

import org.hibernate.*;
import org.hibernate.cfg.*;

/**
 * @author khuck
 *
 */
public class HibernateUtil {
	
	private static SessionFactory sessionFactory;
	
	static {
		try {
			sessionFactory = new AnnotationConfiguration().configure().buildSessionFactory();
		} catch (Throwable ex) {
			throw new ExceptionInInitializerError(ex);
		}
	}

	public static SessionFactory getSessionFactory() {
		return sessionFactory;
	}
	
	public static void shutdown() {
		getSessionFactory().close();
	}
}
