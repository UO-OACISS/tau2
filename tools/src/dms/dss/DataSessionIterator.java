//***********************
//
//Name: DataSessionIterator.java
//
//Description: Implements the standard list interface.
//***********************

package dms.dss;

import java.util.*;
import dms.dss.*;

public class DataSessionIterator implements ListIterator{

		public DataSessionIterator(Vector inVector){
				
				//Check to make sure that the Vector is not null.
				if(inVector == null)
						throw new IllegalArgumentException();
				
				//Safe to continue.				
				listData = inVector;
				size = listData.size();
		}

		public add(Object o){
				throw new UnsupportedOperationException();
		}

		public boolean hasNext(){
				if(!(index >= size))
						return true;
				else
						return false;
		}

		public boolean hasPrevious(){
		}

		public Object next(){
				//Check to make sure we are not over the end.
				if(index >= size)
						throw new NoSuchElementException();

				//Get the object.
				Object object = listData.elementAt(position);
				
				//Increment the index.
				index++;

				return object;
		}

		public int nextIndex(){
				return index++;
		}

		public Object previous(){
		}

		public int previousIndex(){
				return index--;
		}

		public void remove(){
				throw new UnsupportedOperationException();
		}

		public void set(Object o){
				throw new UnsupportedOperationException();
		}

		private Vector listData = null;
		pprivate int size = 0;
		private int index = 0;
}
