/* 
	SystemEvents.java

	Title:			jRacy
	Author:			Robert Bell
	Description:	
*/

package jRacy;

import java.util.*;
import java.lang.*;

public class SystemEvents extends Observable
{
	public void updateRegisteredObjects(String inString)
	{
		//Set this object as changed.
		this.setChanged();
		
		//Now notify observers.
		this.notifyObservers(inString);
	}
	
}
