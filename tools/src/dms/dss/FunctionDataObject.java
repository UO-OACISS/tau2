package com.perfdb.api;

import com.perfdb.util.dbinterface.*;
import com.perfdb.util.io.*;
import com.perfdb.dbmanager.*;
import java.io.*;
import java.net.*;
import java.sql.*;


public class FunctionDataObject {
	private double inclusivePercentage;
	private double inclusive;
	private double exclusivePercentage;
	private double exclusive;
	private int numCalls;
	private int numSubroutines;
	private double inclusivePerCall;

	public void setInclusivePercentage (double inclusivePercentage) {
		this.inclusivePercentage = inclusivePercentage;
	}

	public void setInclusive (double inclusive) {
		this.inclusive = inclusive;
	}

	public void setExclusivePercentage (double exclusivePercentage) {
		this.exclusivePercentage = exclusivePercentage;
	}

	public void setExclusive (double exclusive) {
		this.exclusive = exclusive;
	}

	public void setNumCalls (int numCalls) {
		this.numCalls = numCalls;
	}

	public void setNumSubroutines (int numSubroutines) {
		this.numSubroutines = numSubroutines;
	}

	public void setInclusivePerCall (double inclusivePerCall) {
		this.inclusivePerCall = inclusivePerCall;
	}

	public double getInclusivePercentage () {
		return this.inclusivePercentage;
	}

	public double getInclusive () {
		return this.inclusive;
	}

	public double getExclusivePercentage () {
		return this.exclusivePercentage;
	}

	public double getExclusive () {
		return this.exclusive;
	}

	public int getNumCalls () {
		return this.numCalls;
	}

	public int getNumSubroutines () {
		return this.numSubroutines;
	}

	public double getInclusivePerCall () {
		return this.inclusivePerCall;
	}
}

