/**
 * 
 */
package edu.uoregon.tau.perfdmf.taudb;

import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.HashMap;
import java.util.Map;

/**
 * @author khuck
 *
 */
public class TAUdbThread {
	private int id = 0;
	private TAUdbTrial trial = null;
	private TAUdbSession session = null;
	private int nodeRank = 0;
	private int contextRank = 0;
	private int threadRank = 0;
	private int threadIndex = 0;

	/**
	 * 
	 */
	public TAUdbThread(TAUdbSession session, int id, TAUdbTrial trial, int nodeRank, int contextRank, int threadRank, int threadIndex) {
		this.id = id;
		this.trial = trial;
		this.session = session;
		this.nodeRank = nodeRank;
		this.contextRank = contextRank;
		this.threadRank = threadRank;
		this.threadIndex = threadIndex;
	}

	/**
	 * @return the id
	 */
	public int getId() {
		return id;
	}

	/**
	 * @param id the id to set
	 */
	public void setId(int id) {
		this.id = id;
	}

	/**
	 * @return the trial
	 */
	public TAUdbTrial getTrial() {
		return trial;
	}

	/**
	 * @param trial the trial to set
	 */
	public void setTrial(TAUdbTrial trial) {
		this.trial = trial;
	}

	/**
	 * @return the session
	 */
	public TAUdbSession getSession() {
		return session;
	}

	/**
	 * @param session the session to set
	 */
	public void setSession(TAUdbSession session) {
		this.session = session;
	}

	/**
	 * @return the nodeRank
	 */
	public int getNodeRank() {
		return nodeRank;
	}

	/**
	 * @param nodeRank the nodeRank to set
	 */
	public void setNodeRank(int nodeRank) {
		this.nodeRank = nodeRank;
	}

	/**
	 * @return the contextRank
	 */
	public int getContextRank() {
		return contextRank;
	}

	/**
	 * @param contextRank the contextRank to set
	 */
	public void setContextRank(int contextRank) {
		this.contextRank = contextRank;
	}

	/**
	 * @return the threadRank
	 */
	public int getThreadRank() {
		return threadRank;
	}

	/**
	 * @param threadRank the threadRank to set
	 */
	public void setThreadRank(int threadRank) {
		this.threadRank = threadRank;
	}

	/**
	 * @return the threadIndex
	 */
	public int getThreadIndex() {
		return threadIndex;
	}

	/**
	 * @param threadIndex the threadIndex to set
	 */
	public void setThreadIndex(int threadIndex) {
		this.threadIndex = threadIndex;
	}
	
	public String toString() {
		StringBuilder b = new StringBuilder();
		b.append(this.nodeRank);
		b.append(",");
		b.append(this.contextRank);
		b.append(",");
		b.append(this.threadRank);
		b.append(",");
		b.append(this.threadIndex);
		return b.toString();
	}

	public static Map<Integer, TAUdbThread> getThreads(TAUdbSession session, TAUdbTrial trial, boolean derived) {
		// if the trial has already loaded them, don't get them again.
		if (derived) {
			if (trial.getDerivedThreads() != null && trial.getDerivedThreads().size() > 0) {
				return trial.getDerivedThreads();
			}
		} else {
			if (trial.getThreads() != null && trial.getThreads().size() > 0) {
				return trial.getThreads();
			}
		}
		Map<Integer, TAUdbThread> threads = new HashMap<Integer, TAUdbThread>();
		String condition = "thread_index > -1";
		if (derived) {
			condition = "thread_index < 0";
		}
		String query = "select id, node_rank, context_rank, thread_rank, thread_index from thread where trial = ? and " + condition + " order by thread_index;";
		try {
			PreparedStatement statement = session.getDB().prepareStatement(query);
			statement.setInt(1, trial.getID());
			ResultSet results = statement.executeQuery();
			while(results.next()) {
				Integer id = results.getInt(1);
				int nodeRank = results.getInt(2);
				int contextRank = results.getInt(3);
				int threadRank = results.getInt(4);
				int threadIndex = results.getInt(5);
				TAUdbThread thread = new TAUdbThread (session, id, trial, nodeRank, contextRank, threadRank, threadIndex);
				threads.put(id, thread);
			}
			results.close();
			statement.close();
			if (derived) {
				trial.setDerivedThreads(threads);
			} else {
				trial.setThreads(threads);
			}
		} catch (SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return threads;
	}


	/**
	 * @param args
	 */
	public static void main(String[] args) {
		TAUdbSession session = new TAUdbSession("callpath", false);
		TAUdbTrial trial = TAUdbTrial.getTrial(session, 1, false);
		Map<Integer, TAUdbThread> threads = TAUdbThread.getThreads(session, trial, false);
		for (Integer id : threads.keySet()) {
			TAUdbThread thread = threads.get(id);
			System.out.println(thread.toString());
		}
		threads = TAUdbThread.getThreads(session, trial, true);
		for (Integer id : threads.keySet()) {
			TAUdbThread thread = threads.get(id);
			System.out.println(thread.toString());
		}
		session.close();
	}

}
