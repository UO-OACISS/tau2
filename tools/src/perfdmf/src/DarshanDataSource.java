package edu.uoregon.tau.perfdmf;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.StringTokenizer;

import edu.uoregon.tau.common.MetaDataMap;

public class DarshanDataSource extends DataSource {

	private int linenumber = 0;
	private boolean inData = false;
	private int nprocs = 1;

	public static void main(String[] args){
		String filename = "/home/wspear/Code/darshan-2.2.4/darshan-util/darshan.log";
		File[] files = new File[1];
		files[0] = new File(filename);
		DarshanDataSource ds = new DarshanDataSource(files);
		try {
			ds.load();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public DarshanDataSource(File[] files) {
		super();
		this.files = files;
	}
	
	public DarshanDataSource(File file) {
		super();
		if(files==null){
			files=new File[1];
		}
		this.files[0] = file;
	}

	private File files[];

	public void cancelLoad() {
		return;
	}

	public int getProgress() {
		return 0;
	}

	public void load() throws FileNotFoundException, IOException {
		//Record time.
		long time = System.currentTimeMillis();

		Function function = null;
		FunctionProfile functionProfile = null;

		String inputString = null;

		for (int fIndex = 0; fIndex < files.length; fIndex++) {
			File file = files[fIndex];

			FileInputStream fileIn = new FileInputStream(file);
			InputStreamReader inReader = new InputStreamReader(fileIn);
			BufferedReader br = new BufferedReader(inReader);

			// No timers, but lots of counters. Create time anyway.
			this.addMetric("Time");

			linenumber = 0; //Already read in the first line
			while ((inputString = br.readLine()) != null) {
				// trim the line of whitespace
				inputString = inputString.trim();
				int length = inputString.length();
				if (length == 0) {
					continue;
				} else if (inputString.startsWith("#")) {
					processComment(inputString);
				}
				else if (inputString.startsWith("t")){
					processTotal(inputString);
				}
				else if (inData) {
					processCounter(inputString);
				}
				linenumber++;
			} // while lines in file
		} // for elements in vector v

		this.generateDerivedData();

		time = (System.currentTimeMillis()) - time;
		System.out.println("Done processing data!");
		System.out.println("Time to process (in milliseconds): " + time);
	}

	private void processComment(String inputString) {
		if (inputString.startsWith("#<rank>")) {
			inData = true;
		}
		return;
	}

	private void processTotal(String inputString) {

		return;
	}

	
	private void processCounter(String inputString) {
		//#<rank> <file>  <counter>   <value> <name suffix>   <mount pt>  <fs type>
		StringTokenizer tokenizer = new StringTokenizer(inputString, "\t");
		Integer nodeID;
		try{
		nodeID = Integer.parseInt(tokenizer.nextToken());
		}catch(NumberFormatException e){
			System.out.println("Darshan Data Source Could Not Process: "+inputString);
			return;
		}
		String file = tokenizer.nextToken();
		String counter = tokenizer.nextToken();
		Double value = Double.parseDouble(tokenizer.nextToken());
		tokenizer.nextToken(); // advance past the suffix
		String mountPoint = tokenizer.nextToken();
		String fsType = tokenizer.nextToken();

		// if the number of nodes is 1, then -1 is all of them.
		if (nprocs == 1 && nodeID == -1) {
			nodeID = 0;
		}

		MetaDataMap md = null;
		Node node = null;
		Context context = null;
		edu.uoregon.tau.perfdmf.Thread thread = null;

		// now that we have the tokens, let's do something intelligent with them.
		if (nodeID == -1) {
			md = this.getMetaData();
		} else {
			node = this.addNode(nodeID);
			context = node.addContext(0);
			thread = context.addThread(0);
			md = this.getThread(nodeID,0,0).getMetaData();
		}
		md.put("Darshan Filename", file);
		md.put("Darshan Mountpoint : " + file, mountPoint);
		md.put("Darshan FS Type : " + mountPoint, mountPoint);
		if (counter.endsWith("_TIMESTAMP")) {
			md.put("Darshan " + counter + " : " + file, value.toString());
		} if (counter.endsWith("_DEVICE")) {
			md.put("Darshan " + counter + " : " + file, value.toString());
		} else if (counter.endsWith("_TIME")) {
			Function function = this.addFunction(counter, 1);
			function.addGroup(addGroup("DARSHAN"));
			FunctionProfile functionProfile = new FunctionProfile(function);
			thread.addFunctionProfile(functionProfile);
			functionProfile.setInclusive(0, value);
			functionProfile.setExclusive(0, value);
			functionProfile.setNumCalls(1);
		} else {
			UserEvent event = this.addUserEvent(counter);
			UserEventProfile uep = new UserEventProfile(event);
			thread.addUserEventProfile(uep);
			uep.setNumSamples(1);
			uep.setMaxValue(value);
			uep.setMinValue(value);
			uep.setMeanValue(value);
			uep.setSumSquared(0);
			uep.updateMax();
		}
		return;
	}
}