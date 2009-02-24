/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.ObjectOutput;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Hashtable;
import java.util.List;
import java.util.Map;
import java.util.Set;

import weka.core.*;
import weka.classifiers.Classifier;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToNominal;
import java.io.*;

import edu.uoregon.tau.perfexplorer.cqos.WekaClassifierWrapper;

/**
 * @author khuck
 *
 */
public class CQoSClassifierOperation extends AbstractPerformanceOperation {

	public static final String SUPPORT_VECTOR_MACHINE = WekaClassifierWrapper.SUPPORT_VECTOR_MACHINE;
	//public static final String SUPPORT_VECTOR_MACHINE2 = WekaClassifierWrapper.SUPPORT_VECTOR_MACHINE2;
	public static final String NAIVE_BAYES = WekaClassifierWrapper.NAIVE_BAYES;
    public static final String MULTILAYER_PERCEPTRON = WekaClassifierWrapper.MULTILAYER_PERCEPTRON;
    public static final String LINEAR_REGRESSION = WekaClassifierWrapper.LINEAR_REGRESSION;
    public static final String J48 = WekaClassifierWrapper.J48;
    //public static final String AODE = WekaClassifierWrapper.AODE;
    public static final String ALTERNATING_DECISION_TREE = WekaClassifierWrapper.ALTERNATING_DECISION_TREE;
    public static final String RANDOM_TREE = WekaClassifierWrapper.RANDOM_TREE;

	private String metric = "Time";
	private Set<String> metadataFields = null;
	private String classLabel = null;
	private WekaClassifierWrapper wrapper = null;
	private String classifierType = MULTILAYER_PERCEPTRON;
	private int trainingSize = 0;
	protected List<Map<String,String>> trainingData = null;
	private Map<String,Set<String>> validation = new HashMap<String,Set<String>>();
		
	/**
	 * @param inputs
	 */
	public CQoSClassifierOperation(List<PerformanceResult> inputs, String metric, 
			Set<String> metadataFields, String classLabel) {
		super(inputs);
		this.metric = metric;
		this.metadataFields = metadataFields;
		this.classLabel = classLabel;
		getUniqueTuples();
	}

	private CQoSClassifierOperation(String fileName) {
		super();
		this.wrapper = WekaClassifierWrapper.readClassifier(fileName);
	}

 	private void getUniqueTuples() {
		try {
		// create a map to store the UNIQUE tuples
		Map<Hashtable<String,String>,PerformanceResult> tuples = 
			new HashMap<Hashtable<String,String>,PerformanceResult>();
		System.out.println("Finding unique tuples...");
		// iterate through the inputs
		int index = 1;
		int discarded = 0;
		for (PerformanceResult input : this.inputs) {
			boolean abort1 = false;
			boolean abort2 = false;
			System.out.print("\rProcessing " + index + " of " + this.inputs.size() + " (" + index*100/this.inputs.size() + "% done, " + discarded + " discarded)");
			index += 1;
			// create a local Hashtable 
			Hashtable<String,String> localMeta = new Hashtable<String,String>();
			// get the input's metadata
			TrialMetadata tmp = new TrialMetadata(input);
			Hashtable<String,String> meta = tmp.getCommonAttributes();
			// create a reduced hashtable
			if (this.metadataFields != null) {
				for (String key : this.metadataFields) {
					// don't include the class label - we want just the "best" method for that parameter - not all of them
					if (!key.equals(this.classLabel)) {
						String tmpStr = meta.get(key);
						if (tmpStr == null) {
							//System.out.println("NO VALUE FOUND FOR KEY: "+ key);
							abort1 = true;
						} else {
							String value = meta.get(key);
							//if ((key.equals("success") || key.endsWith("_success")) && value.trim().equals("0")) {
							if (key.endsWith("_success") && value.trim().equals("0")) {
								//System.out.println("Aborting (success=0) Trial: " + input.getTrial().getName());
								abort2 = true;
							} else {
								localMeta.put(key, meta.get(key));
							}
						}
					}
				}
			// otherwise, if the user didn't specify a set of properties, use them all (?)
			} else {
				for (String key : meta.keySet()) {
					// don't include the class label - we want just the "best" method for that parameter - not all of them
					if (!key.equals(this.classLabel))
						localMeta.put(key, meta.get(key));
				}				
			}
			
			// if this iteration was not successful, then don't save it's values.
			if (abort1 || abort2) {
				discarded++;
				//if(abort1) {
					//System.out.println("\n" + input.getTrial().getName());
				//}
				continue;
			}
			
			// check if the metric is one of the metrics, or a metadata field
			if (!input.getMetrics().contains(metric)) {
				// put the hashtable in the set: if its performance is "better" than the existing one
				// or if it doesn't exist yet.
				PerformanceResult result = tuples.get(localMeta);
				String tmpSuccess = localMeta.get("success");
				if (tmpSuccess == null || tmpSuccess.equals("1")) { // don't save this value if it didn't converge!
					if (result == null) {
						tuples.put(localMeta,input);
					} else {
						if (Double.parseDouble(meta.get(metric)) < 
								Double.parseDouble((new TrialMetadata(result)).getCommonAttributes().get(metric))) {
							tuples.put(localMeta,input);
						}
					}
				}
			} else {
				// put the hashtable in the set: if its performance is "better" than the existing one
				// or if it doesn't exist yet.
				PerformanceResult result = tuples.get(localMeta);
				if (result == null) {
					tuples.put(localMeta,input);
				} else {
					if (input.getInclusive(0, input.getMainEvent(), metric) < 
							result.getInclusive(0, result.getMainEvent(), this.metric)) {
						tuples.put(localMeta,input);					
					}
				}
			}
		}
		
		System.out.println("Done.");
		this.trainingData = new ArrayList<Map<String,String>>();

		FileWriter fstream = new FileWriter("/tmp/data.csv");
		BufferedWriter out = new BufferedWriter(fstream);
		String output = new String();
		for (String metaKey : this.metadataFields) {
			output = output + metaKey + ", ";
		}
		output = output + "time";
		out.write(output);
		out.newLine();

		System.out.println("Processing unique tuples...");
		// ok, we have the set of "optimal" methods for each unique tuple.  Convert them to Instances.
		index = 1;
		for (Hashtable<String,String> tmp : tuples.keySet()) {
			System.out.print("\rProcessing " + index + " of " + tuples.keySet().size() + " (" + index*100/tuples.keySet().size() + "% done)");
			index += 1;
			Map<String,String> tmpMap = new HashMap<String,String>();
			
			output = new String();
			// set the independent parameters
			if (this.metadataFields != null) {
				for (String metaKey : this.metadataFields) {
					tmpMap.put(metaKey, tmp.get(metaKey));
					Set valid = validation.get(metaKey);
					if (valid == null) {
						valid = new HashSet<String>();
						validation.put(metaKey, valid);
					}
					valid.add(tmp.get(metaKey));
					if (metaKey.equals(this.classLabel)) {
						output = output + (new TrialMetadata(tuples.get(tmp)).getCommonAttributes().get(classLabel)) + ", ";
					} else {
						output = output + tmp.get(metaKey) + ", ";
					}
				}
			} else {
				for (String metaKey : tmp.keySet()) {
					tmpMap.put(metaKey, tmp.get(metaKey));
				}
			}
			PerformanceResult result = tuples.get(tmp);
			output = output + result.getInclusive(0, result.getMainEvent(), this.metric);
			out.write(output);
			out.newLine();
			tmpMap.put(this.classLabel, (new TrialMetadata(tuples.get(tmp)).getCommonAttributes().get(classLabel)));
			trainingData.add(tmpMap);
		}
		System.out.println("Done.");

		out.close();
		
		for (String first : validation.keySet()) {
			System.out.print(first + ": ");
			for (String second : validation.get(first)) {
				System.out.print(second + ", ");
			}
			System.out.println("");
		}
		} catch (Exception e) {
			System.err.println(e.getMessage());
			e.printStackTrace();			
		}
	}
		

 	public List<PerformanceResult> processData() {
		trainingSize  = this.trainingData.size();
		System.out.println("Total instances for training: " + this.trainingSize);
		System.out.println("Using keys: " + this.metadataFields.toString());

		try {
			this.wrapper = new WekaClassifierWrapper (trainingData, this.classLabel);
			this.wrapper.setClassifierType(classifierType);
			this.wrapper.buildClassifier();
		} catch (Exception e) {
			System.err.println(e.getMessage());
			e.printStackTrace();			
		}
	
		return null;
	}
	
	public String getClass(Map<String,String> inputFields) {
		return wrapper.getClass(inputFields);
	}

	public double getConfidence() {
		return wrapper.getConfidence();
	}

	public void writeClassifier(String fileName) {
		WekaClassifierWrapper.writeClassifier(fileName, this.wrapper);
	}

	public static CQoSClassifierOperation readClassifier(String fileName) {
		return new CQoSClassifierOperation(fileName);
	}

	public String getClassifierType() {
		return this.classifierType;
	}

	public void setClassifierType(String classifierType) {
		this.classifierType = classifierType;			
	}

	public String crossValidateModel() {
		//return this.wrapper.crossValidateModel(Math.max(trainingSize/100, 10));
		return this.wrapper.crossValidateModel(3);
	}

}
