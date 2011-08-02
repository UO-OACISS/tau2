package edu.uoregon.tau.paraprof;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import com.graphbuilder.math.Expression;
import com.graphbuilder.math.ExpressionTree;
import com.graphbuilder.math.FuncMap;
import com.graphbuilder.math.VarMap;

public class ThreeDeeGeneralPlotUtils {
	
	
    static final String BEGIN ="BEGIN_VIZ";
    static final String END ="END_VIZ";
    
	
 public static VarMap getEvaluation(int rank,  int maxRank, int node, int context, int thread, int maxNode, int maxContext, int maxThread, float[] topoVals, float[] varMins, float varMaxs[], float varMeans[], float[] atomValue,int[] axisDim, Map<String,String> expressions){//String[] expressions, int rank,  int maxRank){
    	//System.out.println(rank);
    	FuncMap fm = new FuncMap();
		fm.loadDefaultFunctions();
    	VarMap vm = new VarMap(false);
    	vm.setValue("maxRank", maxRank);
    	vm.setValue("rank", rank);
    	vm.setValue("color", topoVals[3]);
    	vm.setValue("node", node);
    	vm.setValue("context", context);
    	vm.setValue("thread", thread);
    	vm.setValue("event0.val", topoVals[0]);
    	vm.setValue("event1.val", topoVals[1]);
    	vm.setValue("event2.val", topoVals[2]);
    	vm.setValue("event3.val", topoVals[3]);
    	vm.setValue("event0.min", varMins[0]);
    	vm.setValue("event1.min", varMins[1]);
    	vm.setValue("event2.min", varMins[2]);
    	vm.setValue("event3.min", varMins[3]);
    	vm.setValue("event0.max", varMaxs[0]);
    	vm.setValue("event1.max", varMaxs[1]);
    	vm.setValue("event2.max", varMaxs[2]);
    	vm.setValue("event3.max", varMaxs[3]);
    	vm.setValue("event0.mean", varMeans[0]);
    	vm.setValue("event1.mean", varMeans[1]);
    	vm.setValue("event2.mean", varMeans[2]);
    	vm.setValue("event3.mean", varMeans[3]);
    	vm.setValue("atomic0", atomValue[0]);
    	vm.setValue("atomic1", atomValue[1]);
    	vm.setValue("atomic2", atomValue[2]);
    	vm.setValue("atomic3", atomValue[3]);
    	vm.setValue("axisDimX",axisDim[0]);
    	vm.setValue("axisDimY",axisDim[1]);
    	vm.setValue("axisDimZ",axisDim[2]);
    	
    	Expression x;
    	double res;
    	
    	Iterator<Entry<String,String>> it = expressions.entrySet().iterator();
    	
    	while(it.hasNext()){
    		Entry<String,String> e = it.next();
    		x = ExpressionTree.parse(e.getValue());
    		res = x.eval(vm,fm);
    		//System.out.println(e.getKey()+" "+res);
    		vm.setValue(e.getKey(), res);
    	}
    	return vm;
    	
    }


 private static String[] splitEQ(String s){
 	String[] tuple = new String[2];
 	
 	int x1 = s.indexOf('=');

		tuple[0] = s.substring(0,x1).trim();
		tuple[1] = s.substring(x1+1).trim();
 	//System.out.println(s+" "+tuple[0]+" "+tuple[1]);
 	return tuple;
 }
 
 public static Map<String,String> getExpressions(String fileLoc,String expName){
 	BufferedReader br;
 	Map<String,String> expressions = new LinkedHashMap<String,String>();
		try {
			br = new BufferedReader(new FileReader(new File(fileLoc)));
		

		String s;
		
		 boolean foundExp=false;

		while ((s = br.readLine()) != null) {
			
			if(!foundExp && s.startsWith(BEGIN)){
				if(splitEQ(s)[1].equals(expName))
				{
					foundExp=true;
					continue;
				}
			}
			
			if(foundExp){
				if(s.equals(END))
					break;
				if(!s.contains("=")||s.startsWith("#"))
					continue;
				
				String[] tuple = splitEQ(s);
				
				expressions.put(tuple[0], tuple[1]);
			}
			

		}
		
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return expressions;
 }

 public static boolean checkSet(VarMap vm, String var){
 	String[] names = vm.getVariableNames();
 	for(int i=0;i<names.length;i++){
 		if(names[i].equals(var)){
 			return true;//vm.getValue(var)!=0;
 		}
 	}
 	return false;
 }
 
 public static int getPointsPerRank(VarMap vm){
 	int ppm=0;
 	boolean has=checkSet(vm,"x");
 	//System.out.println(has);
 	if(!has){
 		has=true;
 		while(has){
 			//System.out.println(ppm);
 			has = checkSet(vm,"x"+ppm);
 			if(has)
 				ppm++;
 			else
 				break;
 		}
 	}
 	
 	return ppm;
 }
 
 
 public static double[][] getRankCoordinate(VarMap vm, int maxpoint, boolean multiColor){
 	double [][] coords = null;
 	//boolean has=checkSet(vm,"x");
 	if(maxpoint==0){
 	coords = new double[1][4];
 	coords[0][0]=vm.getValue("x");
 	coords[0][1]=vm.getValue("y");
 	coords[0][2]=vm.getValue("z");
 	coords[0][3]=vm.getValue("color");
 	}
 	else{
 		coords=new double[maxpoint][4];
 		maxpoint--;
 		for(int i=0;i<=maxpoint;i++){
 			coords[i][0]=vm.getValue("x"+i);
 			coords[i][1]=vm.getValue("y"+i);
 			coords[i][2]=vm.getValue("z"+i);
 			if(multiColor)
 				coords[i][3]=vm.getValue("color"+i);
 			else
 				coords[i][3]=vm.getValue("color");
 			//System.out.println(coords[i][0]+" "+coords[i][1]+" "+coords[i][2]+" "+coords[i][3]);
 		}
 	}
 	//System.out.println(coords[0][0]+" "+coords[0][1]+" "+coords[0][2]+" "+coords[0][3]);
 	return coords;
 }
 
 
 
 public static int[] parseTuple(String tuple){
 	
 	
 	tuple = tuple.substring(1,tuple.length()-1);
 	String[] tmp =  tuple.split(",");
 	int[] tres = new int[3];
 	for(int i=0;i<tmp.length;i++){
 		if(i<=tmp.length)
 			tres[i]=Integer.parseInt(tmp[i]);
 		else
 			tres[i]=0;
 	}
 	
 	return tres;
 }

 
 
 public static List<String> getCustomTopoNames(String fileLoc){
 	List<String> names = new ArrayList<String>();
 	BufferedReader br;
 	
		try {
			br = new BufferedReader(new FileReader(new File(fileLoc)));
		

		String s;
		
		 

		while ((s = br.readLine()) != null) {
			
			if(s.startsWith(BEGIN))
			{	
			
			int x1 = s.indexOf('=');
			//int x2 = s.indexOf('"', x1 + 1);

			//String id = s.substring(0,x1);
			String name= s.substring(x1+1);
			names.add(name);//expressions.put(id, exp);
			}

		}
		
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
 	
 	return names;
 }
 
}
