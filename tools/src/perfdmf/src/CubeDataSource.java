package edu.uoregon.tau.perfdmf;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;

import javax.swing.JOptionPane;
import javax.swing.filechooser.FileFilter;

import scalasca.cubex.cube.Cube;
import scalasca.cubex.cube.LocationGroupType;
import scalasca.cubex.cube.LocationType;
import scalasca.cubex.cube.datalayout.data.value.CubeTauAtomicMetric;
import scalasca.cubex.cube.datalayout.data.value.ValueType;
import scalasca.cubex.cube.errors.BadCubeReportLayoutException;
import scalasca.cubex.cube.errors.BadSyntaxException;
import scalasca.cubex.cube.errors.NotEnumeratedCnodeException;


/**
 * Reader for cube data
 *
 *
 * @see <a href="http://www.fz-juelich.de/zam/kojak/">
 * http://www.fz-juelich.de/zam/kojak/</a> for more information about cube
 * 
 * <P>CVS $Id: CubeDataSource.java,v 1.4 2009/10/06 07:17:55 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.4 $
 */
public class CubeDataSource extends DataSource
{

	private File file;
	private Cube cube = null;
// 	private volatile Cube4XMLHandler handler = new Cube4XMLHandler(this);
// 	private volatile TrackerInputStream tracker;

	private Group defaultGroup;
	private Group callpathGroup;
	private Metric calls = new Metric();
	private boolean calls_are_in_cube = false;



	private Map<scalasca.cubex.cube.Cnode, Function> cube2tau_cnodes = new HashMap<scalasca.cubex.cube.Cnode, Function>();
	private Map<scalasca.cubex.cube.Location, Thread> cube2tau_threads = new HashMap<scalasca.cubex.cube.Location, Thread>();
	private Map<scalasca.cubex.cube.Metric, Metric> cube2tau_metrics = new HashMap<scalasca.cubex.cube.Metric, Metric>();




	private Map<Function, Function> parentMap = new HashMap<Function, Function>(); // map functions to their parent functions ("A=>B=>C" -> "A=>B")
	private Map<Function, Function> flatMap = new HashMap<Function, Function>(); // map functions to their flat functions ("A=>B=>C" -> "C")


	private double progress_value = 0;
	private String progress_message = "";


	/**
	* Constructor for CubeDataSource
	* @param file      file containing cube data
	*/
	public CubeDataSource(File file)
	{
		super();
		this.file = file;
		calls.setName("Number of calls");

		defaultGroup = addGroup("CUBE_DEFAULT");
		callpathGroup = addGroup("CUBE_CALLPATH");

	}


	static public FileFilter getFilesFilter() { return scalasca.cubex.cube.Cube.getCubeFilesFilter(); }

    
	public void load() throws FileNotFoundException, IOException, DataSourceException, SQLException
	{
		try {
		long time = System.currentTimeMillis();

			progress_message = "Start loading...";

			cube = new Cube();
			progress_message = "Cube object created...";

			cube.openCubeReport(file.getPath());
			progress_message = "Construct TAU dimensions out of CUBE dimensions...";
			progress_value = 0.3; // sofar 50% done

			ArrayList<scalasca.cubex.cube.Metric> metrics = cube.get_metv();
			ArrayList<scalasca.cubex.cube.Metric> root_metrics = cube.get_root_metv();
			// ArrayList<scalasca.cubex.cube.Region> regions =
			// cube.get_regionv();
			ArrayList<scalasca.cubex.cube.Cnode> cnodes = cube.get_cnodev();
			ArrayList<scalasca.cubex.cube.Cnode> root_cnodes = cube.get_root_cnodev();
			// ArrayList<scalasca.cubex.cube.Machine> machines =
			// cube.get_machv();
			// ArrayList<scalasca.cubex.cube.Node> nodes = cube.get_nodev();
			ArrayList<scalasca.cubex.cube.LocationGroup> lgs = cube.get_location_groupv();
			ArrayList<scalasca.cubex.cube.Location> locations = cube.get_locationv();
			ArrayList<scalasca.cubex.cube.Cartesian> topologies = cube.get_cartv();

			ArrayList<scalasca.cubex.cube.Metric> context_events = new ArrayList<scalasca.cubex.cube.Metric>(); // collect metrics from cube, which carries context events for further processing.
			ArrayList<scalasca.cubex.cube.Metric> interval_events = new ArrayList<scalasca.cubex.cube.Metric>(); // collect metrics from cube, which carries context events for further processing.

			progress_message = "Construct TAU metrics out of CUBE metrics...";
			progress_value = 0.32; // sofar 32% done
			for (scalasca.cubex.cube.Metric met : root_metrics)
			{
				addMetrics(context_events, interval_events, met, "");
			}

			progress_message = "Construct TAU functions out of CUBE calltree...";
			progress_value = 0.34; // sofar 34% done
			for (scalasca.cubex.cube.Cnode cnode : root_cnodes)
			{
				addCnodes(cnode, "");
			}

			progress_message = "Construct TAU threads out of CUBE system tree...";
			progress_value = 0.36; // sofar 36% done
			for (scalasca.cubex.cube.LocationGroup lg : lgs)
			{
				addProcess(lg);
			}

			progress_message = "Construct TAU topologies out of CUBE topologies...";
			progress_value = 0.40; // sofar 40% done
			int num =1;
			for (scalasca.cubex.cube.Cartesian cart: topologies)
			{
				if (cart.get_ndim() != 3)
				{
					//Don't open prompts unless there is a main window already present
                                        int numwins = java.awt.Window.getWindows().length;
                                        if(numwins>0){
					int optionReturn=JOptionPane.YES_OPTION;
					
					try {JOptionPane.showConfirmDialog(
							null,
							new String("Topology " + num + "( "+cart.get_name()+") has a different number of coordinates ("+ cart.get_ndim()+") than 3. \n Visualization of this topology will fail. \n Add to TAU profile anyway?"),
							"No supported topologies are detected",
							JOptionPane.YES_NO_OPTION);}
					catch(Exception e){
								
							}
					
					if (optionReturn == JOptionPane.NO_OPTION) 
					{
						continue;
					}
					}
				}
				String prefix = "Topo"+num;
				getMetaData().put(prefix+" Name", cart.get_name());
				getMetaData().put(prefix+" Size", createDimString(cart.get_dimv()));
				getMetaData().put(prefix+" isTorus", createPeriodicityString(cart.get_periodv()));

				for (scalasca.cubex.cube.Location loc : locations)
				{
					addThreadCoordinates(prefix, cart, loc);
				}
				num++;
			}
			progress_message = "Cube object filled with data...";
			progress_value = 0.55; // sofar 55% done
			double progress_step = (0.40)/( (calls_are_in_cube)?(metrics.size()):(metrics.size()+1)); // if there no visits in cube, one run more is needed
			int metric_number = 0;
			for (scalasca.cubex.cube.Metric met : interval_events)
			{
				metric_number++;
				progress_message = "Load data from CUBE to TAU for metric: "+met.getDisplayName() + " ( " + metric_number+ "/"+interval_events.size() +  ")";

				for (scalasca.cubex.cube.Cnode cnode : cnodes)
				{
					for (scalasca.cubex.cube.Location loc : locations)
					{
						double value = cube.get_sev(met, cnode, loc);
						Thread _thread  = cube2tau_threads.get(loc);
						if (_thread == null)
                            continue;
						Function _function  = cube2tau_cnodes.get(cnode);
						FunctionProfile _fp = _thread.getFunctionProfile(_function);
						Metric _met  = cube2tau_metrics.get(met);

						if (_fp == null)
						{
							_fp = new FunctionProfile(_function, getNumberOfMetrics());
							_thread.addFunctionProfile(_fp);
						}
						addValueToProfile(_met, _fp, _function, _thread, value*((met.getUOM().equalsIgnoreCase("sec"))?1000*1000:1));
					}
				}
				progress_value += progress_step;


			}


			if (!calls_are_in_cube)
			{
				progress_message = "Create visits for TAU (becuse Cube is missing the metric \"Visits\")";
				for (scalasca.cubex.cube.Cnode cnode : cnodes)
				{
                    for (scalasca.cubex.cube.Location loc : locations)
					{
						double value = 1.;

						Thread _thread  = cube2tau_threads.get(loc);
                        if (_thread == null)
                            continue;
						Function _function  = cube2tau_cnodes.get(cnode);
						FunctionProfile _fp = _thread.getFunctionProfile(_function);

						if (_fp == null) // actually all cnodes shoud be have been visited and created in previous part above
						{
							_fp = new FunctionProfile(_function, getNumberOfMetrics());
							_thread.addFunctionProfile(_fp);
						}
						addValueToProfile(calls, _fp, _function, _thread, value);
					}
				}
				progress_value += progress_step;

			}



			metric_number=0;
			for (scalasca.cubex.cube.Metric context_event : context_events)
			{
				metric_number++;
				progress_message = "Load context events "+context_event.getDisplayName() +" from CUBE to TAU :  ( " + metric_number+ "/"+context_events.size() +  ")";
				for (scalasca.cubex.cube.Cnode cnode : cnodes)
				{
                    for (scalasca.cubex.cube.Location loc : locations)
					{
						CubeTauAtomicMetric value = (CubeTauAtomicMetric)cube.get_sev_adv(context_event, cnode, loc);

						if (value.getN() != 0)
						{

							Thread _thread  = cube2tau_threads.get(loc);
							Function _function  = cube2tau_cnodes.get(cnode);

							String name =  context_event.getDisplayName() + " : " + _function.toString();
							UserEvent userEvent  = this.addUserEvent	(name);
							UserEventProfile  uep = _thread.getUserEventProfile(userEvent);
							if (uep == null)
							{
								uep = new UserEventProfile(userEvent);
								_thread.addUserEventProfile(uep);

								uep.setNumSamples(value.getN());
								uep.setMaxValue(value.getMax());
								uep.setMinValue(value.getMin());
								uep.setMeanValue(value.getAvg());
								uep.setSumSquared(value.getSum2());
								uep.updateMax();
							}
						}
					}
				}
				progress_value += progress_step;
			}



			progress_message = "Postprocessing...(releade memory)";
			cube= null;
			metrics = null;
			root_metrics = null;
			// regions = null;
			cnodes = null;
			root_cnodes = null;
			// machines = null;
			// nodes = null;
			lgs = null;
			locations = null;
			topologies = null;
			cube2tau_cnodes=null;
			cube2tau_metrics=null;
			cube2tau_threads=null;
			parentMap = null;
			flatMap = null;
			context_events = null;
			interval_events = null;
			System.gc();

			progress_message = "Postprocessing...(set group names)";
			this.setGroupNamesPresent(true);
			progress_message = "Generate derived data...";
			progress_value = 0.95; //95% is done 
			this.generateDerivedData();
			progress_value = 1.; // finish
			progress_message = "Generate derived data...done";
			time = (System.currentTimeMillis()) - time;
			//System.out.println("Time to process (in milliseconds): " + time);
		}
		catch(BadSyntaxException e)
		{
		throw new DataSourceException(e);
		}
		catch(NotEnumeratedCnodeException e)
		{
		throw new DataSourceException(e);
		}
		catch(BadCubeReportLayoutException e)
		{
		throw new DataSourceException(e);
		}

	}



	void addValueToProfile(Metric _met, FunctionProfile _fp, Function _function, Thread _thread, double value)
	{

		if (_met == calls)
		{
			_fp.setNumCalls(value);
		}
		else
		{
			_fp.setExclusive(_met.getID(), value);
		}

		if (_function.isCallPathFunction())
		{ // get the flat profile (C for A => B => C)

			FunctionProfile flatFP = getFlatFunctionProfile(_thread, _function);

			if (_met == calls)
			{
				flatFP.setNumCalls(flatFP.getNumCalls() + value);
			} else
			{
				flatFP.setExclusive(_met.getID(), value + flatFP.getExclusive(_met.getID()));
				//childFP.setInclusive(metric.getID(), value + childFP.getInclusive(metric.getID()));
			}
		}

		if (_met == calls)
		{
			addCallsToParent(_thread, _fp, value);

		//addToNumSubr(thread, fp, value);
		} else
		{
			addToInclusive(_met, _thread, _fp, value);
		}
	}








	// given A => B => C, this retrieves the FP for C
	private FunctionProfile getFlatFunctionProfile(Thread thread, Function function)
	{
		if (!function.isCallPathFunction())
		{
			return null;
		}

		Function childFunction = flatMap.get(function);

		if (childFunction == null)
		{
			String childName = function.getName().substring(function.getName().lastIndexOf("=>") + 2).trim();
			childFunction = this.addFunction(childName);
			childFunction.addGroup(defaultGroup);
			flatMap.put(function, childFunction);
		}
		FunctionProfile childFP = thread.getFunctionProfile(childFunction);
		if (childFP == null)
		{
			childFP = new FunctionProfile(childFunction, this.getNumberOfMetrics());
			thread.addFunctionProfile(childFP);
		}
		return childFP;

	}

	// retrieve the parent profile on a given thread (A=>B for A=>B=>C)
	private FunctionProfile getParent(Thread thread, Function function)
	{
		if (!function.isCallPathFunction())
		{
			return null;
		}

		Function parentFunction = parentMap.get(function);
		if (parentFunction == null)
		{
			String functionName = function.getName();
			String parentName = functionName.substring(0, functionName.lastIndexOf("=>"));
			parentFunction = this.getFunction(parentName);
			parentMap.put(function, parentFunction);
		}
		FunctionProfile parent = thread.getFunctionProfile(parentFunction);
		return parent;
	}




	// recursively add a value to the inclusive amount (go up the tree)
	private void addToInclusive(Metric metric, Thread thread, FunctionProfile fp, double value)
	{
		// add to this fp
		fp.setInclusive(metric.getID(), value + fp.getInclusive(metric.getID()));

		// add to our flat
		FunctionProfile flatFP = getFlatFunctionProfile(thread, fp.getFunction());

		if (flatFP != null)
		{
			flatFP.setInclusive(metric.getID(), value + flatFP.getInclusive(metric.getID()));
		}

		// recurse to A => B if this is A => B => C
		FunctionProfile parent = getParent(thread, fp.getFunction());
		if (parent != null)
		{
			addToInclusive(metric, thread, parent, value);
		}

	}


	// recursively add a call number to the parents (go up the tree)
	private void addCallsToParent(Thread thread, FunctionProfile fp, double value)
	{

		// add numcalls to numsubr of parent (and its flat profile)
		// just parents, not parents-of-parents-of-parents-of...
		FunctionProfile parent = getParent(thread, fp.getFunction());
		if (parent != null)
		{
			parent.setNumSubr(parent.getNumSubr() + value);
			FunctionProfile flat = getFlatFunctionProfile(thread, parent.getFunction());
			if (flat != null)
			{
				flat.setNumSubr(flat.getNumSubr() + value);
			}
		}

		// recurse to A => B if this is A => B => C
		// If propagation to parent-of-parent is wished, uncomment below
// 		if (parent != null)
// 		{
// 			addCallsToParent(thread, parent, value);
// 		}

	}









	public int getProgress()
	{
		// int _progress;
		if (cube == null)
			return 0;
		if (cube.isLoading())
			return(int)(cube.getProgress()*30.); // while progress inside of cube is 50% of overall.
		return (int)(progress_value*100.);
	}

	public String getProgressMessage()
	{
		if (cube == null)
			return "";
		if (cube.isLoading())
			return cube.getProgressMessage();
		return progress_message;
	}


	public void cancelLoad() {
		// TODO Auto-generated method stub

	}

	private  void addMetrics(ArrayList<scalasca.cubex.cube.Metric> _tau_metrics, ArrayList<scalasca.cubex.cube.Metric> _metrics, scalasca.cubex.cube.Metric met, String prefix)
	{
		String new_prefix = prefix + met.getDisplayName();
		String tauMetricName = new_prefix;
		String _tauMetricName = new_prefix.toUpperCase();

		if ( met.getDType().equalsIgnoreCase(ValueType.TAU_ATOMIC_METRIC.toString()))
		{
			_tau_metrics.add(met);
		}
		else
		{
			_metrics.add(met);
			if (_tauMetricName.indexOf("TIME") != -1 && met.getUOM().toUpperCase().equals("OCC"))
			{
				tauMetricName += "_count";
			}
			if (met.getUniqName().equalsIgnoreCase("visits"))
			{
				cube2tau_metrics.put(met, calls);
				calls_are_in_cube = true;
			}
			else
			{
				Metric tau_metric = addMetric(tauMetricName);
				cube2tau_metrics.put(met, tau_metric);
			}
		}


		if (met.getNumberOfChildren() != 0)
			new_prefix = new_prefix + " => ";
		for (Object _met: met.getAllChildren().toArray())
		{
			addMetrics(_tau_metrics, _metrics,  (scalasca.cubex.cube.Metric)_met, new_prefix);
		}
	}


	private void addCnodes(scalasca.cubex.cube.Cnode cnode, String prefix)
	{

		String new_prefix = prefix + cnode.getRegion().getName();
		for (String key : cnode.get_str_parameters().keySet())
		{
			new_prefix  +=
				" < "+key + " > = < " + cnode.get_str_parameter(key) + " > ";

		}
		for (String key : cnode.get_num_parameters().keySet())
		{
			new_prefix  +=
				" < "+key + " > = < " + cnode.get_num_parameter(key) + " > ";
		}


		String tauFunctionName = new_prefix;
		Function function = addFunction(tauFunctionName);
		cube2tau_cnodes.put(cnode, function);

		if (prefix.equals(""))
			function.addGroup(defaultGroup);
		else
			function.addGroup(callpathGroup);

// 		System.out.println("ADDED FUNCTION" + tauFunctionName);
		if (cnode.getNumberOfChildren() != 0)
			new_prefix = new_prefix + " => ";
		for (Object _cnode: cnode.getAllChildren().toArray())
		{
			addCnodes((scalasca.cubex.cube.Cnode)_cnode, new_prefix);
		}
	}


	private void addProcess(scalasca.cubex.cube.LocationGroup lg)
	{

        if (lg.getType() != LocationGroupType.PROCESS)
            return;
		Node node = addNode((int)lg.getRank());
		Context context = node.addContext(0);
		int threadID=0;
		for (Object _thread: lg.getAllChildren().toArray())
		{
			addThread((scalasca.cubex.cube.Location)_thread, context,threadID);
			threadID++;
		}
	}

	private void addThread(scalasca.cubex.cube.Location location, Context context, int threadID)
	{
        if (location.getType() != LocationType.CPU_THREAD)
            return;
		Thread tau_thread = context.addThread(threadID, getNumberOfMetrics());
		Iterator<Entry<String, String>> it = location.get_attrs().entrySet()
				.iterator();
		while (it.hasNext()) {
			Entry<String, String> item = it.next();
			tau_thread.getMetaData().put(item.getKey(), item.getValue());
		}

		cube2tau_threads.put(location, tau_thread);
// 		System.out.println("         Added "+ thread.getName());
	}



	private String createDimString(ArrayList<Integer> dimensions)
	{
		String _str = "(";
		int i= 0;
		for (Integer dimsize: dimensions)
		{
			_str += dimsize;
			if (i<(dimensions.size()-1))
				_str += ",";
		i++;
		}
		_str += ")";
		return _str;
	}


	private String createPeriodicityString(ArrayList<Boolean> periodicity)
	{
		String _str = "(";
		int i= 0;
		for (Boolean isperiodic: periodicity)
		{
			_str += (isperiodic.booleanValue() == true)?"1":"0";
			if (i<(periodicity.size()-1))
				_str += ",";
		i++;
		}
		_str += ")";
		return _str;
	}

	private String createCoordinatesString(ArrayList<Integer> _coords)
	{
		String _str = "(";
		int i= 0;
		for (Integer coord: _coords)
		{
			_str += coord;
			if (i<(_coords.size()-1))
				_str += ",";
		i++;
		}
		_str += ")";
		return _str;
	}


	private void addThreadCoordinates(String prefix, scalasca.cubex.cube.Cartesian cart, scalasca.cubex.cube.Location _cube_thread)
	{
		ArrayList<Integer> _coords = cart.get_coordv (_cube_thread);
		Thread	_tau_thread = cube2tau_threads.get(_cube_thread);
		if (_tau_thread == null)
            return;
		if (_coords.size() == 0)
		{
                    	_tau_thread.getMetaData().put(prefix+" Coords", "()");
		}else
		{
			_tau_thread.getMetaData().put(prefix+" Coords", createCoordinatesString(_coords));
		}
	}


}
