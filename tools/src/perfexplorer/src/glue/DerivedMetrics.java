/**
 * 
 */
package glue;

/**
 * @author khuck
 *
 */
public final class DerivedMetrics {
	public static final String L1_HIT_RATE = "((PAPI_L1_TCA-PAPI_L1_TCM)/PAPI_L1_TCA)";
	public static final String L2_HIT_RATE = "((PAPI_L1_TCM-PAPI_L2_TCM)/PAPI_L1_TCM)";
	public static final String MFLOP_RATE = "(PAPI_FP_INS/P_WALL_CLOCK_TIME)";
	public static final String L1_CACHE_HITS = "(PAPI_L1_TCA-PAPI_L1_TCM)";
	public static final String MEM_ACCESSES = "(PAPI_L1_TCA/P_WALL_CLOCK_TIME)";
	public static final String L2_CACHE_HITS = "(PAPI_L1_TCM-PAPI_L2_TCM)";
	public static final String L2_ACCESSES = "(PAPI_L1_TCM/P_WALL_CLOCK_TIME)";
	public static final String TOT_INS_RATE = "(PAPI_TOT_INS/P_WALL_CLOCK_TIME)";
}
