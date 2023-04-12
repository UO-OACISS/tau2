/*
 * Copyright (c) 2014-2023 Kevin Huck
 * Copyright (c) 2014-2023 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifdef TAU_ROCM_SMI
#include "tau_rocm_smi.hpp"
#include <Profile/Profiler.h>
#include <Profile/TauSampling.h>
#include <Profile/TauMetrics.h>
#include <Profile/TauAPI.h>
#include <Profile/TauPlugin.h>
#include <Profile/TauMetaData.h>
#include <sstream>

#include "json.h"
using json = nlohmann::json;
extern json configuration;
int& get_my_rank(void);

#define RSMI_CALL(call)                                                      \
do {                                                                         \
    rsmi_status_t _status = call;                                            \
    if (_status != RSMI_STATUS_SUCCESS) {                                    \
        const char *errstr;                                                  \
        if (rsmi_status_string(_status, &errstr) == RSMI_STATUS_SUCCESS) {   \
        fprintf(stderr, "%s:%d: error: function %s failed with error %d: %s.\n", \
                __FILE__, __LINE__, #call, _status, errstr);                 \
        exit(-1);                                                            \
        }                                                                    \
    }                                                                        \
} while (0);

#define RSMI_CALL_NOEXIT(call)                                               \
do {                                                                         \
    rsmi_status_t _status = call;                                            \
    if (_status != RSMI_STATUS_SUCCESS) {                                    \
        const char *errstr;                                                  \
        if (rsmi_status_string(_status, &errstr) == RSMI_STATUS_SUCCESS) {   \
        fprintf(stderr, "%s:%d: error: function %s failed with error %d: %s.\n", \
                __FILE__, __LINE__, #call, _status, errstr);                 \
        success = false;                                                     \
        }                                                                    \
    }                                                                        \
} while (0);

#define MILLIONTH 1.0e-6 // scale to MB
#define BILLIONTH 1.0e-9 // scale to GB
#define PCIE_THROUGHPUT 1.0e-3  // to scale KB to MB
#define NVLINK_BW 1.0e-3  // to scale MB/s to GB/s
#define WATTS 1.0e-6  // scale uW to W
#define JOULES 1.0e-6  // scale uJ to J
#define VOLTAGE 1.0e-3  // scale mV to V
#define CELCIUS 1.0e-3  // scale mC to C
#define PERCENT 1.0e2  // scale fraction to %

void write_scatterplot_point(const std::string& name, double value);

bool include_event(const char * component, const char * event_name);

namespace tau { namespace rsmi {

std::set<uint32_t> monitor::activeDeviceIndices;
std::mutex monitor::indexMutex;

monitor::monitor (void) {
    success = true; // used by RSMI_CALL_NOEXIT
    RSMI_CALL_NOEXIT(rsmi_init(0));
    if (!success) return;
    // get the total device count
    RSMI_CALL(rsmi_num_monitor_devices(&deviceCount));
    rsmi_version_t version;
    RSMI_CALL(rsmi_version_get(&version));
    TAU_VERBOSE("RSMI Version %d.%d.%d build %s, found %d total devices\n",
                version.major, version.minor, version.patch,
                version.build, deviceCount);

    devices.reserve(deviceCount);
    // get the unit handles
    for (uint32_t dv_ind = 0 ; dv_ind < deviceCount ; dv_ind++) {
        DeviceInfo info;
        // Get Unique ID
        //RSMI_CALL(rsmi_dev_unique_id_get(dv_ind, &info.unique_id));
        //devices.push_back(info.unique_id);
        queried_once.push_back(false);
        // Get the device id associated with the device with provided device index.
        RSMI_CALL( rsmi_dev_id_get (dv_ind, &info.id));
        // Get the SKU for a desired device associated with the device with provided device index.
        // RSMI_CALL( rsmi_dev_sku_get (dv_ind, info.sku)); <-- not defined in library!
        // Get the device vendor id associated with the device with provided device index.
        RSMI_CALL( rsmi_dev_vendor_id_get (dv_ind, &info.vendor_id));
        // Get the name string of a gpu device.
        RSMI_CALL( rsmi_dev_name_get (dv_ind, info.name, info.str_len));
        // Get the brand string of a gpu device.
        RSMI_CALL( rsmi_dev_brand_get (dv_ind, info.brand, info.str_len));
        // Get the name string for a give vendor ID.
        RSMI_CALL( rsmi_dev_vendor_name_get (dv_ind, info.vendor_name, info.str_len));
        // Get the vram vendor string of a gpu device.
        RSMI_CALL( rsmi_dev_vram_vendor_get (dv_ind, info.vram_vendor_name, info.str_len));
        // Get the serial number string for a device.
        RSMI_CALL( rsmi_dev_serial_number_get (dv_ind, info.serial_number, info.str_len));
        // Get the subsystem device id associated with the device with provided device index.
        RSMI_CALL( rsmi_dev_subsystem_id_get (dv_ind, &info.subsystem_id));
        // Get the name string for the device subsytem.
        RSMI_CALL( rsmi_dev_subsystem_name_get (dv_ind, info.subsystem_name, info.str_len));
        // Get the drm minor number associated with this device.
        RSMI_CALL( rsmi_dev_drm_render_minor_get (dv_ind, &info.drm_minor_id));
        // Get the device subsystem vendor id associated with the device with provided device index.
        RSMI_CALL( rsmi_dev_subsystem_vendor_id_get (dv_ind, &info.subsystem_vendor_id));
        deviceInfos.push_back(std::move(info));
        activateDeviceIndex(dv_ind);
    }
}

monitor::~monitor (void) {
    if (!success) return;
    RSMI_CALL(rsmi_shut_down());
}

bool include_device(uint32_t index) {
    int dev_per_node{
        configuration.count("devices_per_node") > 0 ?
        (int)(configuration["devices_per_node"]) : 1};
    int dev_per_proc{
        configuration.count("devices_per_process") > 0 ?
        (int)(configuration["devices_per_process"]) : 1};
    uint32_t my_device{(uint32_t)(get_my_rank() % dev_per_node)};
    //TAU_VERBOSE("Dev/node: %d, dev/proc: %d, my_device: %lu, index: %lu\n", dev_per_node, dev_per_proc, my_device, index);
    if(my_device == index) return true;
    /* need to figure out the method for handling more than 1 device per process */
    //if(dev_per_node == dev_per_proc) return true;
    return false;
}

void monitor::query(void) {
    if (!success) return;
    TAU_VERBOSE("RSMI query!\n");
    indexMutex.lock();
    // use the copy constructor to get the set of active indices
    std::set<uint32_t> indexSet{activeDeviceIndices};
    indexMutex.unlock();
    // 100 ought to be enough!
    static float previous_energy[100] = {0};
    static float previous_gfx_activity[100] = {0};
    static float previous_mem_activity[100] = {0};
    static uint64_t previous_activity_timestamp[100] = {0L};

    for (uint32_t d : indexSet) {
        uint64_t power = 0;
        uint32_t sensor_index = 0;
        uint64_t timestamp;

        if (!queried_once[d]) {
            TAU_VERBOSE("%s\n", deviceInfos[d].to_string().c_str());
        }

        // power, in microwatts
        RSMI_CALL(rsmi_dev_power_ave_get(d, sensor_index, &power));
        std::stringstream ss;
        ss << "GPU: Device " << d << " Power (W)";
        std::string tmp{ss.str()};
        double value = (double)(power) * WATTS;
        if(include_event("rsmi", tmp.c_str())) {
            Tau_trigger_userevent(tmp.c_str(), value);
            write_scatterplot_point(tmp.c_str(), value);
        }
        ss.str("");

// not available until 4.3?
        // energy, in microjoules
        uint64_t energy = 0;
        float counter_resolution = 0;
        RSMI_CALL(rsmi_dev_energy_count_get(d, &energy, &counter_resolution, &timestamp));
        ss << "GPU: Device " << d << " Energy (J)";
        tmp = ss.str();
        value = (double)(energy) * JOULES;
        if (previous_energy[d] == 0.0) {
            previous_energy[d] = value;
            value = 0.0;
            // don't sample the first value, it's an offset.
        } else {
            auto tval = value;
            value = value - previous_energy[d];
            previous_energy[d] = tval;
            TAU_VERBOSE("Resolution: %f, Value: %f\n", counter_resolution, value);
            if(include_event("rsmi", tmp.c_str())) {
                Tau_trigger_userevent(tmp.c_str(), value);
                write_scatterplot_point(tmp.c_str(), value);
            }
        }
        ss.str("");

        // 6.7 - memory queries
        /*
        if (!queried_once[d]) {
        	uint64_t memory_total;
            RSMI_CALL(rsmi_dev_memory_total_get(d, RSMI_MEM_TYPE_VRAM, &memory_total));
            ss << "GPU: Device " << d << " Memory Total, VRAM (GB)";
            tmp = ss.str();
            value = (double)(memory_total) * BILLIONTH;
        if(include_event("rsmi", tmp.c_str())) {
            Tau_trigger_userevent(tmp.c_str(), value);
            write_scatterplot_point(tmp.c_str(), value);
        }
            ss.str("");
            RSMI_CALL(rsmi_dev_memory_total_get(d, RSMI_MEM_TYPE_VIS_VRAM, &memory_total));
            ss << "GPU: Device " << d << " Memory Total, Vis. VRAM (GB)";
            tmp = ss.str();
            value = (double)(memory_total) * BILLIONTH;
        if(include_event("rsmi", tmp.c_str())) {
            Tau_trigger_userevent(tmp.c_str(), value);
            write_scatterplot_point(tmp.c_str(), value);
        }
            ss.str("");
            RSMI_CALL(rsmi_dev_memory_total_get(d, RSMI_MEM_TYPE_GTT, &memory_total));
            ss << "GPU: Device " << d << " Memory Total, GTT (GB)";
            tmp = ss.str();
            value = (double)(memory_total) * BILLIONTH;
        if(include_event("rsmi", tmp.c_str())) {
            Tau_trigger_userevent(tmp.c_str(), value);
            write_scatterplot_point(tmp.c_str(), value);
        }
            ss.str("");
        }
        */

        uint64_t memory_usage = 0;
        RSMI_CALL(rsmi_dev_memory_usage_get(d, RSMI_MEM_TYPE_VRAM, &memory_usage));
        ss << "GPU: Device " << d << " Memory Used, VRAM (GB)";
        tmp = ss.str();
        value = (double)(memory_usage) * BILLIONTH;
        if(include_event("rsmi", tmp.c_str())) {
            Tau_trigger_userevent(tmp.c_str(), value);
            write_scatterplot_point(tmp.c_str(), value);
        }
        ss.str("");
        RSMI_CALL(rsmi_dev_memory_usage_get(d, RSMI_MEM_TYPE_VIS_VRAM, &memory_usage));
        ss << "GPU: Device " << d << " Memory Used, Vis. VRAM (GB)";
        tmp = ss.str();
        value = (double)(memory_usage) * BILLIONTH;
        if(include_event("rsmi", tmp.c_str())) {
            Tau_trigger_userevent(tmp.c_str(), value);
            write_scatterplot_point(tmp.c_str(), value);
        }
        ss.str("");
        RSMI_CALL(rsmi_dev_memory_usage_get(d, RSMI_MEM_TYPE_GTT, &memory_usage));
        ss << "GPU: Device " << d << " Memory Used, GTT (GB)";
        tmp = ss.str();
        value = (double)(memory_usage) * BILLIONTH;
        if(include_event("rsmi", tmp.c_str())) {
            Tau_trigger_userevent(tmp.c_str(), value);
            write_scatterplot_point(tmp.c_str(), value);
        }
        ss.str("");

        uint32_t memory_busy_percent = 0;
        RSMI_CALL(rsmi_dev_memory_busy_percent_get(d, &memory_busy_percent));
        ss << "GPU: Device " << d << " Memory Busy (%)";
        tmp = ss.str();
        value = (double)(memory_busy_percent);
        if(include_event("rsmi", tmp.c_str())) {
            Tau_trigger_userevent(tmp.c_str(), value);
            write_scatterplot_point(tmp.c_str(), value);
        }
        ss.str("");

        uint32_t memory_pages = 0;
        RSMI_CALL(rsmi_dev_memory_reserved_pages_get(d, &memory_pages, NULL));
        ss << "GPU: Device " << d << " Memory Reserved Pages";
        tmp = ss.str();
        value = (double)(memory_pages);
        if(include_event("rsmi", tmp.c_str())) {
            Tau_trigger_userevent(tmp.c_str(), value);
            write_scatterplot_point(tmp.c_str(), value);
        }
        ss.str("");

/* The MI250X integrated, liquid cooled GCDs don't have fans... */
#if 0
		// Get the fan speed in RPMs of the device with the specified device index and 0-based sensor index.
		int64_t speed = 0;
		RSMI_CALL_NOEXIT(rsmi_dev_fan_rpms_get (d, sensor_index, &speed));
        if (speed > 0) {
		    // Get the max. fan speed of the device with provided device index.
		    uint64_t max_speed = 0;
		    RSMI_CALL_NOEXIT(rsmi_dev_fan_speed_max_get (d, sensor_index, &max_speed));
		    double speed_percent = (speed == 0) ? 0.0 : (double)(speed) / (double)(max_speed);
            ss << "GPU: Device " << d << " Fan Speed (%)";
            tmp = ss.str();
            value = speed_percent * PERCENT;
        if(include_event("rsmi", tmp.c_str())) {
            Tau_trigger_userevent(tmp.c_str(), value);
            write_scatterplot_point(tmp.c_str(), value);
        }
            ss.str("");
        }
#endif

		// Get the temperature metric value for the specified metric, from the specified temperature sensor on the specified
		int64_t temperature = 0;
		RSMI_CALL(rsmi_dev_temp_metric_get(d, sensor_index, RSMI_TEMP_CURRENT, &temperature));
        ss << "GPU: Device " << d << " Temperature (C)";
        tmp = ss.str();
        value = (double)(temperature) * CELCIUS;
        if(include_event("rsmi", tmp.c_str())) {
            Tau_trigger_userevent(tmp.c_str(), value);
            write_scatterplot_point(tmp.c_str(), value);
        }
        ss.str("");

		// Get the voltage metric value for the specified metric, from the specified voltage sensor on the specified device.
		int64_t voltage = 0;
		RSMI_CALL(rsmi_dev_volt_metric_get (d, RSMI_VOLT_TYPE_VDDGFX, RSMI_VOLT_CURRENT, &voltage));
        ss << "GPU: Device " << d << " Voltage (V)";
        tmp = ss.str();
        value = (double)(voltage) * VOLTAGE;
        if(include_event("rsmi", tmp.c_str())) {
            Tau_trigger_userevent(tmp.c_str(), value);
            write_scatterplot_point(tmp.c_str(), value);
        }
        ss.str("");

		// Get percentage of time device is busy doing any processing.
		uint32_t busy_percent = 0;
		RSMI_CALL(rsmi_dev_busy_percent_get (d, &busy_percent));
        ss << "GPU: Device " << d << " Device Busy (%)";
        tmp = ss.str();
        value = (double)(busy_percent);
        if(include_event("rsmi", tmp.c_str())) {
            Tau_trigger_userevent(tmp.c_str(), value);
            write_scatterplot_point(tmp.c_str(), value);
        }
        ss.str("");

		// Get coarse grain utilization counter of the specified device.
		rsmi_utilization_counter_t utilization_counters[2];
        utilization_counters[0].type = RSMI_UTILIZATION_COUNTER_FIRST;
        utilization_counters[1].type = RSMI_COARSE_GRAIN_MEM_ACTIVITY;
		RSMI_CALL(rsmi_utilization_count_get (d, utilization_counters, 2, &timestamp));
        ss << "GPU: Device " << d << " GFX Activity (%)";
        tmp = ss.str();
        value = (double)(utilization_counters[0].value);
        if (previous_gfx_activity[d] == 0.0) {
            previous_gfx_activity[d] = value;
            value = 0.0;
            // don't sample the first value, it's our accumulated baseline.
        } else {
            auto tval = value;
            value = value - previous_gfx_activity[d];
            previous_gfx_activity[d] = tval;
            /* OK, here's what happens. The utilization is captured every *millisecond*,
             * so we have to get the number of milliseconds between queries and get the
             * accumulated utilization, and divide it to get the average over the last
             * time period. */
            auto duration = timestamp - previous_activity_timestamp[d];
            // convert the duration from nanoseconds to milliseconds
            float ms = duration * 1.0e-6;
            value = value / ms;
            // validation, dude
            value = value < 0.0 ? 0.0 : value;
            value = value > 100.0 ? 100.0 : value;
            if(include_event("rsmi", tmp.c_str())) {
                Tau_trigger_userevent(tmp.c_str(), value);
                write_scatterplot_point(tmp.c_str(), value);
            }
        }
        ss.str("");
        ss << "GPU: Device " << d << " Memory Activity (%)";
        tmp = ss.str();
        value = (double)(utilization_counters[1].value);
        if (previous_mem_activity[d] == 0.0) {
            previous_mem_activity[d] = value;
            value = 0.0;
            // don't sample the first value, it's our accumulated baseline.
        } else {
            auto tval = value;
            value = value - previous_mem_activity[d];
            previous_mem_activity[d] = tval;
            /* OK, here's what happens. The utilization is captured every *millisecond*,
             * so we have to get the number of milliseconds between queries and get the
             * accumulated utilization, and divide it to get the average over the last
             * time period. */
            auto duration = timestamp - previous_activity_timestamp[d];
            // convert the duration from nanoseconds to milliseconds
            float ms = duration * 1.0e-6;
            value = value / ms;
            // validation, dude
            value = value < 0.0 ? 0.0 : value;
            value = value > 100.0 ? 100.0 : value;
            if(include_event("rsmi", tmp.c_str())) {
                Tau_trigger_userevent(tmp.c_str(), value);
                write_scatterplot_point(tmp.c_str(), value);
            }
        }
        previous_activity_timestamp[d] = timestamp;
        ss.str("");

		// This function retrieves the gpu metrics information.
		rsmi_gpu_metrics_t pgpu_metrics;
        memset(&pgpu_metrics, 0, sizeof(rsmi_gpu_metrics_t));
		RSMI_CALL(rsmi_dev_gpu_metrics_info_get (d, &pgpu_metrics));
        ss << "GPU: Device " << d << " Clock Frequency, GLX (MHz)";
        tmp = ss.str();
        value = (double)(pgpu_metrics.current_gfxclk);
        if(include_event("rsmi", tmp.c_str())) {
            Tau_trigger_userevent(tmp.c_str(), value);
            write_scatterplot_point(tmp.c_str(), value);
        }
        ss.str("");
        ss << "GPU: Device " << d << " Clock Frequency, SOC (MHz)";
        tmp = ss.str();
        value = (double)(pgpu_metrics.current_socclk);
        if(include_event("rsmi", tmp.c_str())) {
            Tau_trigger_userevent(tmp.c_str(), value);
            write_scatterplot_point(tmp.c_str(), value);
        }
        ss.str("");
        ss << "GPU: Device " << d << " GFX Activity (avg)";
        tmp = ss.str();
        value = (double)(pgpu_metrics.average_gfx_activity);
        if(include_event("rsmi", tmp.c_str())) {
            Tau_trigger_userevent(tmp.c_str(), value);
            write_scatterplot_point(tmp.c_str(), value);
        }
        ss.str("");
        ss << "GPU: Device " << d << " Memory Controller Activity (avg)";
        tmp = ss.str();
        value = (double)(pgpu_metrics.average_umc_activity);
        if(include_event("rsmi", tmp.c_str())) {
            Tau_trigger_userevent(tmp.c_str(), value);
            write_scatterplot_point(tmp.c_str(), value);
        }
        ss.str("");
        ss << "GPU: Device " << d << " UVD|VCN Activity (avg)";
        tmp = ss.str();
        value = (double)(pgpu_metrics.average_mm_activity);
        if(include_event("rsmi", tmp.c_str())) {
            Tau_trigger_userevent(tmp.c_str(), value);
            write_scatterplot_point(tmp.c_str(), value);
        }
        ss.str("");
        ss << "GPU: Device " << d << " Throttle Status";
        tmp = ss.str();
        value = (double)(pgpu_metrics.throttle_status);
        if(include_event("rsmi", tmp.c_str())) {
            Tau_trigger_userevent(tmp.c_str(), value);
            write_scatterplot_point(tmp.c_str(), value);
        }
        ss.str("");

/*
Get the performance level of the device with provided device index.
• rsmi_status_t rsmi_dev_perf_level_get (uint32_t dv_ind, rsmi_dev_perf_level_t ∗perf)
Enter performance determinism mode with provided device index.
• rsmi_status_t rsmi_perf_determinism_mode_set (uint32_t dv_ind, uint64_t clkvalue)
Get the overdrive percent associated with the device with provided device index.
• rsmi_status_t rsmi_dev_overdrive_level_get (uint32_t dv_ind, uint32_t ∗od)
Get the list of possible system clock speeds of device for a specified clock type.
• rsmi_status_t rsmi_dev_gpu_clk_freq_get (uint32_t dv_ind, rsmi_clk_type_t clk_type, rsmi_frequencies_t ∗f)
Reset the gpu associated with the device with provided device index.
• rsmi_status_t rsmi_dev_gpu_reset (int32_t dv_ind)
This function retrieves the voltage/frequency curve information.
• rsmi_status_t rsmi_dev_od_volt_info_get (uint32_t dv_ind, rsmi_od_volt_freq_data_t ∗odv)
This function sets the clock range information.
• rsmi_status_t rsmi_dev_clk_range_set (uint32_t dv_ind, uint64_t minclkvalue, uint64_t maxclkvalue, rsmi_clk_type_t clkType)
This function sets the clock frequency information.
• rsmi_status_trsmi_dev_od_clk_info_set(uint32_tdv_ind,rsmi_freq_ind_tlevel,uint64_tclkvalue,rsmi_clk_type_t clkType)
This function sets 1 of the 3 voltage curve points.
• rsmi_status_t rsmi_dev_od_volt_info_set (uint32_t dv_ind, uint32_t vpoint, uint64_t clkvalue, uint64_t volt- value)
This function will retrieve the current valid regions in the frequency/voltage space.
• rsmi_status_t rsmi_dev_od_volt_curve_regions_get (uint32_t dv_ind, uint32_t ∗num_regions, rsmi_freq_volt_region_t ∗buffer)
Get the list of available preset power profiles and an indication of which profile is currently active.
• rsmi_status_t rsmi_dev_power_profile_presets_get (uint32_t dv_ind, uint32_t sensor_ind, rsmi_power_profile_status_t ∗status)
*/

        queried_once[d] = true;
    }
}

void monitor::activateDeviceIndex(uint32_t index) {
    indexMutex.lock();
    activeDeviceIndices.insert(index);
    indexMutex.unlock();
}

double monitor::getAvailableMemory() {
    double avail{0};
    indexMutex.lock();
    // use the copy constructor to get the set of active indices
    std::set<uint32_t> indexSet{activeDeviceIndices};
    indexMutex.unlock();
    /* just check the first known device for now, assume 1 */
    for (uint32_t d : indexSet) {
        uint64_t memory_total;
        RSMI_CALL(rsmi_dev_memory_total_get(d, RSMI_MEM_TYPE_VRAM, &memory_total));
        uint64_t memory_usage;
        RSMI_CALL(rsmi_dev_memory_usage_get(d, RSMI_MEM_TYPE_VRAM, &memory_usage));
        avail = (double)(memory_total - memory_usage);
        break;
    }

    return avail;
}

} // namespace rsmi
} // namespace apex




#endif // #ifdef TAU_ROCM_SMI
