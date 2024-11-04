#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cfloat>
#include <cmath>
#include "rocm_smi/rocm_smi.h"

#define RSMI_CALL(call)                                                      \
do {                                                                         \
    rsmi_status_t _status = call;                                            \
    if (_status != RSMI_STATUS_SUCCESS) {                                    \
        const char *errstr;                                                  \
        if (rsmi_status_string(_status, &errstr) == RSMI_STATUS_SUCCESS) {   \
        fprintf(stderr, "%s:%d: error: function %s failed with error %d: %s.\n", \
                __FILE__, __LINE__, #call, _status, errstr);                 \
        fprintf(stderr, "\nError: ROCm SMI call failed\n");                 \
        exit(0);                                                                             \
        }                                                                    \
    }                                                                        \
} while (0);

int main(int argc, char * argv[]) {
    rsmi_func_id_iter_handle_t iter_handle, var_iter, sub_var_iter;
    rsmi_func_id_value_t value;
    rsmi_status_t err;
    uint32_t devices{0};
    RSMI_CALL(rsmi_num_monitor_devices(&devices));
    if (devices == 0) {
        std::cerr << "0 Supported RSMI devices found." << std::endl;
        exit(0);                                                                             \
    }

    for (uint32_t i = 0; i < devices; ++i) {
        std::cout << "Supported RSMI Functions:" << std::endl;
        std::cout << "\tVariants (Monitors)" << std::endl;

        RSMI_CALL(rsmi_dev_supported_func_iterator_open(i, &iter_handle));

        while (1) {
            RSMI_CALL(rsmi_func_iter_value_get(iter_handle, &value));
            std::cout << "Function Name: " << value.name << std::endl;

            RSMI_CALL(rsmi_dev_supported_variant_iterator_open(iter_handle, &var_iter));
            if (err != RSMI_STATUS_NO_DATA) {
                std::cout << "\tVariants/Monitors: ";
                while (1) {
                    RSMI_CALL(rsmi_func_iter_value_get(var_iter, &value));
                    if (value.id == RSMI_DEFAULT_VARIANT) {
                        std::cout << "Default Variant ";
                    } else {
                        std::cout << value.id;
                    }
                    std::cout << " (";

                    RSMI_CALL(rsmi_dev_supported_variant_iterator_open(var_iter, &sub_var_iter));
                    if (err != RSMI_STATUS_NO_DATA) {
                        while (1) {
                            RSMI_CALL(rsmi_func_iter_value_get(sub_var_iter, &value));
                            std::cout << value.id << ", ";
                            err = rsmi_func_iter_next(sub_var_iter);
                            if (err == RSMI_STATUS_NO_DATA) {
                                break;
                            }
                        }
                        RSMI_CALL(rsmi_dev_supported_func_iterator_close(&sub_var_iter));
                    }

                    std::cout << "), ";

                    err = rsmi_func_iter_next(var_iter);
                    if (err == RSMI_STATUS_NO_DATA) {
                        break;
                    }
                }
                std::cout << std::endl;
                RSMI_CALL(rsmi_dev_supported_func_iterator_close(&var_iter));
            }
            err = rsmi_func_iter_next(iter_handle);
            if (err == RSMI_STATUS_NO_DATA) {
                break;
            }
        }
        RSMI_CALL(rsmi_dev_supported_func_iterator_close(&iter_handle));
    }
}
