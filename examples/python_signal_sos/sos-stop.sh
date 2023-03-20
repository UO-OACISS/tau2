#!/bin/bash
#DO NOT USE WITH MPI
#MPI needs only one: sosd_stop
env SOS_CMD_PORT=22501 sosd_stop
env SOS_CMD_PORT=20690 sosd_stop
