#!/bin/bash

env SOS_CMD_PORT=22502 sosd_stop
env SOS_CMD_PORT=22501 sosd_stop
env SOS_CMD_PORT=20690 sosd_stop
